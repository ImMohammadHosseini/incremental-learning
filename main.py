
"""

"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from data.data_prepare import split_data_points, filenames_and_labels
from data.generator import CustomGenerator
from src.models.studentModel import StudentModel
from src.RL.env import Envirenment
from src.RL.predictEnv import PredictEnvirenment
from src.RL.agentModel import AgentDeepModel
from utils.core import collect_pred, compute_avg_return

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.specs import tensor_spec
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

import xgboost as xgb


#_________________________________________________________________________-
#GPU test
cpu = tf.config.experimental.list_physical_devices('CPU')[0]
gpu = tf.config.experimental.list_physical_devices('GPU')
if len(gpu)>0:
    gpu = gpu[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    #tf.config.experimental.set_visible_devices(cpu)
    print("GPU known")
else :
    print("GPU unknown")

#_________________________________________________________________________-

bert_path= 'pretrained/bertt/smallest'
preprocessor_path= 'pretrained/bertt/preprocessor'

train_path='datasets/AmazonReviewsforSentimentAnalysis'
test_path='datasets/AmazonReviewsforSentimentAnalysis'

batch_size= 128
max_len= 120
class_num= 2
num_eval_episodes= 10
collect_steps_per_iteration= 1
num_iterations= 10000
log_interval= 10 
eval_interval= 10
replay_buffer_max_length = 1000

if __name__=="__main__":
    #split_data_points(train_path, subset='train')
    filenames, labels= filenames_and_labels(train_path, subset='train') 
    
    Da, Db, Dc= filenames[0:100000], filenames[100000:200000], filenames[200000:]
    Da_y, Db_y, Dc_y= labels[0:100000], labels[100000:200000], labels[200000:]
    
    train_env= Envirenment(class_num)
    tf_train_env= tf_py_environment.TFPyEnvironment(train_env)
    action_tensor_spec= tensor_spec.from_spec(tf_train_env.action_spec())
    actions_num= action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    eval_env= Envirenment(class_num)
    tf_eval_env= tf_py_environment.TFPyEnvironment(eval_env)

    pred_env= PredictEnvirenment(class_num)
    tf_pred_env= tf_py_environment.TFPyEnvironment(pred_env)


    rl_deep_model= AgentDeepModel(class_num)

    agent= dqn_agent.DqnAgent(tf_train_env.time_step_spec(),
                              tf_train_env.action_spec(),
                              q_network=rl_deep_model,
                              optimizer=Adam(learning_rate=1e-3),
                              td_errors_loss_fn=common.element_wise_squared_loss,
                              train_step_counter=tf.Variable(0))# m common.ele
    agent.initialize()
    
    x, y, z= 10000, 10, 20
    students= []
    for i in range(z):
        R1_train_rl= []
        R1_test_rl= []
            
        S= StudentModel(preprocessor_path, bert_path, class_num=class_num,
                        max_len=max_len)
        S.compile(Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

        Da_y= [to_categorical(i, num_classes =class_num, dtype='int32') for i in Da_y] 

        Da_train, Da_val, Da_train_y, Da_val_y= train_test_split(Da, Da_y, 
                                                                 test_size=0.1,
                                                                 random_state=1)
        
        train_data= CustomGenerator(directory=train_path+'/alltrain/', 
                                     filenames= Da_train, 
                                     labels= Da_train_y, 
                                     batch_size= batch_size,
                                     max_len= max_len)
        val_data= CustomGenerator(directory=train_path+'/alltrain/', 
                                   filenames= Da_val, 
                                   labels= Da_val_y, 
                                   batch_size= batch_size,
                                   max_len= max_len)

        hist= S.fit_generator(generator= train_data,
                              steps_per_epoch= int(len(Da_train) // batch_size),
                              epochs= 2,
                              validation_data= val_data,
                              workers = 12,
                              validation_steps= int(len(Da_val) // batch_size))

        students.append(S)

        Db_y= [to_categorical(i, num_classes =class_num, dtype='int32') for i in Db_y] 
      
        Db_generator= CustomGenerator(directory=train_path+'/alltrain/', 
                                       filenames= Db,
                                       labels= Db_y,
                                       batch_size= batch_size,
                                       max_len= max_len)


        R1_train_labels= Db_y[0:int(0.7*len(Db_y))]
        R1_test_labels= Db_y[int(0.7*len(Db_y)):]
        
        for student in students:
            temp= student.predict_generator(Db_generator)
            R1_train_rl.append(temp[0:int(0.7*len(temp))])
            R1_test_rl.append(temp[int(0.7*len(temp)):])
            
        train_env.seed(R1_train_rl, R1_train_labels)
        eval_env.seed(R1_test_rl, R1_test_labels)####
        pred_env.seed(R1_test_rl)

        tf_train_env= tf_py_environment.TFPyEnvironment(train_env)
        tf_eval_env= tf_py_environment.TFPyEnvironment(eval_env)####
        tf_pred_env= tf_py_environment.TFPyEnvironment(pred_env)


        agent.train= common.function(agent.train)

        agent.train_step_counter.assign(0)
        
        avg_return= compute_avg_return(tf_eval_env, agent.policy, 
                                       num_eval_episodes, 
                                       len(students))
        returns= [avg_return]
        #timestep= tf_train_env.reset()

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            agent.collect_data_spec,
            batch_size= 1,
            max_length=replay_buffer_max_length)

        rb_observer = [replay_buffer.add_batch]

        dataset= replay_buffer.as_dataset(num_parallel_calls=3,
                                          sample_batch_size=1,
                                          num_steps=2)#.prefetch(3)
        iterator= iter(dataset)

        '''# Create a driver to collect experience.
        collect_driver= py_driver.PyDriver(train_env,
                                       py_tf_eager_policy.PyTFEagerPolicy(
                                           agent.collect_policy, use_tf_function=True),
                                       [rb_observer],
                                       max_steps=collect_steps_per_iteration)'''

        collect_driver= dynamic_step_driver.DynamicStepDriver(tf_train_env,
                                                              agent.collect_policy,
                                                              observers=rb_observer,
                                                              num_steps=1)
        timestep, policy_state= collect_driver.run()
        
        for _ in range(num_iterations):
            timestep, _= collect_driver.run(timestep, maximum_iterations=1)

            experience, unused_info= next(iterator)
            train_loss= agent.train(experience).loss
            step = agent.train_step_counter.numpy()
            if step % log_interval== 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))
            if step % eval_interval == 0:
                avg_return = compute_avg_return(tf_eval_env, 
                                                agent.policy, 
                                                num_eval_episodes, 
                                                len(students))
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

            
        pred_agent= dqn_agent.DqnAgent(tf_pred_env.time_step_spec(),
                                       tf_pred_env.action_spec(),
                                       q_network=agent._q_network,
                                       optimizer=Adam(learning_rate=1e-3))#,
                                  #td_errors_loss_fn=common.element_wise_squared_loss,
                                  #train_step_counter=tf.Variable(0))# m common.ele
        pred_agent.initialize()

        rl_filter= collect_pred(tf_pred_env, pred_agent.policy, 
                                len(R1_test_labels), 
                                len(students))
        
        dtrain= xgb.DMatrix(rl_filter, label=R1_test_labels, missing=np.NaN)

        param = {'max_depth': 15, 'eta': 1, 'objective': 'binary:hinge'}#binary:hinge
        teacher= xgb.train(param, dtrain, 20)#, evallist


        E=[]
        E_y=[]
        while True:
            Dc_i, Dc= Dc[:10000], Dc[10000:]
            Dc_y_i, Dc_y= Dc_y[:10000], Dc_y[10000:]
        
            Dc_i_generator= CustomGenerator(directory=train_path+'/alltrain/', 
                                             filenames= Dc_i,
                                             labels= Dc_y_i,
                                             batch_size= batch_size,
                                             max_len= max_len)
        
            R1_Dc_rl=[]
            for student in students:
                R1_Dc_rl.append(student.predict_generator(Dc_i_generator))
            
            pred_env.seed(R1_Dc_rl)

            tf_pred_env= tf_py_environment.TFPyEnvironment(pred_env)
        
            rl_Dc_i_filter= collect_pred(tf_pred_env, pred_agent.policy, 
                                         len(Dc_y_i), 
                                         len(students))
        
            dpred= xgb.DMatrix(rl_Dc_i_filter, label=Dc_y_i, missing=np.NaN)
            t_pred=teacher.predict(dpred, iteration_range=(0, 10))
        
            for i in range(len(Dc_y_i)):
                if Dc_y_i[i]!=t_pred[i]:
                    E.append(Dc_i[i])
                    E_y.append(Dc_y_i[i])
        
            if len(E) >= x:
                break

        Da += E
        Da_y += E_y

        Db, Dc = Dc[:100000], Dc[100000:]
        Db_y, Dc_y= Dc_y[:100000], Dc_y[100000:]

        #if len(students) >= y:


"""

"""
import numpy as np
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment




class Envirenment(py_environment.PyEnvironment) :
    def __init__(self, class_num) :
        self._action_spec= array_spec.BoundedArraySpec(shape=(), 
                                                       dtype=np.float32, 
                                                       minimum=0.0, 
                                                       maximum=1.0)#tf_agents.specs
        self._observation_spec= array_spec.BoundedArraySpec(shape=(class_num,), 
                                                            dtype=np.float32, 
                                                            minimum=0.0, 
                                                            maximum=1.0)
        self._episode_ended= False
        
        self.stu_preds= None
        self.ground_truth= None
        
        self.state= None
        self.student_num= 0
        self.episod_i= -1
        self.state_i= 0
        self.metric1= tf.keras.metrics.CategoricalAccuracy()
        #self.metric2= tf.keras.metrics.SparseCategoricalAccuracy()
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def seed(self, stu_preds, labels) :
        self.student_num= len(stu_preds)
        self.stu_perds= stu_preds
        self.ground_truth= labels
        self.episod_i= -1 #-1
        self.state_i= 0
        
    def _step(self, action) :
        
        if self._episode_ended:
            return self._reset() ###
        
        self.metric1.reset_state()
        #self.metric2.reset_state()
        self.metric1.update_state(self.ground_truth[self.episod_i%len(self.ground_truth)], self.state)
        #self.metric2.update_state([self.metric1.result().numpy()], [action])

        #reward= self.metric2.result().numpy()
        #if 1 == self.metric2.result().numpy():
        #    reward= 1
        #else :
        #    reward= 0
        if self.metric1.result().numpy() == action:
            reward = 1
        else :
            reward = -1
            
        self.state_i += 1
        if self.state_i < self.student_num :
            self.state= self.stu_perds[self.state_i][self.episod_i%len(self.ground_truth)]
            return ts.transition(np.array(self.state, dtype=np.float32), 
                                 reward=reward,
                                 discount = 0.9)
        else :
            self._episode_ended= True
            return ts.termination(np.array(self.state, dtype=np.float32), 
                                  reward=reward)
    def _reset(self) :
        self._episode_ended= False
        self.episod_i += 1
        self.state_i= 0
        self.state= self.stu_perds[self.state_i][self.episod_i%len(self.ground_truth)]
        return ts.restart(np.array(self.state, dtype=np.float32))
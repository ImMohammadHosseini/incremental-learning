
"""

"""

import numpy as np

def compute_avg_return(environment, policy, num_episodes, stu_num):
    total_return = 0.0
    for _ in range(num_episodes):
        episode_return = 0.0
        time_step = environment.reset()
        i = 0
        while not time_step.is_last():
            i +=1
            action_step = policy.action(time_step)
            
            #print(action_step.action.numpy()[0])
            #if action_step.action.numpy()[0] == 1:
                #print('zero')
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            #print('i %d' %episode_return)
        #print(episode_return)
        total_return += episode_return

    avg_return = total_return / (num_episodes * stu_num)
    return avg_return.numpy()[0]

def collect_pred(environment, policy, num_episodes, stu_num):
    rl_filter=[]
    for _ in range(num_episodes):
        if len(rl_filter) % 10000 == 0 :
            print(len(rl_filter))
        timestep= environment.reset()
        episod_filter=[]
        #while not timestep.is_last():
        for _ in range(stu_num):
            actionstep = policy.action(timestep)
            if actionstep.action.numpy()[0] == 1:
                episod_filter += list(timestep.observation.numpy()[0])
                #episod_filter.append(timestep.observation.numpy()[0])
            else :
                print('not')
                episod_filter += list(np.empty(stu_num) * np.nan)
                #episod_filter.append(np.empty(stu_num) * np.nan)
        rl_filter.append(episod_filter)
    return rl_filter
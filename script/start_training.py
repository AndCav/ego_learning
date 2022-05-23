#!/usr/bin/env python

import gym
import sys
import time
import os
import argparse
import numpy
import random
import qlearn
from gym import wrappers
from functools import reduce
import ego_learning.register_all_env

# ROS packages required
import rospy
import rospkg
import ego_learning.register_all_env
# config param
parser = argparse.ArgumentParser(description="egoD")

# env
parser.add_argument('--num_steps',     type=int,   default=1024,
                    help='Input length to the network for training')
parser.add_argument('--batch_size',     type=int,   default=1,
                    help='Batch size, number of speakers per batch')
parser.add_argument('--fps', type=int,  default=1000,    help='fps')
parser.add_argument('--nDataLoaderThread', type=int,
                    default=5,     help='Number of loader threads')
parser.add_argument('--env_name',      type=str,
                    default="egoStandupEnv-v0", help='env_name')

# Training details
parser.add_argument('--save_interval',  type=int,   default=4,
                    help='Test and save every [test_interval] epochs')
parser.add_argument('--time_horizon',      type=int,
                    default=2000,    help='Maximum number of epochs')
parser.add_argument('--max_episode_length',      type=int,
                    default=1000,    help='Maximum number of episodes')

# Optimizer
parser.add_argument('--optimizer',      type=str,
                    default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,
                    default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float,
                    default=1e-4,  help='Learning rate')
parser.add_argument('--weight_decay',   type=float, default=0,
                    help='Weight decay in the optimizer')
parser.add_argument('--num_epoch',      type=int,
                    default=10,    help='number of epochs of optimize')

# Loss functions
parser.add_argument('--gamma',             type=float,
                    default=0.99,  help='gamma')
parser.add_argument('--gae_param',             type=float,
                    default=0.95,  help='gae_param')
parser.add_argument('--clip',             type=float,
                    default=0.2,  help='clip')
parser.add_argument('--ent_coeff',             type=float,
                    default=1e-4,  help='ent_coeff')
parser.add_argument('--max_grad_norm',             type=float,
                    default=0.5,  help='max_grad_norm')
parser.add_argument('--seed',             type=float, default=1,  help='seed')
parser.add_argument('--iso_sig',  type=bool,   default=True,
                    help='isolated sigma layer')

## Load and save
parser.add_argument('--cont_train',  type=bool,
                    default=False,     help='continues training')
parser.add_argument('--initial_model',  type=int,
                    default=0,     help='old model num')
parser.add_argument('--save_path',      type=str,
                    default="exp", help='Path for model and logs')

# Model definition
parser.add_argument('--inputsize',         type=int,
                    default=15,     help='inputsize')
parser.add_argument('--robot_number',      type=int,
                    default=1,     help='robot number')
parser.add_argument('--hiddensize', nargs='+', type=int,
                    default=[300, 200, 100],  help='hiddensize')
parser.add_argument('--gruhiddensize',   type=int,
                    default=100,  help='Embedding size in the gru layer')

# For test only
parser.add_argument('--mode',           type=str, help='train test demo')

params = parser.parse_args()

if __name__ == '__main__':
    rospy.init_node('ego_learning_gym_test', anonymous=True, log_level=rospy.INFO)

    env = gym.make(params.env_name, n=params.robot_number)
    # Create the Gym environment
    # env = gym.make('ego_learningTrainingEnv-v0')
    rospy.loginfo ( "Gym environment done")
        
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ego_learning')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True) 
    rospy.loginfo ( "Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)


    Alpha = rospy.get_param("/ego/alpha")
    Epsilon = rospy.get_param("/ego/epsilon")
    Gamma = rospy.get_param("/ego/gamma")
    epsilon_discount = rospy.get_param("/ego/epsilon_discount")
    nepisodes = rospy.get_param("/ego/nepisodes")
    nsteps = rospy.get_param("/ego/nsteps")

    qlearn = qlearn.QLearn(actions=env.action_space,
                    alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

 
    for x in range(nepisodes):

        
        cumulated_reward = 0  
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        

        
        observation = env.reset()
        state = ''.join(map(str, observation))
        
 
        for i in range(nsteps):
            

            action = qlearn.chooseAction(state)

            observation, reward, done, info = env.step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))


            qlearn.learn(state, action, reward, nextState)

            if not(done):
                state = nextState
            else:
   
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

        m, s = divmod(int(time.time() - start_time), 18)
        h, m = divmod(m, 18)
        rospy.loginfo ( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()

from __future__ import division
import numpy as np
from numpy import False_
import rospy
import sys
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
import time
import pandas as pd
from matplotlib import pyplot as plt
from collections import deque
import random
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
import pickle as pkl

cmd_vel = "/segway_des_vel"
Imu_topic = "/imu"


def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


class SelfBalance:
    def __init__(self):
        self.cmd_vel = rospy.Publisher(cmd_vel, Twist, queue_size=1)
        self.subscriber = rospy.Subscriber(
            Imu_topic, Imu, callback=self.imu_callback)
        self.reset = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        #self.pose_sub = rospy.Subscriber('/ground_truth/state',Odometry,callback=self.pose_callback)
        self.i = 0
        self.count = 0
        self.vel = Twist()
        self.vel.linear.y = 0
        self.vel.linear.z = 0
        self.vel.angular.x = 0
        self.vel.angular.y = 0
        self.vel.angular.z = 0
        self.y_angle = 0

        # self.x=0
        # self.y=0

    def imu_callback(self, data):
        self.y_angle = data.orientation.y


rospy.init_node('SelfBalance', anonymous=True)

rAll_list = []
gamma = 0.9
lr = 0.7
vels = np.array([-2000, -1000, -500, -250, -100, 100, 250, 500, 1000, 2000])

Robot = SelfBalance()

Robot.reset()
legal_actions = len(vels)
actions = vels.tolist()

num_episodes = 400
epsilon = 1
epsilon_decay = 0.999
memory_size = 1000
batch_size = 100
show = True
angles = np.arange(-0.314, 0.315, 0.314)
legal_actions
scale = 180/3.14
iters = 2000
limit = 0.085*scale


memory = deque(maxlen=memory_size)
memory
Robot.reset()
time.sleep(1)
Robot.pause()
s = Robot.y_angle*scale
Robot.unpause()
a = vels[np.random.randint(0, legal_actions)]

Robot.vel.linear.x = a
Robot.cmd_vel.publish(Robot.vel)
Robot.pause()
s1 = Robot.y_angle*scale
if abs(s1) <= limit:
    r = 1
    experience = (s, r, a, s1)
    s = s1
else:
    r = -100
    Robot.reset()
    experience = (s, r, a, s1)
    s = Robot.y_angle

memory.append(experience)
Robot.unpause()
# empty_memory = False
# if empty_memory:
for _ in range(memory_size):
    a = vels[np.random.randint(0, legal_actions)]
    Robot.vel.linear.x = a
    Robot.cmd_vel.publish(Robot.vel)
    Robot.pause()
    s1 = Robot.y_angle
    if abs(s1) <= limit:
        r = 1
        experience = (s, r, a, s1)
        s = s1
    else:
        r = -100
        Robot.reset()
        experience = (s, r, a, s1)
        s = Robot.y_angle
    memory.append(experience)
    Robot.unpause()

# else:
#     with open("memory.txt") as f:
#         for l in f:
#             memory.append(l)
# print("loaded memory")
print(memory)
if True:
    batches = random.sample(memory, batch_size)
    states = np.array([batch[0] for batch in batches])
    rewards = np.array([batch[1] for batch in batches])
    actions = np.array([batch[2] for batch in batches])
    new_states = np.array([batch[3] for batch in batches])

print("finished generating memories")
model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(1,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(legal_actions, activation='linear'))
model.compile(loss='mean_absolute_error',
              optimizer=adam_v2.Adam(learning_rate=0.05),)
model.summary()
for i in range(num_episodes):
    Robot.reset()
    Robot.pause()
    s = Robot.y_angle*scale

    rAll = 0
    r = 0
    d = False
    j = 0

    for j in range(iters):
        # while True:
        # j=j+1
        # epsilon greedy. to choose random actions initially when Q is all zeros
        if np.random.random() < epsilon:
            a = vels[np.random.randint(0, legal_actions)]
            epsilon = epsilon*epsilon_decay
        else:
            Q = model.predict(np.array([s]))
            a = vels[np.argmax(Q)]
        Robot.vel.linear.x = a
        Robot.unpause()
        Robot.cmd_vel.publish(Robot.vel)
        s1 = Robot.y_angle*scale
        Robot.pause()
        if abs(s1) > limit:
            d = True

        else:
            d = False
            r = 1

        rAll = rAll+r

        if d:
            # time.sleep(1)
            if rAll < (iters-1):
                r = -100
                experience = (s, r, a, s1)
                memory.append(experience)
                if rAll != 0:
                    print("Episode %d Failed! Reward %d" % (i, rAll))
            rAll_list.append((i, rAll))

            break
        experience = (s, r, a, s1)
        memory.append(experience)
        if j >= (iters-1):
            print("Episode %d Passed! Reward %d after full episode" % (i, rAll))
            rAll_list.append((i, rAll))
            break

        s = s1
        #print("State %d"%s)
        Robot.unpause()
    batches = random.sample(memory, batch_size)
    states = np.array([batch[0] for batch in batches])
    rewards = np.array([batch[1] for batch in batches])
    actions = np.array([batch[2] for batch in batches])
    new_states = np.array([batch[3] for batch in batches])
    Qs = model.predict(states)
    new_Qs = model.predict(new_states)
    for i in range(len(rewards)):
        action_index = list(vels).index(actions[i])
        if rewards[i] == -100:
            Qs[i][action_index] = Qs[i][action_index] + lr*(rewards[i])
        else:
            Qs[i][action_index] = Qs[i][action_index] + lr*(rewards[i]+gamma*np.max(new_Qs[i]))
    model.fit(states, Qs, verbose=0)
    #epsilon = epsilon*epsilon_decay
pkl.dump(rAll_list, open('Rewards_with_lr_%f_gamma_%f_DQN.pkl' % (lr, gamma), 'wb'))
model_json = model.to_json()
with open("model2_lr_%f_episodes_%d.json" % (lr, num_episodes), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2_lr_%f_episodes_%d.h5" % (lr, num_episodes))
print("Saved model to disk")
eps = [ep for (ep, _) in rAll_list]
rewards = [reward for (_, reward) in rAll_list]
rewards = moving_average(rewards, 100)
plot = plt.plot(rewards)
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.title('Rewards vs Episodes for Learning Rate %f and Gamma %f' % (lr, gamma))
plt.savefig('Plot_with_lr_%f_gamma_%f_DQN.jpg' % (lr, gamma))

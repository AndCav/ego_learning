#!/usr/bin/env python


import sys

import numpy as np
from numpy import empty, random
import rospy
import std_msgs
import geometry_msgs
import time
import math
import matplotlib.pyplot as plt
from gazebo_msgs.srv import ApplyBodyWrench, BodyRequest


class Disturbance():
    def __init__(self, newSim):

        self.body_name = 'ego_robot::base_link'
        self.reference_frame = 'ego_robot::base_link'
        self.reference_point = geometry_msgs.msg.Point(x=0, y=0, z=0.7)
        self.newSim = newSim
        self.subber = rospy.Subscriber(
            "/ego/K_feed", std_msgs.msg.Float64MultiArray, self.newFeed)

    #def apply_body_wrench_client(body_name, reference_frame, reference_point, wrench, start_time, duration):
        # rospy.wait_for_service('/gazebo/apply_body_wrench')
        # try:
        #     apply_body_wrench = rospy.ServiceProxy(
        #         '/gazebo/apply_body_wrench', ApplyBodyWrench)
        #     apply_body_wrench(body_name, reference_frame, reference_point, wrench,
        #                       start_time, duration)
        #     self.newSim = False

        # except rospy.ServiceException as e:
        #     print("Service call failed: %s %e")

    def clear_body_wrench_client(body_name):
        rospy.wait_for_service('gazebo/clear_body_wrenches')
        try:
            clear_body_wrench = rospy.ServiceProxy(
                'gazebo/clear_body_wrenches', BodyRequest)
            clear_body_wrench(body_name)

        except rospy.ServiceException as e:
            print("Service call failed: %s%e")
            rospy.sleep(1)

    def newFeed(self, data):
        randomicValue = np.sign(random.uniform(-1, 1)) * \
            random.uniform(low=800.0, high=1500.0, size=None)
        newDuration = int(random.uniform(low=0.0, high=3.0, size=None))
        rosDur = rospy.Duration(newDuration)

        wrench = geometry_msgs.msg.Wrench(force=geometry_msgs.msg.Vector3(x=randomicValue,
                                                                          y=0, z=0), torque=geometry_msgs.msg.Vector3(x=0, y=0, z=0))

        start_time = rospy.Time.now()
        # self.apply_body_wrench_client(self.body_name, self.reference_frame, self.reference_point, wrench, start_time, rosDur)

        #----- applying external wrench
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        try:
            apply_body_wrench = rospy.ServiceProxy(
                '/gazebo/apply_body_wrench', ApplyBodyWrench)
            apply_body_wrench(self.body_name, self.reference_frame,
                              self.reference_point, wrench, start_time, rosDur)
            self.newSim = False
            print("mission complete?",self.newSim)
        except rospy.ServiceException as e:
            print("Service call failed: %s%e")
            rospy.sleep(1)
        #----end of external wrench


if __name__ == "__main__":

    rospy.init_node('disturber', anonymous=True)
    last_time = time.time()
    dist = Disturbance(False)
    rate = rospy.Rate(1)  # 10hz

    while not rospy.is_shutdown():
        rate.sleep()

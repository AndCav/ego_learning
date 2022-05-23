from . import robot_gazebo_env
import rospy
import numpy as np
from gym.utils import seeding

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ModelStates
import tf


class Robot():
    def __init__(self, i, displacement_xyz):
        self.i = i
        self.displacement_xyz = displacement_xyz
        # self.ns = "/ego_" + str(self.i)
        self.ns = "/ego"
        self.publisher_list = []
        # The joint_state_controller control no joint but pub the state of all joints
        self.joint_subscriber = rospy.Subscriber('/joint_states', JointState, self.joints_callback)
        self.global_subscriber = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.model_callback)
        self.imu_subscriber = rospy.Subscriber(
            '/imu', Imu, self.imu_callback)
        self.joints = None
        self.global_pos = None
        self.global_vel = None
        self.model_index = None
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

    def model_callback(self, data):
        if self.model_index:
            self.global_pos = data.pose[self.model_index]
            self.global_pos.position.x -= self.displacement_xyz[0] * self.i
            self.global_pos.position.y -= self.displacement_xyz[1] * self.i
            self.global_pos.position.z -= self.displacement_xyz[2] * self.i
            self.global_vel = data.twist[self.model_index]

    def joints_callback(self, data):
        self.joints = data

    def imu_callback(self, data):
        quaternion = (data.orientation.x, data.orientation.y,
                      data.orientation.z, data.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.x_angle = euler[0]
        self.y_angle = euler[1]
        self.z_angle = euler[2]



class egoRobotEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self, **kwargs):
        """Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.
        # namespace
        i = 0
        self.n = kwargs['n']
        self.y_angle = 0
        self.robots = Robot(i, kwargs['displacement_xyz'])
        # inizializzazione controllori
        self.controllers_list = [
            'right_wheel_controller','left_wheel_controller',]
        for n in self.controllers_list[0:]:
            self.robots.publisher_list.append(rospy.Publisher(
                self.robots.ns + '/' + n + '/command', Float64, queue_size=1))

        self.all_controllers_list = []
        for c in self.controllers_list:
            self.all_controllers_list.append(self.robots.ns + '/' + c)
        reset_controls_bool = True
        super(egoRobotEnv, self).__init__(n=self.n, robot_name_spaces='ego',
                                          controllers_list=self.controllers_list,
                                          reset_controls=reset_controls_bool)
        rospy.logdebug("END init EgoRobotEnv")

 # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _check_imu_ready(self):
        self.imu = None
        rospy.logdebug("Waiting for /imu to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/imu", Imu, timeout=5.0)
                rospy.logdebug("Current /imu READY=>")
            except:
                rospy.logerr("Current /imu not ready yet, retrying for getting imu")

        return self.imu

    def _check_egojoint_ready(self):
        joint = None
        self.move_joints([0,0])
        rospy.logdebug("Waiting for /imu to be READY...")
        while joint is None and not rospy.is_shutdown():
            try:
                joint = rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
                rospy.logdebug("Current /ego/joint_state READY=>")

            except:
                rospy.logerr("Current /ego/joint_state not ready yet, retrying for getting joint_state")

        return joint
    

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self.robots.ns = "ego_robot"
        self.robots.joints = None
        while self.robots.joints is None and not rospy.is_shutdown():
            try:
                self.robots.joints = rospy.wait_for_message(
                    '/joint_states', JointState, timeout=3.0)
            except:
                rospy.logerr("Current /joint_states not ready yet.\n\
                    Do you spawn the robot and launch ros_control?")
            try:
                self.robots.model_index = rospy.wait_for_message(
                    '/gazebo/model_states', ModelStates,  timeout=3.0).name.index(self.robots.ns)
            except rospy.exceptions.ROSException:
                rospy.logerr("Robot model does not exist.")
        self._check_egojoint_ready()
        self._check_imu_ready()
        # rospy.logdebug("ALL SYSTEMS READY")
        return True

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        cmd = 0
        self.move_joints(cmd)

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_joints(self, cmd):
        for j in range(len(self.robots.publisher_list)):
            joint_value = Float64()
            joint_value.data = cmd[j]
            self.robots.publisher_list[j].publish(joint_value)

    def obs_states(self):
        return (self.robots.joints, self.robots.global_pos, self.robots.global_vel)


    # def _check_laser_scan_ready(self):
    #     self.laser_scan = None
    #     rospy.logdebug("Waiting for /scan to be READY...")
    #     while self.laser_scan is None and not rospy.is_shutdown():
    #         try:
    #             self.laser_scan = rospy.wait_for_message(
    #                 "/scan", LaserScan, timeout=1.0)
    #             rospy.logdebug("Current /scan READY=>")

    #         except:
    #             rospy.logerr(
    #                 "Current /scan not ready yet, retrying for getting laser_scan")
    #     return self.laser_scan

    # def _odom_callback(self, data):
    #     self.odom = data

    # def _imu_callback(self, data):
    #     self.imu = data

    # def _laser_scan_callback(self, data):
    #     self.laser_scan = data

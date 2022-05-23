from . import robot_gazebo_LQR_env
import rospy
import numpy as np
from gym.utils import seeding
import math
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
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

        print(self.ns)
        self.joint_subscriber = rospy.Subscriber(
            self.ns + '/joint_states', JointState, self.joints_callback)
        self.global_subscriber = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.model_callback,queue_size=1)
        self.imu_subscriber = rospy.Subscriber(
             '/imu', Imu, self.imu_callback)
        # self.joints = None
        # self.global_pos = None
        # self.global_vel = None
        self.model_index = None
        self.posL_des = 0.0
        self.posR_des = 0.0
        self.velL_des = 0.0
        self.velR_des = 0.0
        self.effortL = 0.0
        self.effortR = 0.0

        self.forward_dist = 0.0
        self.forward_v = 0.0

        self.offset_pitch_ = 0.0  # -0.09
        self.forward_des_ = 0.0
        self.dforward_des_ = 0.0
        self.dforward_des_filt_old_ = 0
        self.theta_des_ = 0.0
        self.dtheta_des_ = 0.0
        self.velL = 0
        self.velR = 0
        self.posL = 0
        self.posR = 0
        self.steps = 0
        self.phi = 0  # pitch angle
        self.dphi = 0  # pitch rate
        self.theta = 0
        self.dtheta = 0
        self.uselessdata=False

    def model_callback(self, data):
        # if self.model_index:
        #     self.global_pos = data.pose[self.model_index]
        #     self.global_pos.position.x -= self.displacement_xyz[0] * self.i
        #     self.global_pos.position.y -= self.displacement_xyz[1] * self.i
        #     self.global_pos.position.z -= self.displacement_xyz[2] * self.i
        #     self.global_vel = data.twist[self.model_index]
        if self.model_index:
            if len(data.pose)>1:
    
                global_pos = data.pose[self.model_index]
                global_vel = data.twist[self.model_index]
                orientation_list = [global_pos.orientation.x, global_pos.orientation.y,
                                    global_pos.orientation.z, global_pos.orientation.w]
                (pippo, self.phi, self.theta) = euler_from_quaternion(orientation_list)
                #self.dphi = global_vel.angular.y
                self.dtheta = global_vel.angular.z
                self.forward_dist = math.sqrt(
                    pow(global_pos.position.x, 2)+pow(global_pos.position.y, 2))
                self.forward_v = math.sqrt(
                    pow(global_vel.linear.x, 2)+pow(global_vel.linear.y, 2))
                self.uselessdata=False
            else:
                self.uselessdata=True
                

    def joints_callback(self, data):
        self.posL = data.position[0]
        self.posR = data.position[1]
        self.velL = data.velocity[0]
        self.velR = data.velocity[1]
        self.effortL = data.effort[0]
        self.effortR = data.effort[1]

    def imu_callback(self, data):
        self.dphi=data.angular_velocity.y


class egoRobotEnv_LQR(robot_gazebo_LQR_env.RobotGazeboEnv):

    def __init__(self, **kwargs):
        """Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.
        # namespace
        i = 0
        self.n = kwargs['n']
        self.y_angle = 0
        self.robots = Robot(i, kwargs['displacement_xyz'])
        self.KfeedPub=rospy.Publisher(self.robots.ns +'/K_feed', Float64MultiArray, queue_size=1)
        # inizializzazione controllori
        self.controllers_list = ["joint_state_controller","right_wheel_controller", "left_wheel_controller"]
        # for n in self.controllers_list[0:]:
        #     self.robots.publisher_list.append(rospy.Publisher(
        #         self.robots.ns + '/' + n + '/command', Float64, queue_size=1))

        self.all_controllers_list = []
        # for c in self.controllers_list:
        #     self.all_controllers_list.append(self.robots.ns + '/' + c)
        reset_controls_bool = True
        super(egoRobotEnv_LQR, self).__init__(n=self.n, robot_name_spaces='ego',
                                              controllers_list=self.controllers_list,
                                              reset_controls=reset_controls_bool)
        rospy.logdebug("END init EgoRobotEnv")

 # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _check_imu_ready(self):
        imu = None
        rospy.logdebug("Waiting for /imu to be READY...")
        while imu is None and not rospy.is_shutdown():
            try:
                imu = rospy.wait_for_message("/imu", Imu, timeout=5.0)
                rospy.logdebug("Current /imu READY=>")
            except:
                rospy.logerr(
                    "Current /imu not ready yet, retrying for getting imu")

        return self.imu

    def _check_egojoint_ready(self):
        joint = None
        #self.move_joints([0, 0])
        rospy.logdebug("Waiting for /imu to be READY...")
        while joint is None and not rospy.is_shutdown():
            try:
                joint = rospy.wait_for_message(
                    "/ego/joint_states", JointState, timeout=5.0)
                rospy.logdebug("Current /ego/joint_state READY=>")

            except:
                rospy.logerr(
                    "Current /ego/joint_state not ready yet, retrying for getting joint_state")

        return joint

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self.robots.ns = "ego_robot"
        self.robots.joints = None
        self.robots.model_index=None

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
        # self._check_egojoint_ready()
        # self._check_imu_ready()
        # rospy.logdebug("ALL SYSTEMS READY")
        return True

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        #cmd = [0, 0]
        # self.move_joints(cmd)

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
    # def move_joints(self, cmd):
    #     for j in range(len(self.robots.publisher_list)):
    #         joint_value = Float64()
    #         joint_value.data = cmd[j]
    #         self.robots.publisher_list[j].publish(joint_value)

    def obs_states(self):
        r = self.robots
        return (r.posL, r.velL, r.effortL, r.posR, r.velR, r.effortR, r.phi, r.dphi, r.forward_dist, r.forward_v, r.theta, r.dtheta)

    # def _odom_callback(self, data):
    #     self.odom = data

    # def _imu_callback(self, data):
    #     self.imu = data

    # def _laser_scan_callback(self, data):
    #     self.laser_scan = data

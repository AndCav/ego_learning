from .robot_env import ego_env

from gym import spaces
import numpy as np
import rospy
import rospkg
import rosparam
import os

from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import GetLinkState


def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file)
    path_config_file = os.path.join(config_dir, yaml_file_name)

    paramlist = rosparam.load_file(path_config_file)

    for params, ns in paramlist:
        rosparam.upload_params(ns, params)


class StandupTaskEnv(ego_env.egoRobotEnv):

    # high = numpy.array([np.inf,
    #                     np.inf,
    #                     np.inf,
    #                     np.inf,
    #                     self.max_pitch,
    #                     self.max_yaw,
    #                     self.max_sonar_value])

    # low = numpy.array([self.work_space_x_min,
    #                    self.work_space_y_min,
    #                    self.work_space_z_min,
    #                    -1*self.max_roll,
    #                    -1*self.max_pitch,
    #                    -numpy.inf,
    #                    self.min_sonar_value])
    def __init__(self, **kwargs):
        # Load all the params first
        LoadYamlFileParamsTest("ego_learning", "config", "standup_param.yaml")
        # Variables that we retrieve through the param server, loded when launch training launch.
        self.reward_height_b = rospy.get_param('/ego/reward_height_b')
        self.reward_height_k = rospy.get_param('/ego/reward_height_k')
        self.effort_penalty = rospy.get_param('/ego/effort_penalty')
        self.effort_max = rospy.get_param('/ego/effort_max')
        self.epoch_steps = rospy.get_param('/ego/epoch_steps')
        self.running_step = rospy.get_param('/ego/running_step')

        # self.linear_forward_speed = rospy.get_param('/ego/linear_forward_speed')
        # self.linear_turn_speed = rospy.get_param('/ego/linear_turn_speed')
        # self.angular_speed = rospy.get_param('/ego/angular_speed')
        # self.init_linear_forward_speed = rospy.get_param('/ego/init_linear_forward_speed')
        # self.init_linear_turn_speed = rospy.get_param('/ego/init_linear_turn_speed')

        # self.new_ranges = rospy.get_param('/ego/new_ranges')
        # self.min_range = rospy.get_param('/ego/min_range')
        # self.max_laser_value = rospy.get_param('/ego/max_laser_value')
        # self.min_laser_value = rospy.get_param('/ego/min_laser_value')

        # Get Desired Point to Get
        self.desired_pose = Pose()
        rospy.wait_for_service('gazebo/get_link_state')
        self.get_link_state = rospy.ServiceProxy(
            'gazebo/get_link_state', GetLinkState)

        # We set the reward range, which is not compulsory but here we do it.
        # self.reward_range = (-np.inf, np.inf)
        # Construct the RobotEnv so we know the dimension of cmd
        super(StandupTaskEnv, self).__init__(**kwargs)
        # Only variable needed to be set here
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32)
        self._init_env_variables()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18, 1), dtype=np.float32)  # perchè 60 ??

        rospy.logdebug("END init TestTaskEnv")

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        pass

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.steps = 0
        self.cmd = np.zeros(self.n*2)
        self.x = 0
        self.pitch = 0
        self.yaw = 0
        self.y = 0

    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        self.cmd = self.effort_max * action
        self.move_joints(self.cmd)
        rospy.sleep(self.running_step)
        self.steps += 1


    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        states = self.obs_states()

        obs = np.zeros(18, dtype=float)
        r = self.robots
        joints, global_pos, global_vel = states
        orientation_q = global_pos.orientation
        self.x = global_pos.position.x
        self.x_vel = global_vel.linear.x
        orientation_list = [orientation_q.x,
            orientation_q.y, orientation_q.z, orientation_q.w]
        (self.roll, self.pitch, self.yaw) = (
            euler_from_quaternion(orientation_list))
        buffer = np.concatenate([
            joints.position,
            joints.velocity,
            joints.effort,
            (
                global_pos.position.x,
                global_pos.position.y,
                global_pos.position.z,
                self.roll,
                self.pitch,
                self.yaw,
                global_vel.linear.x,
                global_vel.linear.y,
                global_vel.linear.z,
                global_vel.angular.x,
                global_vel.angular.y,
                global_vel.angular.z
            )
        ])
        obs[0:len(buffer)] = buffer
        return obs


    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        done = self.steps >= self.epoch_steps
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        if(abs(self.pitch) < 0.50):
            reward = (self.reward_height_b - abs(self.pitch)) * \
                self.reward_height_k
            #print("reward from pitch",reward)
            reward += (self.reward_height_b - abs(self.yaw)) * \
                self.reward_height_k
            reward += (0.2 - abs(self.x)) * self.reward_height_k
           # reward -= self.effort_penalty * sum(map(abs, self.robots.joints.effort)) / self.effort_max
           # reward += (0.1 - abs(self.x_vel)) * self.reward_height_k
        else:
            reward = -1

        return reward
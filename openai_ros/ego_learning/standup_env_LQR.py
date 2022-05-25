
from cmath import pi
import numpy
from torch import float32, true_divide
from .robot_env import ego_env_LQR

from gym import spaces
import numpy as np
import rospy
import rospkg
import rosparam
import os
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import GetLinkState
from std_msgs.msg import Float64MultiArray
import roslaunch


def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file)
    path_config_file = os.path.join(config_dir, yaml_file_name)

    paramlist = rosparam.load_file(path_config_file)

    for params, ns in paramlist:
        rosparam.upload_params(ns, params)


class StandupTaskEnv(ego_env_LQR.egoRobotEnv_LQR):

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
    R_ = 0.125
    W_ = 0.55
    N_ = 1/0.31

    def __init__(self, **kwargs):

        # self.package = 'lqr_controller'
        # self.executable = 'lqr_controller'

        # self.node = roslaunch.core.Node(self.package, self.executable)

        self.launch = roslaunch.scriptapi.ROSLaunch()
        self.launch.start()
        self.process = None
        # Load all the params first
        LoadYamlFileParamsTest("ego_learning", "config",
                               "standup_LQR_param.yaml")
        # Variables that we retrieve through the param server, loded when launch training launch.
        self.reward_height_b = rospy.get_param('/ego/reward_height_b')
        self.reward_height_k = rospy.get_param('/ego/reward_height_k')
        self.effort_penalty = rospy.get_param('/ego/effort_penalty')
        self.effort_max = rospy.get_param('/ego/effort_max')
        self.epoch_steps = rospy.get_param('/ego/epoch_steps')
        self.running_step = rospy.get_param('/ego/running_step')
        self.W_pitch = 1.5
        self.W_pitchRate = 15.0
        self.W_yaw = 0.12
        self.W_yawRate = 0.12
        self.W_for = 1
        self.W_forV = 5

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
            low=-1.0, high=1.0, shape=(1, 6), dtype=numpy.float32)
        self._init_env_variables()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12, 1), dtype=numpy.float32)

        rospy.logdebug("END init TestTaskEnv")

    # def process_stop(self):
    #     self.process.stop()

    def kill_launch(self):
        self.launch.shutdown()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        initial_matrix = [-8.0,   -909.8, -67, -168,  40.2236,
                          50.2114, -8.0,   -909.8, -67, -168, -40.2236, -50.2114]

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.steps = 0
        #self.process_restart()
        self.cmd = np.zeros(self.n*2)
        #print("in init_standup_lqr_env")

    def _lqr_run(self, action):
        r = self.robots
        trsh_L_ = 24.5
        trsh_R_ = 24.5

        dt = 0.01
        # r.forward_dist = 0.5*(r.posL + r.posR)
        # r.forward_v = 0.5*(r.velL+r.velR)

        # r.theta = (self.R_ / self.W_) * (r.posR - r.posL)
        # r.dtheta = (self.R_ / self.W_) * (r.velR - r.velL)
        #filtraggio velocità
        # state=np.array([(r.forward_des_ - r.forward_dist),  0.0 - (r.phi + r.offset_pitch_), (dforward_des_filt_ - r.forward_v), 0 - r.dphi, (r.theta_des_ - r.theta), (r.dtheta_des_ - r.dtheta)])
        # state.reshape(6,1)
        # k_feed=np.array([action, [action[0],action[1],action[2],action[3],-action[4],-action[5]]])
        # #print(k_feed)
        # cmd=np.matmul(k_feed,state)
        # cmd.reshape(1,2)
        # cmd[1]+=trsh_L_*np.sign(cmd[1])
        # cmd[0]+=trsh_R_*np.sign(cmd[0])

        # if(abs(cmd[0]) > 100):
        #     cmd[0]=100*np.sign(cmd[0])
        # if(abs(cmd[1]) > 100):
        #     cmd[1]=100*np.sign(cmd[1])

        # cmd = cmd*0.108029313

        return cmd

    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        #self.cmd = self.(effort_max * action
        #cmd=self._lqr_run(action)
        #self.move_joints(self.cmd)
        #print("in set action")
        #action=[10*action[0],1000*action[1],100*action[2],500*action[3],100*action[4],100*action[5]]
        kfeed = Float64MultiArray()

        #-8.5148 - 909.6067 - 15.4836 - 168.1726
     #   -11.5025475025177, -821.4803575456142, -96.33865350484848, -276.7057056427002, 14.917079657173154, 5.821642486000059
        # kfeed.data = [8+50*action[0], -909.8+1000*action[1], 100 *
        #                           action[2], 500*action[3], 100*action[4], 100*action[5]]
        kfeed.data = [-8.0+float(8*action.flatten()[0]), -909.8+float(909*action.flatten()[1]), -67+float(
            67*action.flatten()[2]), -168+float(168*action.flatten()[3]),  40.2236+float(40*action.flatten()[4]), 50.2114+float(50*action.flatten()[5])]
        # kfeed.data = [-8.0,   -909.8, -67, -168,  40.2236,
        #              50.2114, -8.0,   -909.8, -67, -168, -40.2236, -50.2114]
        self.KfeedPub.publish(kfeed)
        # rospy.set_param('/ego/k_feed', [float(80*action[0]), float(500*action[1]),
        #                 float(100*action[2]), float(250*action[3]), float(100*action[4]), float(100*action[5])])

        #self.process_stop()
        #self.process_restart()
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

        #print("in get obs")
        obs = np.zeros((12, 1), dtype=numpy.float32)
        (posL, velL, effortL, posR, velR, effortR, phi, dphi,
         forward_dist, forward_v, theta, dtheta) = states
        #print("phi {:.2f}, dphi {:.2f}, theta {:.2f},dtheta {:.2f}, forward_dist {:.2f} , forward_v {:.2f}".format(r.phi, r.dphi, r.theta, r.dtheta, r.forward_dist, r.forward_v))
        # buffer = [posL, velL, effortL/11, posR, velR, effortR/11,
        #           phi/pi, dphi, forward_dist, forward_v, theta/(2*pi), dtheta]
        obs[0] = float(posL)
        obs[1] = float(velL)
        obs[2] = float(effortL)
        obs[3] = float(posR)
        obs[4] = float(velR)
        obs[5] = float(effortR)
        obs[6] = float(phi)
        obs[7] = float(dphi)
        obs[8] = float(forward_dist)
        obs[9] = float(forward_v)
        obs[10] = float(theta)
        obs[11] = float(dtheta)
        # obs.reshape(12,1)
        return obs

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """

        done = self.steps >= self.epoch_steps
        # if done and abs(self.robots.phi)<0.9:
        #     print("k_feed = ")
        #     kfeed= [float(10*action[0]), float(1000*action[1]),
        #               float(100*action[2]), float(500*action[3]), float(100*action[4]), float(100*action[5])]
        #     print(kfeed)
        return done

    # def needsreset(self):
    #     if(abs(r.phi) > 1):
    #         self.reset_world_proxy()
    #         return True
    #     else:
    #         return False

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        r = self.robots

        if(abs(r.phi) < 1):
            if(r.uselessdata == False):
                reward = (self.reward_height_b - abs(r.phi)) * self.W_pitch
                reward += (self.reward_height_b - abs(r.dphi)) * \
                    self.W_pitchRate

                reward += (0.5 - abs(r.theta)) * self.W_yaw
                reward += (1 - abs(r.dtheta)) * self.W_yawRate

                reward += (0.01 - abs(r.forward_dist)) * self.W_for
                reward += (1 - abs(r.forward_v)) * self.W_forV
            else:
                reward = 0

           # reward -= self.effort_penalty * sum(map(abs, self.robots.joints.effort)) / self.effort_max
           # reward += (0.1 - abs(self.x_vel)) * self.reward_height_k
        else:
            if(r.uselessdata == False):
                # reward = (self.reward_height_b - abs(r.phi)) * self.W_pitch
                # reward += (0.01 - abs(r.forward_dist)) * self.W_for
                reward = -1
            else:
                reward = 0

        #print("is in reward")
        return reward

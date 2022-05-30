import os
import sys
import rospy
import gym
from gym.utils import seeding
from std_msgs.msg import Float64
from .gazebo_connection_LQR import GazeboConnection
from .controllers_connection_LQR import ControllersConnection
# https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
from ego_learning.msg import RLExperimentInfo

# https://github.com/openai/gym/blob/master/gym/core.py


class RobotGazeboEnv(gym.GoalEnv):

    def __init__(self, n, robot_name_spaces, controllers_list, reset_controls, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION"):

        # To reset Simulations
        rospy.logdebug("START init RobotGazeboEnv")
        self.gazebo = GazeboConnection(
            start_init_physics_parameters, reset_world_or_sim)
        self.controllers_objects = [ControllersConnection(namespace=robot_name_spaces[i],
                                                          controllers_list=controllers_list)
                                    for i in range(n)]
        self.reset_controls = True
        self.seed()

        # Set up ROS related variables
        self.episode_num = 0
        self.cumulated_episode_rewards = 0  # [0]*n
        self.reward_pub = rospy.Publisher(
            '/openai/reward', RLExperimentInfo, queue_size=1)

        # We Unpause the simulation and reset the controllers if needed
        """
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        """
        self.gazebo.unpauseSim()
        # if self.reset_controls:
        #     for o in self.controllers_objects:
        #         o.reset_controllers()
        self.right_pub = rospy.Publisher(
            "/ego/right_wheel_controller/command", Float64, queue_size=1)
        self.left_pub = rospy.Publisher(
            "/ego/left_wheel_controller/command", Float64, queue_size=1)
        self._check_all_systems_ready()
        self.gazebo.pauseSim()
        rospy.logdebug("END init RobotGazeboEnv")

    # Env methods
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        # rospy.logdebug("START STEP OpenAIROS")

        self.gazebo.unpauseSim()
        self._set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        rewards = self._compute_reward(obs, done)
        # for i in range(len(rewards)): #se vogliamo più di un robot
        self.cumulated_episode_rewards += rewards

        # rospy.logdebug("END STEP OpenAIROS")

        return obs, rewards, done, info

    def reset(self):
        # rospy.logdebug("Reseting RobotGazeboEnvironment")
        self._reset_sim()
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        # rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        Publishes the cumulated reward of the episode and
        increases the episode number by one.
        :return:
        """
        # rospy.loginfo("PUBLISHING REWARD...")
        # for i in range(len(self.cumulated_episode_rewards)):
        self._publish_reward_topic(
            0, self.cumulated_episode_rewards, self.episode_num)
        self.cumulated_episode_rewards = 0
        #rospy.loginfo("PUBLISHING REWARD...DONE="+str(self.cumulated_episode_reward)+",EP="+str(self.episode_num))

        self.episode_num += 1

    def _publish_reward_topic(self, robot_id, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.robot_id = robot_id
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation in this way:
            controller.switch_off()
            gazebo.deleteModel_()
            gazebo.pauseSim()
            gazebo.reset_simulation_proxy()
            gazebo.respawn()
            rospy.sleep(2)
            controller.load_controller()
            rospy.sleep(0.5)
            gazebo.unpauseSim()
            controller.switch_on()
        """
        rospy.logdebug("RESET SIM START")
        if self.reset_controls:
            # rospy.logdebug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            #print("switching off")
            # apparentemente è inutile perchè una volta eliminato il robot i controllori muoiono
            self.controllers_objects[0].switch_off()

            self.gazebo.deleteModel_()

            self.gazebo.resetSim()

            self.gazebo.pauseSim()
            self.gazebo.respawn()
            rospy.sleep(4)
            self.controllers_objects[0].load_controller()
            rospy.sleep(1)
            rospy.wait_for_service("/ego/controller_manager/switch_controller")
            self.gazebo.unpauseSim()
            
            self.controllers_objects[0].switch_on()
           # self.left_pub.publish(0)
           # self.right_pub.publish(0.0)
            
            # self._check_all_systems_ready()
            
            self.gazebo.pauseSim()


            # self.gazebo.resetWorld()


        else:
            # rospy.logwarn("DONT RESET CONTROLLERS")
            self.gazebo.resetSim()
            self._set_init_pose()
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()
        # self.gazebo.resetSim()

        # rospy.logdebug("RESET SIM END")
        return True

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

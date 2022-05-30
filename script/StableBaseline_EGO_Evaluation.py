import gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import CheckpointCallback

# import time
# from datetime import datetime

import ego_learning.register_all_env
import rospy

memory_path = "/home/cp/Desktop/egolearning_history/SAC_ego_smallDisturbance"

if __name__ == '__main__':
    # Create environment
    interruption = False
    rospy.init_node('ego_gyb_SAC', anonymous=True, log_level=rospy.INFO)
    env_name = "egoStandupEnv_LQR-v0"
    env = gym.make(env_name)
    print(env_name)
    desired_final_episode = 1000
    memory_path += "/rl_model_"+str(desired_final_episode)+"_steps"
    model = SAC.load(memory_path, env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=500)

    # Enjoy trained agent
    obs = env.reset()

    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

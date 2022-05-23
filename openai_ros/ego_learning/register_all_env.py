from gym.envs.registration import register

register(
        id='egoStandupEnv-v0',
        entry_point='ego_learning.standup_env:StandupTaskEnv',
        kwargs={'n': 1, 
                'displacement_xyz': [0, 10, 0]}
    )

register(
        id='egoStandupEnv_LQR-v0',
        entry_point='ego_learning.standup_env_LQR:StandupTaskEnv',
        kwargs={'n': 1, 
                'displacement_xyz': [0, 10, 0]}
    )

from . import offline_env, atari_envs

def get_replay_dataset(env_name, replay_type, subsample):
    """
        Load Atari replay dataset: remember to set ATARI_DATASET_DIR correctly.

        Args:
            env_name: name of the Atari game. (e.g. 'breakout')
            replay_type: 'initial1', 'initial10', 'mixed10', 'expert1'
            subsample: subsample the dataset by this factor.
        Returns:
            offline_env.Dataset object (from which you can call dataset.sample(batch_size=64) e.g.)
    """
    if replay_type == 'initial1':
        epochs = [0]
    elif replay_type == 'initial10':
        epochs = [i for i in range(5)]
    elif replay_type == 'mixed10':
        epochs = [i * 10 for i in range(5)]
    elif replay_type == 'expert1':
        epochs = [49]
    else:
        raise ValueError('Unknown replay type: {}'.format(replay_type))
    
    data_dict = offline_env.get_dataset(
        game=atari_envs.capitalize_game_name(env_name),
        index=1,
        epochs=epochs,
        stack=True,
        subsample=subsample,
    )
    return offline_env.Dataset(data_dict)

def get_youtube_video_dataset(env_name, augment=True):
    """
        Load Youtube video dataset: remember to set ATARI_VIDEODATASET_DIR correctly.

        Args:
            env_name: name of the Atari game. (e.g. 'breakout')
            augment: Whether to do random cropping.
        Returns:
            offline_env.Dataset object (from which you can call dataset.sample(batch_size=64) e.g.)
    """
    if augment:
        data_dict = offline_env.get_video_dataset_with_augmentation(
            game=env_name,
        )
    else:
        data_dict = offline_env.get_video_dataset(
            game=env_name,
        )
    return offline_env.Dataset(data_dict)

def make_env(env_name, add_monitor=True):
    import gym
    env = gym.make(f'{env_name}-epoch-1-v0', stack=True, sticky=True)
    if add_monitor:
        from jaxrl_m.evaluation import EpisodeMonitor
        env = EpisodeMonitor(env)
    return env
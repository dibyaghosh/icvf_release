import numpy as np
import gym
import cv2

from gym import spaces
from gym.wrappers import AtariPreprocessing, TransformReward, FrameStack
from .offline_env import OfflineEnv


def capitalize_game_name(game):
    game = game.replace('-', '_')
    return ''.join([g.capitalize() for g in game.split('_')])


class AtariEnv(gym.Env):
    def __init__(self,
                 game,
                 stack=False,
                 sticky_action=False,
                 clip_reward=False,
                 terminal_on_life_loss=False,
                 **kwargs):
        # set action_probability=0.25 if sticky_action=True
        env_id = '{}NoFrameskip-v{}'.format(game, 0 if sticky_action else 4)

        # use official atari wrapper
        env = AtariPreprocessing(gym.make(env_id),
                                 terminal_on_life_loss=terminal_on_life_loss)

        if stack:
            env = FrameStack(env, num_stack=4)

        if clip_reward:
            env = TransformReward(env, lambda r: np.clip(r, -1.0, 1.0))

        self._env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def render(self, mode='human'):
        self._env.render(mode)

    def seed(self, seed=None):
        super().seed(seed)
        self._env.seed(seed)


class OfflineAtariEnv(AtariEnv, OfflineEnv):
    def __init__(self, **kwargs):
        game = capitalize_game_name(kwargs['game'])
        del kwargs['game']
        AtariEnv.__init__(self, game=game, **kwargs)
        OfflineEnv.__init__(self, game=game, **kwargs)

from gym.envs.registration import register

# list from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
for game in [
        'adventure', 'air-raid', 'alien', 'amidar', 'assault', 'asterix',
        'asteroids', 'atlantis', 'bank-heist', 'battle-zone', 'beam-rider',
        'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede',
        'chopper-command', 'crazy-climber', 'defender', 'demon-attack',
        'double-dunk', 'elevator-action', 'enduro', 'fishing-derby', 'freeway',
        'frostbite', 'gopher', 'gravitar', 'hero', 'ice-hockey', 'jamesbond',
        'journey-escape', 'kangaroo', 'krull', 'kung-fu-master',
        'montezuma-revenge', 'ms-pacman', 'name-this-game', 'phoenix',
        'pitfall', 'pong', 'pooyan', 'private-eye', 'qbert', 'riverraid',
        'road-runner', 'robotank', 'seaquest', 'skiing', 'solaris',
        'space-invaders', 'star-gunner', 'tennis', 'time-pilot', 'tutankham',
        'up-n-down', 'venture', 'video-pinball', 'wizard-of-wor',
        'yars-revenge', 'zaxxon'
]:

    for index in range(5):
        register(id='{}-mixed-v{}'.format(game, index),
                 entry_point='d4rl_atari.envs:OfflineAtariEnv',
                 max_episode_steps=108000,
                 kwargs={
                     'game': game,
                     'index': index + 1,
                     'start_epoch': 1,
                     'last_epoch': 1,
                 })

        register(id='{}-medium-v{}'.format(game, index),
                 entry_point='d4rl_atari.envs:OfflineAtariEnv',
                 max_episode_steps=108000,
                 kwargs={
                     'game': game,
                     'index': index + 1,
                     'start_epoch': 10,
                     'last_epoch': 10
                 })

        register(id='{}-expert-v{}'.format(game, index),
                 entry_point='d4rl_atari.envs:OfflineAtariEnv',
                 max_episode_steps=108000,
                 kwargs={
                     'game': game,
                     'index': index + 1,
                     'start_epoch': 50,
                     'last_epoch': 50
                 })

    for index in range(1):
        for epoch in range(1):
            register(id='{}-epoch-{}-v{}'.format(game, epoch + 1, index),
                     entry_point=OfflineAtariEnv,
                     max_episode_steps=108000,
                     kwargs={
                         'game': game,
                         'index': index + 1,
                         'start_epoch': epoch + 1,
                         'last_epoch': epoch + 1,
                     })
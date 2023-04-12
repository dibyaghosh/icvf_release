import numpy as np
import os
import gym
import gzip

from os.path import expanduser
from subprocess import Popen

URI = 'gs://atari-replay-datasets/dqn/{}/{}/replay_logs/'
BASE_DIR = os.environ.get('ATARI_DATASET_DIR', os.path.join(expanduser('~'), 'nfs', 'atari'))
BASE_VIDEO_DIR = os.environ.get('ATARI_VIDEODATASET_DIR', os.path.join(expanduser('~'), 'nfs', 'video_atari'))


def get_dir_path(env, index, epoch, base_dir=BASE_DIR):
    return os.path.join(base_dir, env, str(index), str(epoch))

def inspect_dir_path(env, index, epoch, base_dir=BASE_DIR):
    path = get_dir_path(env, index, epoch, base_dir)
    if not os.path.exists(path):
        return False
    for name in ['observation', 'action', 'reward', 'terminal']:
        if not os.path.exists(os.path.join(path, name + '.gz')):
            return False
    return True


def _download(name, env, index, epoch, dir_path):
    file_name = '$store$_{}_ckpt.{}.gz'.format(name, epoch)
    uri = URI.format(env, index) + file_name
    path = os.path.join(dir_path, '{}.gz'.format(name))
    print('downloading {} to {}'.format(uri, path))
    p = Popen(['gsutil', '-m', 'cp', '-R', uri, path])
    p.wait()
    return path


def _load(name, dir_path):
    path = os.path.join(dir_path, name + '.gz')
    with gzip.open(path, 'rb') as f:
        print('loading {}...'.format(path))
        return np.load(f, allow_pickle=False)

def download_dataset(env, index, epoch, base_dir=BASE_DIR):
    dir_path = get_dir_path(env, index, epoch, base_dir)
    _download('observation', env, index, epoch, dir_path)
    _download('action', env, index, epoch, dir_path)
    _download('reward', env, index, epoch, dir_path)
    _download('terminal', env, index, epoch, dir_path)


def _stack(observations, terminals, n_channels=4):
    rets = []
    t = 1
    for i in range(observations.shape[0]):
        if t < n_channels:
            padding_shape = (n_channels - t, ) + observations.shape[1:]
            padding = np.zeros(padding_shape, dtype=np.uint8)
            observation = observations[i - t + 1:i + 1]
            observation = np.vstack([padding, observation])
        else:
            # avoid copying data
            observation = observations[i - n_channels + 1:i + 1]

        rets.append(observation)

        if terminals[i]:
            t = 1
        else:
            t += 1
    return rets

def distance_from_beginning(terminals):
    dists = [1]
    for i in range(1, len(terminals)):
        if terminals[i-1]:
            dists.append(1)
        else:
            dists.append(dists[-1] + 1)
    return np.array(dists)

def get_dataset(game, index, epochs, stack=False, subsample=None):
    observation_stack = []
    action_stack = []
    reward_stack = []
    terminal_stack = []
    distance_stack = []
    for epoch in epochs:
        path = get_dir_path(game, index, epoch)
        if not inspect_dir_path(game, index, epoch):
            os.makedirs(path, exist_ok=True)
            download_dataset(game, index, epoch)

        observations = _load('observation', path)
        actions = _load('action', path)
        rewards = _load('reward', path)
        terminals = _load('terminal', path)

        # sanity check
        assert observations.shape == (1000000, 84, 84)
        assert actions.shape == (1000000, )
        assert rewards.shape == (1000000, )
        assert terminals.shape == (1000000, )


        distances = distance_from_beginning(terminals)

        if subsample is not None:
            N = int(1000000 * subsample)
            observations = observations[:N]
            actions = actions[:N]
            rewards = rewards[:N]
            terminals = terminals[:N]
            distances = distances[:N]

        observation_stack.append(observations)
        action_stack.append(actions)
        reward_stack.append(rewards)
        terminal_stack.append(terminals)
        distance_stack.append(distances)

    if len(observation_stack) > 1:
        observations = np.vstack(observation_stack)
        actions = np.vstack(action_stack).reshape(-1)
        rewards = np.vstack(reward_stack).reshape(-1)
        terminals = np.vstack(terminal_stack).reshape(-1)
        distances = np.vstack(distance_stack).reshape(-1)
    else:
        observations = observation_stack[0]
        actions = action_stack[0]
        rewards = reward_stack[0]
        terminals = terminal_stack[0]
        distances = distance_stack[0]

    # memory-efficient stacking
    if stack:
        observations = np.lib.stride_tricks.sliding_window_view(observations, 4, 0)

    print('Final buffer shape: ', observations.shape, actions.shape, rewards.shape, terminals.shape, distances.shape)

    masks = 1.0 - terminals
    data_dict = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals,
        'distances': distances,
        'masks': masks
    }
    return data_dict

def get_video_dataset(game, atari_head=False):
    if atari_head:
        path = os.path.join(BASE_VIDEO_DIR, f'../atari_head_{game.lower()}.npz')
    else:
        path = os.path.join(BASE_VIDEO_DIR, f'{game.lower()}.npz')
    print('loading {}...'.format(path))
    data = np.load(path)
    observations = data['observations'].astype(np.uint8)
    terminals = data['terminals']
    distances = distance_from_beginning(terminals)
    observations = np.lib.stride_tricks.sliding_window_view(observations, 4, 0)
    masks = 1.0 - terminals

    data_dict = {
        'observations': observations,
        'terminals': terminals,
        'distances': distances,
        'masks': masks
    }
    return data_dict

def get_video_dataset_with_augmentation(game, atari_head=False):
    if atari_head:
        path = os.path.join(BASE_VIDEO_DIR, f'../atari_head_{game.lower()}.npz')
    else:
        path = os.path.join(BASE_VIDEO_DIR, f'{game.lower()}.npz')

    # path = os.path.join(BASE_VIDEO_DIR, f'{game.lower()}.npz')
    print('loading {}...'.format(path))
    data = np.load(path)
    observations = data['observations'].astype(np.uint8)
    terminals = data['terminals']
    distances = distance_from_beginning(terminals)
    observations = np.pad(observations, ((0, 0), (8, 8), (8, 8)), 'edge')
    print('After padding: ', observations.shape)
    observations = np.lib.stride_tricks.sliding_window_view(observations, 4, 0)

    masks = 1.0 - terminals
    data_dict = {
        'observations': observations,
        'terminals': terminals,
        'distances': distances,
        'masks': masks
    }
    return data_dict


class OfflineEnv(gym.Env):
    def __init__(self,
                 game=None,
                 index=None,
                 start_epoch=None,
                 last_epoch=None,
                 stack=False,
                 **kwargs):
        super(OfflineEnv, self).__init__()
        self.game = game
        self.index = index
        self.start_epoch = start_epoch
        self.last_epoch = last_epoch
        self.stack = stack

    def get_dataset(self):
        return get_dataset(self.game, self.index, range(self.start_epoch, self.last_epoch + 1), self.stack)

class Dataset:
    def __init__(self, data_dict):
        self.dataset = data_dict

    def get_observations(self, idxs):
        """
        Returns:
            observations: (batch_size, 84, 84, 4) (does not mask out invalid observations)
            valids: (batch_size, 4) (boolean mask of valid observations)

            do `observations * valids[:, None, None, :]` to get the correctly masked observations

        """
        observations = self.dataset['observations'][idxs]
        distances = self.dataset['distances'][idxs + 3]
        valids = np.stack([distances-3, distances-2, distances-1, distances], axis=-1) > 0
        observations = observations # * valids[:, None, None, :]
        if observations.shape[-2] != 84: # Need to crop
            padding = (observations.shape[-2] - 84) // 2
            crop_x = np.random.randint(0, padding * 2)
            crop_y = np.random.randint(0, padding * 2)
            observations = observations[:, crop_x:crop_x+84, crop_y:crop_y+84, :]
        return observations, valids

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, len(self.dataset['observations']) - 1, batch_size)
        
        observations, valids = self.get_observations(idxs)
        next_observations, next_valids = self.get_observations(idxs + 1)

        batch = {
            k: v[idxs + 3] for k, v in self.dataset.items() if k != 'observations'
        }
        batch['observations'] = observations
        batch['next_observations'] = next_observations
        batch['valids'] = valids
        batch['next_valids'] = next_valids
        return batch

    def sample_gc(self, batch_size, idxs=None, same_sg=0.1):
        if idxs is None:
            idxs = np.random.randint(0, len(self.dataset['observations']) - 1, batch_size)
        
        goal_idxs = np.clip(idxs + 1 + np.random.choice(100, batch_size), 0, len(self.dataset['observations']) - 1)
        same_goal = np.random.rand(batch_size) < same_sg
        goal_idxs = np.where(same_goal, idxs, goal_idxs)    

        observations, valids = self.get_observations(idxs)
        next_observations, next_valids = self.get_observations(idxs + 1)
        goal_observations, goal_valids = self.get_observations(goal_idxs)

        batch = {
            k: v[idxs + 3] for k, v in self.dataset.items() if k != 'observations'
        }
        batch['observations'] = observations
        batch['next_observations'] = next_observations
        batch['goals'] = goal_observations
        batch['valids'] = valids
        batch['next_valids'] = next_valids
        batch['goal_valids'] = goal_valids

        batch['rewards'] = same_goal.astype(np.float32)
        return batch
    
    def sample_gcz(self, batch_size, idxs=None, same_sg=0.1, same_gz=0.1):
        if idxs is None:
            idxs = np.random.randint(0, len(self.dataset['observations']) - 1, batch_size)
        
        goal_dists = np.random.choice(100, batch_size)
        goal_idxs = np.clip(idxs + 1 + goal_dists, 0, len(self.dataset['observations']) - 1)
        same_goal = np.random.rand(batch_size) < same_sg
        goal_idxs = np.where(same_goal, idxs, goal_idxs)

        intent_goal_dists = np.random.choice(100, batch_size)
        intent_goal_idxs = np.clip(goal_idxs + intent_goal_dists, 0, len(self.dataset['observations']) - 1)
        same_intent_goal = np.random.rand(batch_size) < same_gz
        intent_goal_idxs = np.where(same_intent_goal, goal_idxs, intent_goal_idxs)

        observations, valids = self.get_observations(idxs)
        next_observations, next_valids = self.get_observations(idxs + 1)
        goal_observations, goal_valids = self.get_observations(goal_idxs)
        intent_goal_observations, intent_goal_valids = self.get_observations(intent_goal_idxs)

        batch = {
            k: v[idxs + 3] for k, v in self.dataset.items() if k != 'observations'
        }

        batch['observations'] = observations
        batch['next_observations'] = next_observations
        batch['goals'] = goal_observations
        batch['desired_goals'] = intent_goal_observations
        batch['valids'] = valids
        batch['next_valids'] = next_valids
        batch['goal_valids'] = goal_valids
        batch['intent_goal_valids'] = intent_goal_valids

        batch['rewards'] = (goal_idxs == idxs).astype(np.float32) - 1.0
        batch['desired_rewards'] = (intent_goal_idxs == idxs).astype(np.float32) - 1.0

        batch['masks'] = (goal_idxs != idxs).astype(np.float32)
        batch['desired_masks'] =  (intent_goal_idxs != idxs).astype(np.float32)

        return batch
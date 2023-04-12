from jaxrl_m.dataset import Dataset
import dataclasses
import numpy as np
import jax
import ml_collections

@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    terminal: bool = True
    max_distance: int = None
    curr_goal_shift: int = 0

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
            'max_distance': ml_collections.config_dict.placeholder(int),
            'curr_goal_shift': 0,
        })

    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.dataset.size-self.curr_goal_shift, size=batch_size)
        
        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        if self.max_distance is not None:
            final_state_indx = np.clip(final_state_indx, 0, indx + self.max_distance)
            
        distance = np.random.rand(batch_size)
        middle_goal_indx = np.round(((indx) * distance + final_state_indx * (1- distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)
        
        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx+self.curr_goal_shift], self.dataset['observations'])

        return batch

@dataclasses.dataclass
class GCSDataset(GCDataset):
    p_samegoal: float = 0.5
    intent_sametraj: bool = False

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
            'p_samegoal': 0.5,
            'intent_sametraj': False,
            'max_distance': ml_collections.config_dict.placeholder(int),
            'curr_goal_shift': 0,
        })

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        if self.intent_sametraj:
            desired_goal_indx = self.sample_goals(indx, p_randomgoal=0.0, p_trajgoal=1.0 - self.p_currgoal, p_currgoal=self.p_currgoal)
        else:
            desired_goal_indx = self.sample_goals(indx)
        
        goal_indx = self.sample_goals(indx)
        goal_indx = np.where(np.random.rand(batch_size) < self.p_samegoal, desired_goal_indx, goal_indx)

        success = (indx == goal_indx)
        desired_success = (indx == desired_goal_indx)

        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        batch['desired_rewards'] = desired_success.astype(float) * self.reward_scale + self.reward_shift
        
        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
            batch['desired_masks'] = (1.0 - desired_success.astype(float))
        
        else:
            batch['masks'] = np.ones(batch_size)
            batch['desired_masks'] = np.ones(batch_size)
        
        goal_indx = np.clip(goal_indx + self.curr_goal_shift, 0, self.dataset.size-1)
        desired_goal_indx = np.clip(desired_goal_indx + self.curr_goal_shift, 0, self.dataset.size-1)
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])
        batch['desired_goals'] = jax.tree_map(lambda arr: arr[desired_goal_indx], self.dataset['observations'])

        return batch


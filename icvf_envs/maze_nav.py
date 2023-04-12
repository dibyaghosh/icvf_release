def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

install_and_import('shapely')
install_and_import('plotly')
install_and_import('seaborn')

import math
import numpy as np

import gym
from gym import spaces

from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
import seaborn as sns

from shapely.geometry import LineString
from shapely.geometry import Point

# import logging
# logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

import plotly.graph_objects as go

from matplotlib import collections as mc
import plotly.express as px


colors = sns.color_palette("colorblind", 16)

class Viz:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(3, 3), dpi=200)

    def visualize(self, trajs, goals=None):
        s_lst = [traj["observations"] for traj in trajs]
        self.ax.clear()

        if goals is None:
            for s, color in zip(s_lst, colors):

                self.ax.scatter(s[0, 0], s[0, 1], marker="o", color=color)
                self.ax.plot(s[:, 0], s[:, 1], color=color)
        else:
            for s, goal, color in zip(s_lst, goals, colors):

                self.ax.scatter(s[0, 0], s[0, 1], marker="o", color=color)
                self.ax.scatter(goal[0], goal[1], marker="x", color=color)
                self.ax.plot(s[:, 0], s[:, 1], color=color)

        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-3.5, 3.5)
        
        return -(self.fig)
    
    def visualize_s(self, observations):
        self.ax.clear()

        self.ax.hist2d(observations[:, 0], observations[:, 1], bins=(50, 50))

        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-3.5, 3.5)
        
        return mpl_to_plotly(self.fig)

    def base_visualize(self, init_pos, goal_pos):
        self.ax.clear()
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-3.5, 3.5)
        # self.ax.scatter(init_pos[0], init_pos[1], marker="o", s=200, color="xkcd:sky blue", label="start")
        # self.ax.scatter(goal_pos[0], goal_pos[1], marker="X", s=200, color="xkcd:leaf green", label="goal")
        # self.ax.legend()
        return mpl_to_plotly(self.fig)
        



class MazeNav(gym.Env):
    def __init__(self, target_r=0.1, safe_radius=0.1, radius=0.1, random_reset=False, random_goal=False, goal_observable=False, 
            dt=0.1, init_pos=(0., 0.), goal_pos=(2.5, 2.5), 
            obsts=[
                [(-3., -3.), (-3., 3.)], 
                [(3., -3.), (3., 3.)],
                [(-3., -3.), (3., -3.)],
                [(-3., 3.), (3., 3.)],
                [(1.0, -2.), (1.0, 2.)],
                [(2.0, -3.), (2.0, -0.5)],
                [(2.0, 0.5), (2.0, 3.)]],
            safe_obsts=[]):
        
        assert 0.0 < target_r <= 1.0
        self.n = 2
        self.target_r = target_r
        self.radius = radius
        self.safe_radius = safe_radius
        self.dt = dt
        self.init_pos = np.array(init_pos)
        self.goal_pos = np.array(goal_pos)
        self.action_space = spaces.Box(-np.ones(self.n), np.ones(self.n), dtype=np.float32)
        if goal_observable:
            self.observation_space = spaces.Box(-np.ones(self.n * 3) * 3., np.ones(self.n * 3) * 3., dtype=np.float32)
        else:
            self.observation_space = spaces.Box(-np.ones(self.n * 2) * 3., np.ones(self.n * 2) * 3., dtype=np.float32)

        self.obsts = obsts
        self.safe_obsts = safe_obsts
        self.random_reset = random_reset
        self.random_goal = random_goal

        self.goal_observable = goal_observable

        self.viz = Viz()

    @property
    def context(self):
        return self.goal_pos.copy()

    @property
    def context_dim(self):
        return 2

    def _compute_reward(self, new_p, goal_pos):

        if self._collide(new_p, self.safe_radius, self.safe_obsts):
            reward = -1.01
        else:
            reward = 0.
        
        reward += (np.linalg.norm(goal_pos - new_p, axis=-1) <= self.target_r).astype(float)

        return reward

    def rdm_fn(self, observations, actions, next_observations, contexts):

        batch_size = observations.shape[0]
        if self.safe_radius <= self.radius:
            rewards = self._compute_reward(next_observations[:, :self.n], contexts[:, :self.n])
            return rewards, \
                rewards > 0., \
                rewards <= 0.,
                
        rs, ds, ms = [], [], []
        for observation, action, next_observation, context in zip(observations, actions, next_observations, contexts):
            r = self._compute_reward(next_observation[:self.n], context[:self.n])
            d = False
            m = True
            rs.append(r)
            ds.append(d)
            ms.append(m)
        return np.array(rs), np.array(ds), np.array(ms)

    def draw_obsts(self, fig):
        for obst in self.safe_obsts:
            dx, dy = (-(obst[1][1] - obst[0][1]), obst[1][0] - obst[0][0])
            d = (dx ** 2 + dy ** 2) ** 0.5
            dx, dy = dx / d * self.safe_radius / 2., dy / d * self.safe_radius / 2.

            trace = go.Scatter(
                x=[obst[0][0] + dx, obst[0][0] - dx, obst[1][0] - dx, obst[1][0] + dx, obst[0][0] + dx], 
                y=[obst[0][1] + dy, obst[0][1] - dy, obst[1][1] - dy, obst[1][1] + dy, obst[0][1] + dy], 
                fill="toself", fillcolor="red", line_color="red", marker_line_width=1, marker_size=1
            )

            fig.add_trace(trace)
            
            fig.add_shape({
                "type": "circle",
                "x0": obst[0][0] - self.safe_radius / 2.,
                "y0": obst[0][1] - self.safe_radius / 2.,
                "x1": obst[0][0] + self.safe_radius / 2.,
                "y1": obst[0][1] + self.safe_radius / 2.,
                "fillcolor": "red",
                "line_color":"red",
            })
            fig.add_shape({
                "type": "circle",
                "x0": obst[1][0] - self.safe_radius / 2.,
                "y0": obst[1][1] - self.safe_radius / 2.,
                "x1": obst[1][0] + self.safe_radius / 2.,
                "y1": obst[1][1] + self.safe_radius / 2.,
                "fillcolor": "red",
                "line_color":"red",
            })

        for obst in self.obsts:
            dx, dy = (-(obst[1][1] - obst[0][1]), obst[1][0] - obst[0][0])
            d = (dx ** 2 + dy ** 2) ** 0.5
            dx, dy = dx / d * self.safe_radius / 2., dy / d * self.safe_radius / 2.

            trace = go.Scatter(
                x=[obst[0][0] + dx, obst[0][0] - dx, obst[1][0] - dx, obst[1][0] + dx, obst[0][0] + dx], 
                y=[obst[0][1] + dy, obst[0][1] - dy, obst[1][1] - dy, obst[1][1] + dy, obst[0][1] + dy], 
                    fill="toself", fillcolor="black", line_color="black", marker_line_width=1, marker_size=1
            )

            fig.add_trace(trace)

            fig.add_shape({
                "type": "circle",
                "x0": obst[0][0] - self.radius / 2.,
                "y0": obst[0][1] - self.radius / 2.,
                "x1": obst[0][0] + self.radius / 2.,
                "y1": obst[0][1] + self.radius / 2.,
                "fillcolor": "black",
                "line_color":"black",
            })
            fig.add_shape({
                "type": "circle",
                "x0": obst[1][0] - self.radius / 2.,
                "y0": obst[1][1] - self.radius / 2.,
                "x1": obst[1][0] + self.radius / 2.,
                "y1": obst[1][1] + self.radius / 2.,
                "fillcolor": "black",
                "line_color":"black",
            })


    def visualize_s(self, states, **kwargs):
        fig = go.Figure(go.Histogram2d(
            x=states[:, 0],
            y=states[:, 1],
            autobinx=False,
            xbins=dict(start=-3, end=3, size=0.1),
            autobiny=False,
            ybins=dict(start=-3, end=3, size=0.1),
            colorscale='YlGnBu',
        ))
        self.draw_obsts(fig)
        return fig

    def visualize(self, *args, **kwargs):
        fig = self.viz.visualize(*args, **kwargs)
        self.draw_obsts(fig)
        return fig

    def visualize_vs(self, vs_img):
        fig = px.imshow(vs_img,
            zmin=0., zmax=100., origin="lower") # , extent=extent)
        # self.draw_obsts(fig)
        return fig

    def base_visualize(self):
        fig = self.viz.base_visualize(self.init_pos, self.goal_pos)
        self.draw_obsts(fig)
        fig.update_layout(
            margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=0, #top margin
            ), 
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.update_xaxes(visible=False, showticklabels=False)
        return fig

    def sample_goal(self):
        rnd_p = (np.random.rand(2) - 0.5) * 6.
        while self._collide(rnd_p, self.radius, self.obsts):
            rnd_p = (np.random.rand(2) - 0.5) * 6.
        return rnd_p

    def context_sampler(self):
        return self.sample_goal()

    def reset(self, goal=None, random_init=False):
        if goal is not None:
            goal = tuple(goal)
            assert len(goal) == 2 and type(goal[0]) == float and type(goal[1]) == float
            self.goal_pos = np.array(goal)
        else:
            if self.random_goal:
                self.goal_pos = self.sample_goal()

        ob = np.concatenate((np.random.randn(self.n) * 0.1, np.random.randn(self.n) * 0.1), axis=-1)
        ob[:self.n] += self.init_pos

        if random_init or self.random_reset:
            rnd_p = self.sample_goal()
            ob[:self.n] = rnd_p
            ob[self.n:] = (np.random.rand(self.n) - 0.5) * 0.1
        else:
            ob = np.clip(ob, -1., 1.)
        
        ob[self.n:] = ob[self.n:] / (max(np.linalg.norm(ob[self.n:]), 1.0))
        self._ob = ob
        if self.goal_observable:
            ob = np.concatenate([self._ob.copy(), self.goal_pos], axis=-1)
            # print(ob.shape)
            return ob
        return self._ob.copy()

    def _check_obst(self, new_p, obst, radius):
        o1, o2 = obst[0], obst[1]

        new_p = Point(*new_p)
        c = new_p.buffer(radius).boundary
        l = LineString([tuple(o1), tuple(o2)])

        i = c.intersection(l)

        return i.is_empty

    def _collide(self, p, radius, obsts):
        for obst in obsts:
            if not self._check_obst(p, obst, radius):
                return True
        return False

    def step(self, a):
        a = a / (max(np.linalg.norm(a) / 0.5, 1.0))
        p, v = self._ob[:self.n], self._ob[self.n:]
        
        dv = self.dt * a
        new_v = v + dv
        new_v = new_v / (max(np.linalg.norm(a), 1.0))

        dp = ((v + new_v) / 2.) * self.dt
        new_p = p + dp
        
        if self._collide(new_p, self.radius, self.obsts):
            new_v = np.zeros_like(v)
            new_p = p

        reward = self._compute_reward(new_p, self.goal_pos)

        self._ob = np.concatenate([new_p, new_v], axis=-1)

        if reward > 0.:
            done = True
        else:
            done = False

        if self.goal_observable:
            return np.concatenate([self._ob.copy(), self.goal_pos], axis=-1), reward, done, {"is_success": reward > 0.}
        
        return self._ob.copy(), reward, done, {"is_success": reward > 0.}


obst_conf_test = [   
    [(-3., -3.), (-3., 3.)], 
    [(3., -3.), (3., 3.)],
    [(-3., -3.), (3., -3.)],
    [(-3., 3.), (3., 3.)],
]

obst_conf = [   
    [(-3., -3.), (-3., 3.)], 
    [(3., -3.), (3., 3.)],
    [(-3., -3.), (3., -3.)],
    [(-3., 3.), (3., 3.)],
    [(0.0, 3.), (2.0, 0.5)],
    [(0.0, -3.), (2.0, -0.5)]]

safe_obst_conf = [
    [(1.0, 0.25), (1.0, -0.25)],
    [(1.25, 1.5), (2.0, 1.5)]
]
safe_obst_conf_hard = [
    [(1.0, 0.35), (1.0, -0.35)],
    [(1.25, 1.5), (2.5, 1.5)]
]

goal = (1.5, 2.5)

obst_conf_A = [   
    [(-3., -3.), (-3., 3.)], 
    [(3., -3.), (3., 3.)],
    [(-3., -3.), (3., -3.)],
    [(-3., 3.), (3., 3.)],
    [(1.0, -3.), (1.0, 2.)],
    [(2.0, -3.), (2.0, -0.5)],
    [(2.0, 0.5), (2.0, 3.)],
    [(0.0, 3.), (0.0, 0.5)],
    [(0.0, -3.), (0.0, -0.5)]]

obst_conf_B = [   
    [(-3., -3.), (-3., 3.)], 
    [(3., -3.), (3., 3.)],
    [(-3., -3.), (3., -3.)],
    [(-3., 3.), (3., 3.)],
    [(1.0, -2.), (1.0, 3.)],
    [(2.0, -3.), (2.0, -0.5)],
    [(2.0, 0.5), (2.0, 3.)],
    [(0.0, 3.), (0.0, 0.5)],
    [(0.0, -3.), (0.0, -0.5)]]


gym.envs.register(
    id="toy-simple-v1",
    entry_point=MazeNav,
    kwargs={
        "radius": 0.10,
        "target_r": 0.20,
        "random_reset": False,
        "obsts": obst_conf,
        "goal_pos": goal,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

gym.envs.register(
    id="toy-simple-rr-v1",
    entry_point=MazeNav,
    kwargs={
        "radius": 0.10,
        "target_r": 0.20,
        "random_reset": True,
        "obsts": obst_conf,
        "goal_pos": goal,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)


gym.envs.register(
    id="toy-simple-rr-rg-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.20,
        "radius": 0.10,
        "safe_radius": 0.10,
        "random_reset": True,
        "random_goal": True,
        "goal_observable": True,
        "obsts": obst_conf,
        "goal_pos": goal,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

gym.envs.register(
    id="toy-test-rr-rg-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.25,
        "radius": 0.25,
        "safe_radius": 0.25,
        "random_reset": True,
        "random_goal": True,
        "goal_observable": True,
        "obsts": obst_conf_test,
        "goal_pos": goal,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

gym.envs.register(
    id="toy-test-ft-rr-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.25,
        "radius": 0.25,
        "safe_radius": 0.25,
        "random_reset": True,
        "random_goal": False,
        "goal_observable": False,
        "obsts": obst_conf_test,
        "goal_pos": goal,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

gym.envs.register(
    id="toy-simple-ft-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.20,
        "radius": 0.10,
        "safe_radius": 0.10,
        "random_reset": False,
        "random_goal": False,
        "goal_observable": False,
        "obsts": obst_conf,
        "safe_obsts": safe_obst_conf,
        "goal_pos": goal,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)


gym.envs.register(
    id="toy-simple-ft-rr-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.20,
        "radius": 0.10,
        "safe_radius": 0.10,
        "random_reset": True,
        "random_goal": False,
        "goal_observable": False,
        "obsts": obst_conf,
        "safe_obsts": safe_obst_conf,
        "goal_pos": goal,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)
gym.envs.register(
    id="toy-simple-ft-hard-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.20,
        "radius": 0.10,
        "safe_radius": 0.10,
        "random_reset": False,
        "random_goal": False,
        "goal_observable": False,
        "obsts": obst_conf,
        "safe_obsts": safe_obst_conf_hard,
        "goal_pos": goal,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

gym.envs.register(
    id="toy-simple-ft-hard-rr-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.20,
        "radius": 0.10,
        "safe_radius": 0.10,
        "random_reset": True,
        "random_goal": False,
        "goal_observable": False,
        "obsts": obst_conf,
        "safe_obsts": safe_obst_conf_hard,
        "goal_pos": goal,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

gym.envs.register(
    id="toy-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.20,
        "obsts": obst_conf_A,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

gym.envs.register(
    id="toy-rr-v1",
    entry_point=MazeNav,
    kwargs={
        "radius": 0.10,
        "target_r": 0.20,
        "random_reset": True,
        "obsts": obst_conf_A,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)


gym.envs.register(
    id="toy-rr-v1",
    entry_point=MazeNav,
    kwargs={
        "radius": 0.10,
        "target_r": 0.20,
        "random_reset": True,
        "obsts": obst_conf_A,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

gym.envs.register(
    id="toy-rr-B-v1",
    entry_point=MazeNav,
    kwargs={
        "radius": 0.10,
        "target_r": 0.20,
        "random_reset": True,
        "obsts": obst_conf_B,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

gym.envs.register(
    id="toy-rr-rg-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.20,
        "radius": 0.10,
        "safe_radius": 0.10,
        "random_reset": True,
        "random_goal": True,
        "goal_observable": True,
        "obsts": obst_conf_A,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)


gym.envs.register(
    id="toy-rr-tg-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.20,
        "radius": 0.10,
        "safe_radius": 0.10,
        "random_reset": True,
        "random_goal": False,
        "goal_observable": True,
        "obsts": obst_conf_A,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)


gym.envs.register(
    id="toy-ft-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.20,
        "radius": 0.10,
        "safe_radius": 0.20,
        "random_reset": False,
        "random_goal": False,
        "goal_observable": False,
        "obsts": obst_conf_A,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

gym.envs.register(
    id="toy-ft-rr-v1",
    entry_point=MazeNav,
    kwargs={
        "target_r": 0.20,
        "radius": 0.10,
        "random_reset": True,
        "random_goal": False,
        "goal_observable": False,
        "safe_radius": 0.15,
        "obsts": obst_conf_A,
    },
    max_episode_steps=100,
    reward_threshold=1000.,
)

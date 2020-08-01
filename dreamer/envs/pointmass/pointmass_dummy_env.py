import numpy as np
from gym import utils
from envs import mujoco_env
import os
import tensorflow as tf
from gym import spaces

class Pointmass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self.seeding = False
        self.env_timestep = 0
        self.env_name = 'pointmass'

        # counter
        self.counter = 0
        self.x = 0
        self.y = 0
        self.reset_goal = np.array([0.5, 0.5, 0])

        # dummy
        self.wall_x_min = 0
        self.wall_x_max = 0
        self.wall_y_min = 0
        self.wall_y_max = 0

        self.goal_dim = 2 # (shouldn't be changed at all during planning)

        # placeholder
        self.target_sid = -1

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/pointmass.xml', 5)
        utils.EzPickle.__init__(self)
        self.observation_dim = 4
        self.action_dim = 2

        self.target_sid = self.model.site_name2id("target")
        self.skip = self.frame_skip

        # action and obs space
        low = -1*np.ones((self.action_dim,))
        high = 1.0*np.ones((self.action_dim,))
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        high = 1.0*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # counter
        self.counter = 0
        self.x = 0
        self.y = 0


    #################################


    def _get_obs(self):

        # new position
        self.x += 0.1
        self.y += 0.1

        # update counter
        self.counter += 1

        return np.array([
            self.x, #[1]
            self.y, #[1]
            self.reset_goal[0],
            self.reset_goal[1],
        ])


    #################################


    def _step(self, a):

        # self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()

        reward, done = self.get_reward(ob, a)

        score = self.get_score(ob)

        # finalize step
        env_info = {'ob': ob,
                    'rewards': self.reward_dict,
                    'score': score}

        return ob, reward, done, env_info


    #################################


    def get_score(self, obs):
        pos = obs[:-self.goal_dim]
        target_pos = obs[-self.goal_dim:]
        score = -1*np.abs(pos-target_pos)
        return score


    def get_reward(self, observations, actions=None):

        #initialize and reshape as needed, for batch mode
        self.reward_dict = {}
        if(len(observations.shape)==1):
            observations = np.expand_dims(observations, axis = 0)
            batch_mode = False
        else:
            batch_mode = True

        #get vars
        pos = observations[:, :-self.goal_dim]
        target_pos = observations[:, -self.goal_dim:]

        #calc rew
        dist = np.linalg.norm(pos - target_pos, axis=1)
        self.reward_dict['r_total'] = -10*dist

        #done is always false for this env
        dones = np.zeros((observations.shape[0],))

        #return
        if(not batch_mode):
            return self.reward_dict['r_total'][0], dones[0]
        return self.reward_dict['r_total'], dones


    def tf_get_residual(self, observations):

        scale_rew_residual = 0.1

        pos = observations[:, :-self.goal_dim]
        target_pos = observations[:, -self.goal_dim:]

        residual = pos - target_pos

        return scale_rew_residual*residual


    #################################


    def reset(self):
        _ = self.reset_model()

        observation, _reward, done, _info = self._step(np.zeros(2))
        ob = self._get_obs()

        return ob

    def reset_model(self, seed=None):

        # unused
        self.reset_pose = np.array([0,0,0])
        self.reset_vel = 0.0*self.init_qvel.copy()
        self.reset_goal = np.array([0.5, 0.5, 0])

        return self.do_reset(self.reset_pose, self.reset_vel, self.reset_goal)

    def do_reset(self, reset_pose, reset_vel, reset_goal):

        # reset counter
        self.counter = 0

        self.x = 0
        self.y = 0

        #reset target (not used)
        self.reset_goal = reset_goal.copy()
        self.model.site_pos[self.target_sid] = self.reset_goal

        #return
        return self._get_obs()

import numpy as np
from gym import utils
from envs import mujoco_env
import os
import tensorflow as tf
from gym import spaces
from mujoco_py import MjViewer

MAKE_OBSTACLES = False

FIXED_START = False #### True
FIXED_GOAL = False 

INCLUDE_VEL = False

class PointmassHard(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self.seeding = False
        self.env_timestep = 0
        self.env_name = 'pointmass' ## keep same plotting/etc properties as normal pointmass

        self.goal_dim = 2 # (shouldn't be changed at all during planning)

        # placeholder
        self.target_sid = -1
        self.pm_sid = -1

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/pointmass_hard.xml', 5)
        utils.EzPickle.__init__(self)

        if INCLUDE_VEL:
            self.observation_dim = 6
        else:
            self.observation_dim = 4
        self.action_dim = 2

        self.target_sid = self.model.site_name2id("target")
        self.pm_sid = self.model.site_name2id("site_pm")
        self.skip = self.frame_skip

        # action and obs space
        low = -1*np.ones((self.action_dim,))
        high = 1.0*np.ones((self.action_dim,))
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        high = 1.0*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)


    #################################


    def _get_obs(self):

        curr_goal = self.model.site_pos[self.target_sid][:self.goal_dim]

        # curr_site_x = self.data.site_xpos[self.pm_sid][0]
        # curr_site_y = self.data.site_xpos[self.pm_sid][1]

        if INCLUDE_VEL:
            return np.concatenate([
                self.data.qpos.flat, #[2]
                self.data.qvel.flat, #[2]
                curr_goal, #[2]
            ])
        else:
            return np.concatenate([
                self.data.qpos.flat, #[2]
                curr_goal, #[2]
            ])

    #################################


    def _step(self, a):

        self.do_simulation(a, self.frame_skip)
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
        pos = obs[:2]
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
        pos = observations[:, :2]
        target_pos = observations[:, -self.goal_dim:]

        #calc rew
        dist = np.linalg.norm(pos - target_pos, axis=1)
        self.reward_dict['r_total'] = -10*dist

        #done is always false for this env
        dones = np.zeros((observations.shape[0],))==1

        #return
        if(not batch_mode):
            return self.reward_dict['r_total'][0], dones[0]
        return self.reward_dict['r_total'], dones


    def tf_get_residual(self, observations, reward_res_weight=0.001):

        pos = observations[:, :2]
        target_pos = observations[:, -self.goal_dim:]

        residual = pos - target_pos

        return reward_res_weight*residual


    #################################


    def reset(self):
        _ = self.reset_model()

        observation, _reward, done, _info = self._step(np.zeros(2))
        ob = self._get_obs()

        return ob

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)

        ########## reset position

        if FIXED_START:
            self.reset_pose = self.init_qpos.copy()
        else:
            self.reset_pose = self.init_qpos.copy()

            which_region= np.random.randint(2)
            if which_region==0:
                self.reset_pose[0] = np.random.uniform(low=-0.6, high=-0.05) #x
                self.reset_pose[1] = np.random.uniform(low=-0.8, high=0.6) #y
            else:
                self.reset_pose[0] = np.random.uniform(low=0.07, high=0.85) #x
                self.reset_pose[1] = np.random.uniform(low=-0.8, high=-0.65) #y

        
        ########## reset vel (0)

        self.reset_vel = 0.0*self.init_qvel.copy()

        ########## reset goal

        if FIXED_GOAL:
            ## start in top right
            self.reset_goal = np.array([0.5, 0.5, 0])
        else:
            ## random somewhere in right side
            goal_x = np.random.uniform(0.45, 0.85)
            goal_y = np.random.uniform(-0.3, 0.65)
            self.reset_goal = np.array([goal_x, goal_y, 0])

        return self.do_reset(self.reset_pose, self.reset_vel, self.reset_goal)

    def do_reset(self, reset_pose, reset_vel, reset_goal):

        self.set_state(reset_pose, reset_vel)

        #reset target
        self.reset_goal = reset_goal.copy()
        self.model.site_pos[self.target_sid] = self.reset_goal
        self.sim.forward()
        
        #return
        return self._get_obs()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent * 1.2

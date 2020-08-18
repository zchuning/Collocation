import atexit
import functools
import sys
import threading
import traceback

import gym
import numpy as np
from PIL import Image

class SawyerPushXYEnv:

  LOCK = threading.Lock()

  def __init__(self, rand_init_goal, action_repeat):
    from mujoco_py import MjRenderContext
    import metaworld.envs.mujoco.sawyer_xyz as sawyer
    with self.LOCK:
      self._env = sawyer.SawyerPushEnvV2()

    self._rand_init_goal = rand_init_goal
    self._action_repeat = action_repeat
    self._width = 64
    self._size = (self._width, self._width)
    self._action_space = gym.spaces.Box(
        np.array([-1, -1]),
        np.array([+1, +1]),
    )

    self._offscreen = MjRenderContext(self._env.sim, True, 0, 'egl', True)
    self._offscreen.cam.azimuth = 205
    self._offscreen.cam.elevation = -165
    self._offscreen.cam.distance = 2.6
    self._offscreen.cam.lookat[0] = 1.1
    self._offscreen.cam.lookat[1] = 1.1
    self._offscreen.cam.lookat[2] = -0.1


  @property
  def observation_space(self):
    shape = self._size + (3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    goal_space = gym.spaces.Box(low=self._env.goal_low, high=self._env.goal_high)
    return gym.spaces.Dict({'image': img_space, 'goal': goal_space})

  @property
  def action_space(self):
    return self._action_space

  def get_goal(self):
    return self._env.goal

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      self._env.hand_init_pos = np.array([0, 0.4, 0.02])
      if self._rand_init_goal:
        # self._env.goal = np.random.uniform(
        #     np.array([-0.3, 0.3, 0.02]),
        #     np.array([0.3, 0.9, 0.02]),
        #     size=(self._env.goal_space.low.size),
        # )
        self._env.goal = np.random.uniform(
            self._env.goal_space.low,
            self._env.goal_space.high,
            size=(self._env.goal_space.low.size),
        )
      self._env.reset()
    return self._get_obs()

  def step(self, action):
    total_reward = 0.0
    action_padded = np.zeros(4)
    action_padded[:2] = action
    for step in range(self._action_repeat):
      _, reward, done, info = self._env.step(action_padded)
      total_reward += reward
      if done:
        break
    obs = self._get_obs()
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode)

  def _get_obs(self):
    self._offscreen.render(self._width, self._width, -1)
    image = np.flip(self._offscreen.read_pixels(self._width, self._width)[0], 1)
    goal = self.get_goal()
    return {'image': image, 'goal': goal}


class MetaWorld:

  LOCK = threading.Lock()

  def __init__(self, name, random_init, action_repeat):
    from mujoco_py import MjRenderContext
    import metaworld.envs.mujoco.sawyer_xyz as sawyer
    domain, task = name.split('_', 1)
    with self.LOCK:
      if task == 'SawyerReachEnv':
        self._env = sawyer.SawyerReachPushPickPlaceEnv(task_type='reach')
      else:
        self._env = getattr(sawyer, task)()

    self._random_init = random_init
    self._action_repeat = action_repeat
    self._width = 64
    self._size = (self._width, self._width)

    self._offscreen = MjRenderContext(self._env.sim, True, 0, 'egl', True)
    self._offscreen.cam.azimuth = 205
    self._offscreen.cam.elevation = -165
    self._offscreen.cam.distance = 2.6
    self._offscreen.cam.lookat[0] = 1.1
    self._offscreen.cam.lookat[1] = 1.1
    self._offscreen.cam.lookat[2] = -0.1

  @property
  def observation_space(self):
    shape = self._size + (3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      if self._random_init:
        self._env.hand_init_pos = np.random.uniform(
            self._env.hand_low,
            self._env.hand_high,
            size=self._env.hand_low.size
        )
      state = self._env.reset()
    return self._get_obs(state)

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state)
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode)

  def _get_obs(self, state):
    self._offscreen.render(self._width, self._width, -1)
    image = np.flip(self._offscreen.read_pixels(self._width, self._width)[0], 1)
    return {'image': image, 'state': state}



class DeepMindControl:

  def __init__(self, name, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class Collect:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    transition['action'] = np.zeros(self._env.action_space.shape)
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ActionRepeat:

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      obs, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return obs, total_reward, done, info


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs

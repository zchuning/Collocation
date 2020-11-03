import atexit
import functools
import sys
import threading
import traceback
import os

import gym
import numpy as np
from PIL import Image

if 'MUJOCO_RENDERER' in os.environ:
  RENDERER = os.environ['MUJOCO_RENDERER']
else:
  RENDERER = 'glfw'


class DreamerEnv():
  LOCK = threading.Lock()

  def __init__(self, action_repeat, width=64):
    self._action_repeat = action_repeat
    self._width = width
    self._size = (self._width, self._width)

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


class DeskEnv(DreamerEnv):
  def __init__(self, task=None, action_repeat=1, width=64):
    super().__init__(action_repeat, width=width)
    from envs.franka_desk.franka_desk import FrankaDesk
    with self.LOCK:
      env_params = {
        'viewer_image_height': self._width,
        'viewer_image_width': self._width,
        'textured': True,
      }
      self._env = FrankaDesk(env_params)

  def _get_obs(self, state):
    image = self._env.render()
    state['image'] = image
    return state

  
class DreamerMujocoEnv(DreamerEnv):
  def __init__(self, task=None, action_repeat=1):
    super().__init__(action_repeat)
    from mujoco_py import MjRenderContext
    from envs.pointmass.pointmass_env import Pointmass
    from envs.push.push_env import Push
    from envs.pointmass.pointmass_hard_env import PointmassHard
    # from envs.pointmass.pointmass_smart_env import Pointmass as PointmassSmart
    with self.LOCK:
      if task == 'pm_obstacle':
        self._env = Pointmass()
      elif task == 'pm_push':
        self._env = Push()

    self._offscreen = MjRenderContext(self._env.sim, True, 0, RENDERER, True)
    self._offscreen.cam.azimuth = 0
    self._offscreen.cam.elevation = 90
    self._offscreen.cam.distance = 2.6
    self._offscreen.cam.lookat[0] = 0
    self._offscreen.cam.lookat[1] = 0
    self._offscreen.cam.lookat[2] = 0


class KitchenEnv(DreamerEnv):
  def __init__(self, task=None, action_repeat=1):
    super().__init__(action_repeat)
    import d4rl
    with self.LOCK:
      if task == 'kitchen_microwave':
        self._env = KitchenMicrowave(ref_min_score=0.0, ref_max_score=1.0)
      elif task == 'kitchen_cabinet':
        self._env = KitchenCabinet(ref_min_score=0.0, ref_max_score=1.0)
      else:
        self._env = gym.make(task)

  def _get_obs(self, state):
    image = self.render('rgb_array')
    return {'image': image, 'state': state}


# Define additional Kitchen tasks
try:
  import d4rl
  from d4rl.kitchen.kitchen_envs import KitchenBase
  
  class KitchenMicrowave(KitchenBase):
      TASK_ELEMENTS = ['microwave']

  class KitchenCabinet(KitchenBase):
    TASK_ELEMENTS = ['slide cabinet']
except:
  pass


class MetaWorld(DreamerEnv):
  def __init__(self, name, action_repeat):
    super().__init__(action_repeat)
    from mujoco_py import MjRenderContext
    import metaworld.envs.mujoco.sawyer_xyz as sawyer
    domain, task = name.split('_', 1)
    with self.LOCK:
      if task == 'SawyerReachEnv':
        self._env = sawyer.SawyerReachPushPickPlaceEnv(task_type='reach')
      else:
        self._env = getattr(sawyer, task)()

    self._action_repeat = action_repeat
    self._width = 64
    self._size = (self._width, self._width)

    self._offscreen = MjRenderContext(self._env.sim, True, 0, RENDERER, True)
    self._offscreen.cam.azimuth = 205
    self._offscreen.cam.elevation = -165
    self._offscreen.cam.distance = 2.6
    self._offscreen.cam.lookat[0] = 1.1
    self._offscreen.cam.lookat[1] = 1.1
    self._offscreen.cam.lookat[2] = -0.1

  # TODO remove this. This has to be inside dreamer, but the argument is hidden inside wrappers unfortunately...
  def reset(self):
    # self._env.hand_init_pos = np.array([0, .6, .1])
    return super().reset()

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      total_reward += min(reward, 100000)
      if done:
        break
    obs = self._get_obs(state)
    return obs, total_reward, done, info

  def render_goal(self):
    obj_init_pos_temp = self._env.init_config['obj_init_pos'].copy()
    self._env.init_config['obj_init_pos'] = self._env.goal
    self._env.obj_init_pos = self._env.goal
    self._env.hand_init_pos = self._env.goal
    self.reset()
    action = np.zeros(self._env.action_space.low.shape)
    state, reward, done, info = self._env.step(action)
    goal_obs = self._get_obs(state)
    goal_obs['reward'] = 0.0
    self._env.hand_init_pos = self._env.init_config['hand_init_pos']
    self._env.init_config['obj_init_pos'] = obj_init_pos_temp
    self._env.obj_init_pos = self._env.init_config['obj_init_pos']
    self.reset()
    return goal_obs


class MetaWorldVis(MetaWorld):
  def __init__(self, name, action_repeat, width):
    super().__init__(name, action_repeat)
    self._width = width
    self._size = (self._width, self._width)
  
  def render_state(self, state):
    assert (len(state.shape) == 1)
    # Save init configs
    hand_init_pos = self._env.hand_init_pos
    obj_init_pos = self._env.init_config['obj_init_pos']
    # Render state
    hand_pos, obj_pos, hand_to_goal = np.split(state, 3)
    self._env.hand_init_pos = hand_pos
    self._env.init_config['obj_init_pos'] = obj_pos
    self._env.reset_model()
    obs = self._get_obs(state)
    # Revert environment
    self._env.hand_init_pos = hand_init_pos
    self._env.init_config['obj_init_pos'] = obj_init_pos
    self._env.reset()
    return obs['image']
  
  def render_states(self, states):
    assert (len(states.shape) == 2)
    imgs = []
    for s in states:
      img = self.render_state(s)
      imgs.append(img)
    return np.array(imgs)


class MetaWorldSparseReward(MetaWorld):
  def __init__(self, name, action_repeat):
    super().__init__(name, action_repeat)

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, _, done, info = self._env.step(action)
      reward = info['success']
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state)
    return obs, total_reward, done, info


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

  def __init__(self, env, callbacks=None, precision=32, save_sparse_reward=False):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None
    self._save_sparse_reward = save_sparse_reward

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['reward'] = reward
    if self._save_sparse_reward:
      transition['sparse_reward'] = info.get('success')
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
    if self._save_sparse_reward:
      transition['sparse_reward'] = 0.0
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

class Async:

  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, ctor, strategy='process'):
    self._strategy = strategy
    if strategy == 'none':
      self._env = ctor()
    elif strategy == 'thread':
      import multiprocessing.dummy as mp
    elif strategy == 'process':
      import multiprocessing as mp
    else:
      raise NotImplementedError(strategy)
    if strategy != 'none':
      self._conn, conn = mp.Pipe()
      self._process = mp.Process(target=self._worker, args=(ctor, conn))
      atexit.register(self.close)
      self._process.start()
    self._obs_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._obs_space:
      self._obs_space = self.__getattr__('observation_space')
    return self._obs_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    if self._strategy == 'none':
      return getattr(self._env, name)
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    blocking = kwargs.pop('blocking', True)
    if self._strategy == 'none':
      return functools.partial(getattr(self._env, name), *args, **kwargs)
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    promise = self._receive
    return promise() if blocking else promise

  def close(self):
    if self._strategy == 'none':
      try:
        self._env.close()
      except AttributeError:
        pass
      return
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

  def step(self, action, blocking=True):
    return self.call('step', action, blocking=blocking)

  def reset(self, blocking=True):
    return self.call('reset', blocking=blocking)

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except ConnectionResetError:
      raise RuntimeError('Environment worker crashed.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError(f'Received message of unexpected type {message}')

  def _worker(self, ctor, conn):
    try:
      env = ctor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          break
        raise KeyError(f'Received message of unknown type {message}')
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print(f'Error in environment process: {stacktrace}')
      conn.send((self._EXCEPTION, stacktrace))
    conn.close()

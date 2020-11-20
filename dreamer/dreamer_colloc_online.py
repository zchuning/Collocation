import argparse
import functools
import os
import pathlib
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import tensorflow as tf
from blox.utils import timing

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import dreamer
import dreamer_colloc
from utils import wrappers, tools


def define_config():
  config = dreamer_colloc.define_config()
  config.precision = 32
  return config


class DreamerCollocOnline(dreamer_colloc.DreamerColloc):
  def __init__(self, config, datadir, actspace, writer):
    dreamer.Dreamer.__init__(self, config, datadir, actspace, writer)

  @tf.function
  def _policy_summaries(self, feat_pred, act_pred, init_feat):
    # Collocation
    img_pred = self._decode(feat_pred).mode()
    tools.graph_summary(self._writer, tools.video_summary, 'plan', img_pred + 0.5)
    # TODO add error as in _image_summaries

    # Forward prediction
    feat_pred = self._dynamics.imagine_feat(act_pred[None], init_feat)
    img_pred = self._decode(tf.concat((init_feat[None], feat_pred), 1)).mode()
    tools.graph_summary(self._writer, tools.video_summary, 'model', img_pred + 0.5)

    # Deterministic prediction
    feat_pred = self._dynamics.imagine_feat(act_pred[None], init_feat, deterministic=True)
    img_pred = self._decode(tf.concat((init_feat[None], feat_pred), 1)).mode()
    tools.graph_summary(self._writer, tools.video_summary, 'model_mean', img_pred + 0.5)

  # @tf.function
  def plan(self, feat, log_images):
    # TODO speed this up
    if self._c.planning_task == "colloc_second_order":
      act_pred, img_pred, feat_pred, info = self.collocation_so(None, None, False, None, feat, verbose=False)
      for k, v in info['metrics'].items():
        self._metrics[f'opt_{k}'].update_state(v)
    elif self._c.planning_task == "shooting_cem":
      from planners.shooting_cem import ShootingCEMAgent
      act_pred, img_pred, feat_pred = ShootingCEMAgent.shooting_cem(self, None, None, init_feat=feat, verbose=False)
    elif self._c.planning_task == "shooting_gd":
      from planners.shooting_gd import ShootingGDAgent
      act_pred, img_pred, feat_pred = ShootingGDAgent.shooting_gd(self, None, None, init_feat=feat, verbose=False)

    if tf.equal(log_images, True):
      self._policy_summaries(feat_pred, act_pred, feat)
    return act_pred

  def policy(self, obs, state, training):
    feat, latent = self.get_init_feat(obs, state)

    if state is not None and state[2].shape[0] > 0:
      # Cached actions
      actions = state[2]
    else:
      with timing("Plan constructed in: "):
        actions = self.plan(feat, not training)
    action = actions[0:1]
    action = self._exploration(action, training)

    state = (latent, action, actions[1:])
    return action, state


def main(config):
  datadir = dreamer.setup(config, config.logdir)
  # Create environments.
  writer = tf.summary.create_file_writer(
      str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  train_envs = [wrappers.Async(lambda: dreamer.make_env(
      config, writer, 'train', datadir, store=config.train_store), config.parallel)
      for _ in range(config.envs)]
  test_envs = [wrappers.Async(lambda: dreamer.make_env(
      config, writer, 'test', datadir, store=False), config.parallel)
      for _ in range(config.envs)]
  actspace = train_envs[0].action_space

  # Prefill dataset with random episodes.
  step = dreamer.count_steps(datadir, config)
  prefill = max(0, config.prefill - step)
  print(f'Prefill dataset with {prefill} steps.')
  random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
  tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
  writer.flush()

  # Train and regularly evaluate the agent.
  step = dreamer.count_steps(datadir, config)
  print(f'Simulating agent for {config.steps-step} steps.')
  agent = DreamerCollocOnline(config, datadir, actspace, writer)
  if (config.logdir / 'variables.pkl').exists():
    print('Load checkpoint.')
    agent.load(config.logdir / 'variables.pkl')
  state = None
  while step < config.steps:
    print('Start evaluation.')
    tools.simulate(
        functools.partial(agent, training=False), test_envs, episodes=1)
    writer.flush()
    print('Start collection.')
    steps = config.eval_every // config.action_repeat
    state = tools.simulate(agent, train_envs, steps, state=state)
    step = dreamer.count_steps(datadir, config)
    agent.save(config.logdir / 'variables.pkl')
    if config.save_every:
      agent.save(config.logdir / f'variables_{agent.get_step() // config.save_every}.pkl')
  for env in train_envs + test_envs:
    env.close()


if __name__ == '__main__':
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  parser = argparse.ArgumentParser()
  for key, value in define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  main(parser.parse_args())

import argparse
import collections
import functools
import json
import imageio
import os
import pathlib
import sys
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from blox.utils import AverageMeter
from blox.utils import timing
import time

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import wrappers
import dreamer
import dreamer_colloc
import gn_solver
from utils import logging


def define_config():
  config = dreamer_colloc.define_config()
  config.precision = 32
  return config


class DreamerCollocOnline(dreamer_colloc.DreamerColloc):
  def __init__(self, config, datadir, actspace, writer):
    dreamer.Dreamer.__init__(self, config, datadir, actspace, writer)

  @tf.function()
  def train(self, data, log_images=False):
    # TODO quite sure the float32 thing needs to be a config setting
    with tf.GradientTape() as model_tape:
      embed = self._encode(data)
      post, prior = self._dynamics.observe(embed, data['action'])
      feat = self._dynamics.get_feat(post)
      image_pred = self._decode(feat)
      reward_pred = self._reward(feat)
      likes = tools.AttrDict()
      likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
      likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
      if self._c.inverse_model:
        inverse_pred = self._inverse(feat[:, :-1], feat[:, 1:])
        likes.inverse = tf.reduce_mean(inverse_pred.log_prob(data['action'][:, :-1]))
      if self._c.pcont:
        pcont_pred = self._pcont(feat)
        pcont_target = self._c.discount * data['discount']
        likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
        likes.pcont *= self._c.pcont_scale
      prior_dist = self._dynamics.get_dist(prior)
      post_dist = self._dynamics.get_dist(post)
      div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
      div = tf.maximum(div, self._c.free_nats)
      model_loss = self._c.kl_scale * div - sum(likes.values())

    model_norm = self._model_opt(model_tape, model_loss)

    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      if self._c.log_scalars:
        self._scalar_summaries(
            data, feat, prior_dist, post_dist, likes, div,
            model_loss, model_norm)
      if tf.equal(log_images, True):
        self._image_summaries(data, embed, image_pred)

  def _scalar_summaries(
      self, data, feat, prior_dist, post_dist, likes, div,
      model_loss, model_norm):
    self._metrics['model_grad_norm'].update_state(model_norm)
    self._metrics['prior_ent'].update_state(prior_dist.entropy())
    self._metrics['post_ent'].update_state(post_dist.entropy())
    for name, logprob in likes.items():
      self._metrics[name + '_loss'].update_state(-logprob)
    self._metrics['div'].update_state(div)
    self._metrics['model_loss'].update_state(model_loss)

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

  @tf.function
  def plan(self, feat, log_images):
    # TODO speed this up
    # Note: it is possible to get rid of tf.function by removing lambda-functions in collocation_so. This is possible
    # by removing the init_residual_function and enforcing it as a hard constraint.
    # TODO check whether anonymous functions work correctly in graph mode! They seem to, but it's not clear
    act_pred, img_pred, feat_pred = self.collocation_so(None, None, False, None, feat, verbose=False)
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
  if config.gpu_growth:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))
  config.steps = int(config.steps)
  config.logdir.mkdir(parents=True, exist_ok=True)
  print('Logdir', config.logdir)

  # Create environments.
  datadir = config.logdir / 'episodes'
  writer = tf.summary.create_file_writer(
      str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  train_envs = [wrappers.Async(lambda: dreamer.make_env(
      config, writer, 'train', datadir, store=True), config.parallel)
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

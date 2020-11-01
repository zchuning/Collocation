import argparse
import os
import pathlib
import sys

from tqdm import tqdm

""" This script is used to train Dreamer on an offline dataset without online interaction"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

from utils import tools
from dreamer import Dreamer, make_env
import dreamer

def define_config():
  config = dreamer.define_config()
  config.datadir = config.logdir / 'episodes'
  
  return config

class DreamerPrediction(Dreamer):

  def __init__(self, config, datadir, actspace, writer):
    super().__init__(config, datadir, actspace, writer)
    self._step = tf.Variable(0, dtype=tf.int64)
  
  def __call__(self, training=True):
    step = self._step.numpy().item()
    tf.summary.experimental.set_step(step)
    log = self._should_log(step)
    log_images = self._c.log_images and log
    self.train(next(self._dataset), log_images)
    if log:
      self._write_summaries()
    if training:
      self._step.assign_add(1)
  
  def _train(self, data, log_images):
    with tf.GradientTape() as model_tape:
      embed = self._encode(data)
      post, prior = self._dynamics.observe(embed, data['action'])
      feat = self._dynamics.get_feat(post)
      image_pred = self._decode(feat)
      reward_pred = self._reward(feat)
      likes = tools.AttrDict()
      likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
      likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
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
    self._metrics['action_ent'].update_state(self._actor(feat).entropy())
    
    
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
  datadir = config.datadir
  writer = tf.summary.create_file_writer(
    str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  env = make_env(config, writer, 'train', datadir, store=True)
  actspace = env.action_space
  
  # Train the agent.
  agent = DreamerPrediction(config, datadir, actspace, writer)
  if (config.logdir / 'variables.pkl').exists():
    print('Load checkpoint.')
    agent.load(config.logdir / 'variables.pkl')
  
  # Train
  for i in tqdm(range(config.steps)):
    agent()
    agent.save(config.logdir / 'variables.pkl')


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

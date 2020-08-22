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
import time

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import wrappers
from dreamer import Dreamer, preprocess, make_bare_env
import dreamer
from utils import logging
from vis import npy2gif


def define_config():
  config = dreamer.define_config()
  config.precision = 32
  
  # Planning
  config.planning_task = 'colloc_gd'
  config.planning_horizon = 10
  config.mpc_steps = 10
  config.cem_steps = 60
  config.cem_batch_size = 10000
  config.cem_elite_ratio = 0.01
  config.gd_steps = 2000
  config.gd_lr = 0.05
  config.lambda_int = 50
  config.lambda_lr = 1
  config.nu_lr = 100
  config.dyn_loss_scale = 5000
  config.act_loss_scale = 100
  config.visualize = True
  config.logdir_colloc = config.logdir  # logdir is used for loading the model, while logdir_colloc for output
  config.logging = 'tensorboard'  # 'tensorboard' or 'disk'
  config.eval_tasks = 1
  return config


class DreamerColloc(Dreamer):
  def __init__(self, config, datadir, actspace):
    super().__init__(config, datadir, actspace, None)
    tf.summary.experimental.set_step(0)
    if config.logging == 'tensorboard':
      self.logger = logging.TBLogger(config.logdir_colloc)
    else:
      self.logger = logging.DiskLogger(config.logdir_colloc)

  def visualize_colloc(self, img_pred, act_pred, init_feat):
    # Use actions to predict trajectory
    curr_state = self._dynamics.from_feat(init_feat)
    state_pred = self._dynamics.imagine(act_pred[None], curr_state)
    feat_pred = self._dynamics.get_feat(state_pred)
    model_imgs = self._decode(feat_pred).mode().numpy()
    
    # Write images
    self.logger.log_image("colloc_imgs", img_pred.numpy().reshape(-1, 64, 3))
    self.logger.log_image("model_imgs", model_imgs.reshape(-1, 64, 3))
    self.logger.log_video("model", model_imgs)

  def collocation_goal(self, init, goal, optim):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len = (self._actdim + feat_size) * horizon

    # Get initial states and goal states
    null_latent = self._dynamics.initial(len(init['image']))
    null_action = tf.zeros((len(init['image']), self._actdim), self._float)
    init_embedding = self._encode(preprocess(init, self._c))
    init_latent, _ = self._dynamics.obs_step(null_latent, null_action, init_embedding)
    init_feat = self._dynamics.get_feat(init_latent)
    goal_embedding = self._encode(preprocess(goal, self._c))
    goal_latent, _ = self._dynamics.obs_step(null_latent, null_action, goal_embedding)
    goal_feat = self._dynamics.get_feat(goal_latent)

    # Initialize decision variables
    means = tf.zeros(var_len, dtype=self._float)
    stds = tf.ones(var_len, dtype=self._float)
    dyn_loss, forces = [], []

    if optim == 'gd':
      opt = tf.keras.optimizers.Adam(learning_rate=0.05)
      lambdas = tf.ones(horizon)
      nus = tf.ones([horizon, self._actdim])
      traj = tf.reshape(tfd.MultivariateNormalDiag(means, stds).sample(), [horizon, -1])
      actions = tf.Variable(traj[:, :self._actdim])
      feats = tf.Variable(traj[:-1, self._actdim:])
      for i in range(self._c.gd_steps):
        print("Gradient descent step {0}".format(i + 1))
        with tf.GradientTape() as g:
          g.watch(actions)
          g.watch(feats)
          feats_full = tf.concat([init_feat, feats, goal_feat], axis=0)
          # Compute total force squared
          force = tf.reduce_sum(tf.square(actions))
          # Compute dynamics loss
          actions_reshaped = tf.expand_dims(actions, 0)
          stoch = tf.expand_dims(feats_full[:-1, :self._c.stoch_size], 0)
          deter = tf.expand_dims(feats_full[:-1, self._c.stoch_size:], 0)
          states = {'stoch': stoch, 'deter': deter}
          priors = self._dynamics.img_step(states, actions_reshaped)
          feats_pred = tf.squeeze(tf.concat([priors['mean'], priors['deter']], axis=-1))
          log_prob_frame = tf.reduce_sum(tf.square(feats_pred - feats_full[1:]), axis=1)
          log_prob = tf.reduce_sum(lambdas * log_prob_frame)
          actions_viol = tf.clip_by_value(tf.square(actions) - 1, 0, np.inf)
          actions_constr = tf.reduce_sum(nus * actions_viol)
          loss = self._c.dyn_loss_scale * log_prob + self._c.act_loss_scale * actions_constr
        print(f"Frame-wise log probability: {log_prob_frame}")
        print(f"Dynamics: {log_prob}")
        grads = g.gradient(loss, {'actions': actions, 'feats': feats})
        opt.apply_gradients([(grads['actions'], actions)])
        opt.apply_gradients([(grads['feats'], feats)])
        dyn_loss.append(log_prob)
        forces.append(force)
        if i % self._c.lambda_int == self._c.lambda_int - 1:
          lambdas += self._c.lambda_lr * log_prob_frame
          nus += self._c.nu_lr * (actions_viol)
          print(tf.reduce_mean(log_prob_frame, axis=0))
          print(f"Lambdas: {lambdas}\n Nus: {nus}")
      act_pred = actions
      img_pred = self._decode(feats_full[:min(horizon, mpc_steps)]).mode()
    else:
      raise ValueError("Unsupported optimizer")

    print(act_pred)
    print("Final average dynamics loss: {0}".format(dyn_loss[-1] / horizon))
    print("Final average force: {0}".format(forces[-1] / horizon))
    if self._c.visualize:
      self.visualize_colloc(forces, dyn_loss, img_pred, act_pred, init_feat)
    return act_pred, img_pred

  def collocation_gd(self, obs, save_images, step):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len = (self._actdim + feat_size) * horizon

    # Get initial states
    latent = self._dynamics.initial(len(obs['image']))
    # TODO this is wrong for PM since 0,0 is not a noop. Check whether the model is trained correctly with this.
    action = tf.zeros((len(obs['image']), self._actdim), self._float)
    embedding = self._encode(preprocess(obs, self._c))
    latent, _ = self._dynamics.obs_step(latent, action, embedding)
    init_feat = self._dynamics.get_feat(latent)

    # Initialize decision variables
    means = tf.zeros(var_len, dtype=self._float)
    stds = tf.ones(var_len, dtype=self._float)
    t = tf.Variable(tfd.MultivariateNormalDiag(means, stds).sample())

    # Gradient descent parameters
    dyn_loss, act_loss, rewards = [], [], []
    dyn_loss_frame = []
    opt = tf.keras.optimizers.Adam(learning_rate=self._c.gd_lr)
    lambdas = tf.ones(horizon)
    nus = tf.ones([horizon, self._actdim])
    # Gradient descent loop
    for i in range(self._c.gd_steps):
      print("Gradient descent step {0}".format(i + 1))
      with tf.GradientTape() as g:
        g.watch(t)
        tr = tf.reshape(t, [horizon, -1])
        actions = tf.expand_dims(tr[:, :self._actdim], 0)
        feats = tf.concat([init_feat, tr[:, self._actdim:]], axis=0)
        # Compute reward
        reward = tf.reduce_sum(self._reward(feats).mode())
        # Compute dynamics loss
        states = self._dynamics.from_feat(feats[None, :-1])
        priors = self._dynamics.img_step(states, actions)
        feats_pred = tf.squeeze(tf.concat([priors['mean'], priors['deter']], axis=-1))
        # TODO this is NLL, not log_prob
        log_prob_frame = tf.reduce_sum(tf.square(feats_pred - feats[1:]), axis=1)
        log_prob = tf.reduce_sum(lambdas * log_prob_frame)
        actions_viol = tf.clip_by_value(tf.square(actions) - 1, 0, np.inf)
        actions_constr = tf.reduce_sum(nus * actions_viol)
        fitness = - reward + self._c.dyn_loss_scale * log_prob + self._c.act_loss_scale * actions_constr
      print(f"Frame-wise log probability: {log_prob_frame}")
      print(f"Reward: {reward}, dynamics: {log_prob}")
      grad = g.gradient(fitness, t)
      opt.apply_gradients([(grad, t)])
      dyn_loss.append(tf.reduce_sum(log_prob_frame))
      dyn_loss_frame.append(log_prob_frame)
      act_loss.append(tf.reduce_sum(actions_viol))
      rewards.append(reward)
      if i % self._c.lambda_int == self._c.lambda_int - 1:
        lambdas += self._c.lambda_lr * log_prob_frame
        nus += self._c.nu_lr * (actions_viol)
        print(tf.reduce_mean(log_prob_frame, axis=0))
        print(f"Lambdas: {lambdas}\n Nus: {nus}")

    tr = tf.reshape(t, [horizon, -1])
    act_pred = tr[:min(horizon, mpc_steps), :self._actdim]
    feat_pred = tr[:min(horizon, mpc_steps), self._actdim:]
    img_pred = self._decode(feat_pred).mode()
    print(f"Final average dynamics loss: {dyn_loss[-1] / horizon}")
    print(f"Final average action violation: {act_loss[-1] / horizon}")
    print(f"Final average reward: {rewards[-1] / horizon}")
    if save_images:
      self.logger.log_graph(
        'losses', {f'rewards/{step}': rewards, f'dynamics/{step}': dyn_loss, f'action_violation/{step}': act_loss})
      self.visualize_colloc(img_pred, act_pred, init_feat)
    return act_pred, img_pred, feat_pred

  def collocation_cem(self, obs):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    elite_size = int(self._c.cem_batch_size * self._c.cem_elite_ratio)
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len = (self._actdim + feat_size) * horizon
    batch = self._c.cem_batch_size

    # Get initial states
    latent = self._dynamics.initial(len(obs['image']))
    action = tf.zeros((len(obs['image']), self._actdim), self._float)
    embedding = self._encode(preprocess(obs, self._c))
    latent, _ = self._dynamics.obs_step(latent, action, embedding)
    init_feat = self._dynamics.get_feat(latent)

    lambdas = tf.ones(horizon)
    def eval_fitness(t):
      t = tf.reshape(t, [batch, horizon, -1])
      actions = t[:, :, :self._actdim]
      feats = tf.concat([tf.tile(tf.expand_dims(init_feat, 0), [batch, 1, 1]),
                         t[:, :, self._actdim:]], axis=1)
      # Compute reward
      reward = tf.reduce_sum(self._reward(feats).mode(), axis=1)
      # Compute dynamics loss
      states = {'stoch': feats[:, :-1, :self._c.stoch_size],
                'deter': feats[:, :-1, self._c.stoch_size:]}
      priors = self._dynamics.img_step(states, actions)
      feats_pred = tf.squeeze(tf.concat([priors['mean'], priors['deter']], axis=-1))
      # Unweighted log probability, used for updating lambdas
      log_prob_frame = tf.reduce_sum(tf.square(feats_pred - feats[:, 1:]), axis=-1)
      # Weighted log probability, used in fitness
      log_prob_weighted = tf.reduce_sum(lambdas * log_prob_frame, 1)
      fitness = - reward + self._c.dyn_loss_scale * log_prob_weighted
      return fitness, reward, log_prob_frame

    # CEM loop:
    dyn_loss, rewards = [], []
    gamma = 0.00001
    means = tf.zeros(var_len, dtype=self._float)
    stds = tf.ones(var_len, dtype=self._float)
    for i in range(self._c.cem_steps):
      print("CEM step {0} of {1}".format(i + 1, self._c.cem_steps))
      # Sample trajectories and evaluate fitness
      samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[batch])
      fitness, rew, dyn_frame = eval_fitness(samples)
      rewards.append(tf.reduce_mean(rew).numpy())
      dyn_loss.append(tf.reduce_mean(tf.reduce_sum(dyn_frame, axis=1)).numpy())
      # Get elite samples
      elite_inds = tf.argsort(fitness)[:elite_size]
      elite_dyn_frame = tf.gather(dyn_frame, elite_inds)
      elite_samples = tf.gather(samples, elite_inds)
      # Refit distribution
      means = tf.reduce_mean(elite_samples, axis=0)
      stds = tf.math.reduce_std(elite_samples, axis=0)
      # MPPI
      # elite_fitness = tf.gather(fitness, elite_inds)
      # weights = tf.expand_dims(tf.nn.softmax(gamma * elite_fitness), axis=1)
      # means = tf.reduce_sum(weights * elite_samples, axis=0)
      # stds = tf.sqrt(tf.reduce_sum(weights * tf.square(elite_samples - means), axis=0))
      # Lagrange multiplier
      # Update lambdas at a slower rate
      if i % self._c.lambda_int == self._c.lambda_int - 1:
        lambdas += tf.reduce_mean(elite_dyn_frame, axis=0)
        print(tf.reduce_mean(dyn_frame, axis=0))
        print(lambdas)

    means_pred = tf.reshape(means, [horizon, -1])
    act_pred = means_pred[:min(horizon, mpc_steps), :self._actdim]
    img_pred = self._decode(means_pred[:min(horizon, mpc_steps), self._actdim:]).mode()
    print("Final average dynamics loss: {0}".format(dyn_loss[-1] / horizon))
    print("Final average reward: {0}".format(rewards[-1] / horizon))
    if self._c.visualize:
      self.visualize_colloc(rewards, dyn_loss, img_pred, act_pred, init_feat)
    return act_pred, img_pred

  def shooting_cem(self, obs, min_action=-1, max_action=1):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    elite_size = int(self._c.cem_batch_size * self._c.cem_elite_ratio)
    var_len = self._actdim * horizon
    batch = self._c.cem_batch_size

    # Get initial states
    latent = self._dynamics.initial(len(obs['image']))
    action = tf.zeros((len(obs['image']), self._actdim), self._float)
    embedding = self._encode(preprocess(obs, self._c))
    latent, _ = self._dynamics.obs_step(latent, action, embedding)
    init_feat = self._dynamics.get_feat(latent)

    def eval_fitness(t):
      init_feats = tf.tile(init_feat, [batch, 1])
      actions = tf.reshape(t, [batch, horizon, -1])
      init_states = {'stoch': init_feats[:, :self._c.stoch_size],
                     'deter': init_feats[:, self._c.stoch_size:]}
      priors = self._dynamics.imagine(actions, init_states)
      feats = tf.squeeze(tf.concat([priors['stoch'], priors['deter']], axis=-1))
      rewards = tf.reduce_sum(self._reward(feats).mode(), axis=1)
      return rewards, feats

    # CEM loop:
    rewards = []
    means = tf.zeros(var_len, dtype=self._float)
    stds = tf.ones(var_len, dtype=self._float)
    for i in range(self._c.cem_steps):
      print("CEM step {0} of {1}".format(i + 1, self._c.cem_steps))
      # Sample action sequences and evaluate fitness
      samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[batch])
      samples = tf.clip_by_value(samples, min_action, max_action)
      fitness, feats = eval_fitness(samples)
      rewards.append(tf.reduce_mean(fitness).numpy())
      # Refit distribution to elite samples
      _, elite_inds = tf.nn.top_k(fitness, elite_size, sorted=False)
      elite_samples = tf.gather(samples, elite_inds)
      means, vars = tf.nn.moments(elite_samples, 0)
      stds = tf.sqrt(vars + 1e-6)

    means_pred = tf.reshape(means, [horizon, -1])
    act_pred = means_pred[:min(horizon, mpc_steps)]
    feat_pred = feats[elite_inds[0]]
    img_pred = self._decode(feat_pred[:min(horizon, mpc_steps)]).mode()
    print("Final average reward: {0}".format(rewards[-1] / horizon))
    if self._c.visualize:
      import matplotlib.pyplot as plt
      plt.title("Reward Curve")
      plt.plot(range(len(rewards)), rewards)
      plt.savefig('./lr.jpg')
      plt.show()
    return act_pred, img_pred

  def _train(self, data, log_images):
    with tf.GradientTape() as model_tape:
      embed = self._encode(data)
      post, prior = self._dynamics.observe(embed, tf.cast(data['action'], tf.float32))
      feat = self._dynamics.get_feat(post)
      image_pred = self._decode(feat)
      reward_pred = self._reward(feat)
      likes = tools.AttrDict()
      likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
      likes.reward = tf.reduce_mean(reward_pred.log_prob(tf.cast(data['reward'], tf.float32)))
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

    with tf.GradientTape() as actor_tape:
      imag_feat = self._imagine_ahead(post)
      reward = self._reward(imag_feat).mode()
      if self._c.pcont:
        pcont = self._pcont(imag_feat).mean()
      else:
        pcont = self._c.discount * tf.ones_like(reward)
      value = self._value(imag_feat).mode()
      returns = tools.lambda_return(
          reward[:-1], value[:-1], pcont[:-1],
          bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
      discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
          [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
      actor_loss = -tf.reduce_mean(discount * returns)

    with tf.GradientTape() as value_tape:
      value_pred = self._value(imag_feat)[:-1]
      target = tf.stop_gradient(returns)
      value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))

    model_norm = self._model_opt(model_tape, model_loss)
    actor_norm = self._actor_opt(actor_tape, actor_loss)
    value_norm = self._value_opt(value_tape, value_loss)

    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      if self._c.log_scalars:
        self._scalar_summaries(
            data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, model_norm, value_norm,
            actor_norm)
      # if tf.equal(log_images, True):
      #   self._image_summaries(data, embed, image_pred)

  def _image_summaries(self, data, embed, image_pred):
    truth = data['image'][:6] + 0.5
    recon = image_pred.mode()[:6]
    init, _ = self._dynamics.observe(embed[:6, :5], tf.cast(data['action'], tf.float32)[:6, :5])
    init = {k: v[:, -1] for k, v in init.items()}
    prior = self._dynamics.imagine(tf.cast(data['action'], tf.float32)[:6, 5:], init)
    openl = self._decode(self._dynamics.get_feat(prior)).mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    openl = tf.concat([truth, model, error], 2)
    tools.graph_summary(
        self._writer, tools.video_summary, 'agent/openl', openl)


def make_env(config):
  env = make_bare_env(config)
  env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  env = wrappers.RewardObs(env)
  return env


def colloc_simulate(agent, config, env, save_images=True):
  """ Run planning loop """
  # Define task-related variables
  pt = config.planning_task
  actspace = env.action_space
  obs = env.reset()
  obs['image'] = [obs['image']]
  # Obtain goal observation for goal-based collocation
  is_goal_based = (len(config.planning_task.split('_')) == 3)
  if is_goal_based:
    goal_obs = env.render_goal()
    goal_obs['image'] = [goal_obs['image']]
  
  num_iter = config.time_limit // config.action_repeat
  img_preds, act_preds, frames = [], [], []
  total_reward = 0
  start = time.time()
  for i in range(0, num_iter, config.mpc_steps):
    print("Planning step {0} of {1}".format(i + 1, num_iter))
    # Run single planning step
    if pt == 'colloc_cem':
      act_pred, img_pred = agent.collocation_cem(obs)
    elif pt == 'colloc_gd':
      act_pred, img_pred, feat_pred = agent.collocation_gd(obs, save_images, i)
    elif pt == 'colloc_gd_goal':
      act_pred, img_pred = agent.collocation_goal(obs, goal_obs, 'gd')
    elif pt == 'shooting':
      act_pred, img_pred = agent.shooting_cem(obs)
    elif pt == 'random':
      act_pred = tf.random.uniform((config.mpc_steps,) + actspace.shape, actspace.low[0], actspace.high[0])
      img_pred = None
    else:
      raise ValueError("Unimplemented planning task")
    act_pred_np = act_pred.numpy()
    act_preds.append(act_pred_np)
    if img_pred is not None:
      img_preds.append(img_pred.numpy())
    # Simluate in environment
    for j in range(len(act_pred_np)):
      obs, reward, done, _ = env.step(act_pred_np[j])
      total_reward += reward
      frames.append(obs['image'])
    obs['image'] = [obs['image']]
    # Break if running goal-based collocation
    if is_goal_based:
      break
  end = time.time()
  print(f"Planning time: {end - start}")
  if save_images:
    if img_pred is not None:
      img_preds = np.vstack(img_preds)
      # TODO mark beginning in the gif
      agent.logger.log_video("plan", img_preds)
    agent.logger.log_video("execution", frames)
  print("Total reward: " + str(total_reward))
  agent.logger.log_graph('true_reward', {'rewards/true': [total_reward]})
  import pdb; pdb.set_trace()
  agent._reward(feat_pred).mean()
  
  return total_reward
  
  
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
  config.logdir_colloc.mkdir(parents=True, exist_ok=True)

  # Create environment.
  env = make_env(config)

  # Create agent.
  actspace = env.action_space
  datadir = config.logdir / 'episodes'
  agent = DreamerColloc(config, datadir, actspace)
  agent.load(config.logdir / 'variables.pkl')

  reward_meter = AverageMeter()
  for i in range(config.eval_tasks):
    save_images = (i % 10 == 0) and config.visualize
    reward_meter.update(colloc_simulate(agent, config, env, save_images))
  print(f'Average reward across {config.eval_tasks} tasks: {reward_meter.avg}')
  import pdb; pdb.set_trace()


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

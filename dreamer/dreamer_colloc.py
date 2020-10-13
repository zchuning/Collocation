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
from blox.utils import AverageMeter, timing
from blox.basic_types import map_dict
import time

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import wrappers
from dreamer import Dreamer, preprocess, make_bare_env
import dreamer
import gn_solver
import gn_solver_goal
from utils import logging


def define_config():
  config = dreamer.define_config()

  # Planning
  config.planning_task = 'colloc_gd'
  config.planning_horizon = 10
  config.mpc_steps = 10
  config.cem_steps = 60
  config.cem_batch_size = 10000
  config.cem_elite_ratio = 0.01
  config.mppi = False
  config.mppi_gamma = 0.0001
  config.gd_steps = 2000
  config.gd_lr = 0.05
  config.gn_damping = 1e-3
  config.lam_int = 1
  config.lam_lr = 1
  config.lam_step = 10
  config.nu_lr = 100
  config.mu_int = -1
  config.dyn_threshold = 1e-1
  config.act_threshold = 1e-1
  config.rew_threshold = 1e-8
  config.coeff_normalization = 10000
  config.dyn_loss_scale = 5000
  config.act_loss_scale = 100
  config.rew_loss_scale = 1
  config.rew_res_wt = 1
  config.dyn_res_wt = 1
  config.act_res_wt = 1
  config.visualize = True
  config.logdir_colloc = config.logdir  # logdir is used for loading the model, while logdir_colloc for output
  config.logging = 'tensorboard'  # 'tensorboard' or 'disk'
  config.eval_tasks = 1
  config.goal_based = False
  config.hyst_ratio = 0.1
  return config


class DreamerColloc(Dreamer):
  def __init__(self, config, datadir, actspace):
    super().__init__(config, datadir, actspace, None)
    tf.summary.experimental.set_step(0)
    if config.logging == 'tensorboard':
      self.logger = logging.TBLogger(config.logdir_colloc)
    else:
      self.logger = logging.DiskLogger(config.logdir_colloc)

  def compute_rewards(self, feats):
    return self._reward(feats)

  def forward_dynamics(self, states, actions):
    return self._dynamics.img_step(states, actions)

  def forward_dynamics_feat(self, feats, actions):
    # TODO would be nice to handle this with a class
    states = self._dynamics.from_feat(feats)
    state_pred = self._dynamics.img_step(states, actions)
    feat_pred = self._dynamics.get_feat(state_pred)
    return feat_pred

  def decode_feats(self, feats):
    return self._decode(feats).mode()

  def visualize_colloc(self, img_pred, act_pred, init_feat, step=-1):
    # Use actions to predict trajectory
    feat_pred = self._dynamics.imagine_feat(act_pred[None], init_feat)
    model_imgs = self._decode(tf.concat((init_feat[None], feat_pred), 1)).mode().numpy()
    self.logger.log_video(f"model/{step}", model_imgs)

    # Deterministic prediction
    feat_pred = self._dynamics.imagine_feat(act_pred[None], init_feat, deterministic=True)
    model_imgs = self._decode(tf.concat((init_feat[None], feat_pred), 1)).mode().numpy()
    self.logger.log_video(f"model_mean/{step}", model_imgs)

    # Write images
    # self.logger.log_image("colloc_imgs", img_pred.numpy().reshape(-1, 64, 3))
    # self.logger.log_image("model_imgs", model_imgs.reshape(-1, 64, 3))

  def pair_residual_func_body(self, x_a, x_b, goal,
      lam=np.ones(1, np.float32), nu=np.ones(1, np.float32), mu=np.ones(1, np.float32)):
    actions_a = x_a[:, -self._actdim:][None]
    feats_a = x_a[:, :-self._actdim][None]
    states_a = self._dynamics.from_feat(feats_a)
    prior_a = self._dynamics.img_step(states_a, actions_a)
    x_b_pred = tf.concat([prior_a['mean'], prior_a['deter']], -1)[0]
    dyn_res = x_b[:, :-self._actdim] - x_b_pred
    act_res = tf.clip_by_value(tf.math.abs(x_a[:, -self._actdim:]) - 1, 0, np.inf)
    # act_res = tf.clip_by_value(tf.square(x_a[:, -self._actdim:]) - 1, 0, np.inf)
    rew = self._reward(x_b[:, :-self._actdim]).mode()
    rew_res = tf.math.softplus(-rew) # 1.0 - tf.math.sigmoid(rew) # tf.math.softplus(-rew) # -tf.math.log_sigmoid(rew) # Softplus
    # rew_res = rew_c * (1.0 - tf.math.sigmoid(rew)) # Sigmoid
    # rew_res = rew_c * (1.0 / (rew + 10000))[:, None] # Inverse
    # rew_res = rew_c * tf.sqrt(-tf.clip_by_value(rew-100000, -np.inf, 0))[:, None] # shifted reward
    # rew_res = rew_c * (x_b[:, :-self._actdim] - goal) # goal-based reward

    # Compute coefficients
    # TODO redefine weights to not be square roots
    dyn_c = tf.sqrt(lam)[:, None] * self._c.dyn_res_wt
    act_c = tf.sqrt(nu)[:, None] * self._c.act_res_wt
    rew_c = tf.sqrt(mu)[:, None] * tf.cast(self._c.rew_res_wt, act_c.dtype)

    normalize = self._c.coeff_normalization / (tf.reduce_mean(dyn_c) + tf.reduce_mean(act_c) + tf.reduce_mean(rew_c))
    dyn_c = dyn_c * normalize
    act_c = act_c * normalize
    rew_c = rew_c * normalize

    dyn_res = dyn_c * dyn_res
    act_res = act_c * act_res
    rew_res = rew_c * rew_res
    objective = tf.concat([dyn_res, act_res, rew_res], 1)
    return objective

  @tf.function
  def opt_step(self, plan, init_feat, goal_feat, lam, nu, mu):
    init_residual_func = lambda x: (x[:, :-self._actdim] - init_feat) * 1000
    pair_residual_func = lambda x_a, x_b : self.pair_residual_func_body(x_a, x_b, goal_feat, lam, nu, mu)
    plan = gn_solver.solve_step_inference(pair_residual_func, init_residual_func, plan, damping=self._c.gn_damping)
    return plan

  def collocation_so(self, obs, goal_obs, save_images, step, init_feat=None, verbose=True):
    hor = self._c.planning_horizon
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = feat_size + self._actdim
    coeff_upperbound = 1e10
    dyn_threshold = self._c.dyn_threshold
    act_threshold = self._c.act_threshold
    rew_threshold = self._c.rew_threshold

    if init_feat is None:
      init_feat, _ = self.get_init_feat(obs)
    if goal_obs is not None:
      goal_feat, _ = self.get_init_feat(goal_obs)
    else:
      goal_feat = None
    plan = tf.random.normal(((hor + 1) * var_len_step,), dtype=self._float)
    # Set the first state to be the observed initial state
    plan = tf.concat([init_feat[0], plan[feat_size:]], 0)
    plan = tf.reshape(plan, [1, hor + 1, var_len_step])
    lam = tf.ones(hor)
    nu = tf.ones(hor)
    mu = tf.ones(hor)

    # Run second-order solver
    dyn_losses, act_losses, rewards = [], [], []
    model_rewards = []
    plans = []
    dyn_coeffs, act_coeffs = [], []
    for i in range(self._c.gd_steps):
      # Run Gauss-Newton step
      # with timing("Single Gauss-Newton step time: "):
      plan = self.opt_step(plan, init_feat, goal_feat, lam, nu, mu)
      plan_res = tf.reshape(plan, [hor+1, -1])
      feat_preds, act_preds = tf.split(plan_res, [feat_size, self._actdim], 1)
      plans.append(plan)
      # act_preds_clipped = tf.clip_by_value(act_preds, -1, 1)
      # plan = tf.reshape(tf.concat([feat_preds, act_preds_clipped], -1), plan.shape)

      # Compute and record dynamics loss and reward
      init_loss = tf.linalg.norm(feat_preds[0:1] - init_feat)
      rew_raw = self._reward(feat_preds).mode()
      reward = tf.reduce_sum(rew_raw)
      states = self._dynamics.from_feat(feat_preds[None, :-1])
      priors = self._dynamics.img_step(states, act_preds[None, :-1])
      priors_feat = tf.squeeze(tf.concat([priors['mean'], priors['deter']], axis=-1))
      dyn_viol = tf.reduce_sum(tf.square(priors_feat - feat_preds[1:]), 1)
      act_viol = tf.reduce_sum(tf.clip_by_value(tf.square(act_preds[:-1]) - 1, 0, np.inf), 1)
      # act_viol = tf.reduce_sum(tf.square(tf.clip_by_value(tf.abs(act_preds[:-1]) - 1, 0, np.inf)), 1)

      # Record losses and effective coefficients
      dyn_loss = tf.reduce_sum(dyn_viol)
      act_loss = tf.reduce_sum(act_viol)
      dyn_coeff = self._c.dyn_res_wt**2 * tf.reduce_sum(lam)
      act_coeff = self._c.act_res_wt**2 * tf.reduce_sum(nu)
      dyn_losses.append(dyn_loss)
      act_losses.append(act_loss)
      rewards.append(reward)
      dyn_coeffs.append(dyn_coeff)
      act_coeffs.append(act_coeff)

      model_feats = self._dynamics.imagine_feat(act_preds[None, :], init_feat, deterministic=True)
      model_rew = self._reward(model_feats).mode()
      model_rewards.append(tf.reduce_sum(model_rew))

      # Update lagrange multipliers
      if i % self._c.lam_int == self._c.lam_int - 1:
        # if dyn_loss / hor > dyn_threshold and dyn_coeff < coeff_upperbound:
        #   lam = lam * self._c.lam_step
        # if dyn_loss / hor < dyn_threshold * self._c.hyst_ratio:
        #   lam = lam / self._c.lam_step
        # if act_loss / hor > act_threshold and act_coeff < coeff_upperbound:
        #   nu = nu * self._c.lam_step
        # if act_loss / hor < act_threshold * self._c.hyst_ratio:
        #   nu = nu / self._c.lam_step

        lam_step = 1.0 + 0.1 * tf.math.log((dyn_viol + 0.1 * dyn_threshold) / dyn_threshold) / tf.math.log(10.0)
        nu_step  = 1.0 + 0.1 * tf.math.log((act_viol + 0.1 * act_threshold) / act_threshold) / tf.math.log(10.0)
        lam = lam * lam_step
        nu = nu * nu_step

        # lam_step = 1.0 + 0.1 * tf.math.log(((dyn_loss / hor) + dyn_threshold) / (2.0 * dyn_threshold)) / tf.math.log(10.0)
        # nu_step  = 1.0 + 0.1 * tf.math.log(((act_loss / hor) + act_threshold) / (2.0 * act_threshold)) / tf.math.log(10.0)

      # if i % self._c.mu_int == self._c.mu_int - 1:
        # mu_step = 1.0 + 0.1 * tf.math.log((tf.math.softplus(rew_raw[1:]) + rew_threshold) / (2.0 * rew_threshold)) / tf.math.log(10.0)
        # mu = mu * mu_step
        # if tf.reduce_mean(1.0 / (rew_raw + 10000)) > rew_threshold:
        #   mu = mu * self._c.lam_step

    act_preds = act_preds[:min(hor, self._c.mpc_steps)]
    feat_preds = feat_preds[:min(hor, self._c.mpc_steps)]
    if verbose:
      print(f"Final average dynamics loss: {dyn_losses[-1] / hor}")
      print(f"Final average action violation: {act_losses[-1] / hor}")
      print(f"Final total reward: {rewards[-1]}")
      print(f"Final average initial state violation: {init_loss}")
    curves = dict(rewards=rewards, dynamics=dyn_losses, action_violation=act_losses,
                  dynamics_coeff=dyn_coeffs, action_coeff=act_coeffs, model_rewards=model_rewards)
    if save_images:
      img_preds = self._decode(feat_preds).mode()
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in curves.items()})
      self.visualize_colloc(img_preds, act_preds, init_feat, step)
    else:
      img_preds = None
    info = map_dict(lambda x: x[-1] / hor, curves)
    info['plans'] = plans
    return act_preds, img_preds, feat_preds, info

  def collocation_so_goal(self, obs, goal_obs, save_images, step, init_feat=None, verbose=True):
    hor = self._c.planning_horizon
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = feat_size + self._actdim
    damping = 1e-3

    if init_feat is None:
      init_feat, _ = self.get_init_feat(obs)
    goal_feat, _ = self.get_init_feat(goal_obs)
    plan = tf.random.normal((1, hor + 1, var_len_step,), dtype=self._float)

    def fix_first_last(plan):
      # Set the first state to be the observed initial state
      plan = tf.reshape(plan, [1, (hor + 1) * var_len_step])
      plan = tf.concat([init_feat, plan[:, feat_size:-feat_size], goal_feat], 1)
      return tf.reshape(plan, [1, hor + 1, var_len_step])
    plan = fix_first_last(plan)

    def pair_residual_func(x_a, x_b):
      actions_a = x_a[:, -self._actdim:]
      feats_a = x_a[:, :-self._actdim]
      states_a = self._dynamics.from_feat(feats_a[None])
      prior_a = self._dynamics.img_step(states_a, actions_a[None])
      feats_b_pred = tf.concat([prior_a['mean'], prior_a['deter']], -1)[0]
      dyn_res = x_b[:, :-self._actdim] - feats_b_pred
      act_res = tf.clip_by_value(tf.square(actions_a) - 1, 0, np.inf)
      return tf.concat([dyn_res, act_res], 1)

    init_residual_func = lambda x : (x[:, :-self._actdim] - init_feat) * 1000
    # init_residual_func = lambda x: tf.zeros((1,0))

    # Run second-order solver
    dyn_losses, rewards, act_losses = [], [], []
    for i in range(self._c.gd_steps):
      # Run Gauss-Newton step
      with timing("Single Gauss-Newton step time: "):
        plan = gn_solver.solve_step_inference(pair_residual_func, init_residual_func, plan, damping=damping)
        plan = fix_first_last(plan)
      # Compute and record dynamics loss and reward
      feat_plan, act_plan = tf.split(plan[0], [feat_size, self._actdim], 1)
      reward = tf.reduce_sum(self._reward(feat_plan).mode())
      states = self._dynamics.from_feat(feat_plan[None, :-1])
      next_states = self._dynamics.img_step(states, act_plan[None, :-1])
      next_feats = tf.squeeze(tf.concat([next_states['mean'], next_states['deter']], axis=-1))
      dyn_loss = tf.reduce_sum(tf.square(next_feats - feat_plan[1:]))
      act_loss = tf.reduce_sum(tf.clip_by_value(tf.square(act_plan) - 1, 0, np.inf))
      dyn_losses.append(dyn_loss)
      rewards.append(reward)
      act_losses.append(act_loss)

    import pdb; pdb.set_trace()
    act_plan = act_plan[:min(hor, self._c.mpc_steps)]
    feat_plan = feat_plan[1:min(hor, self._c.mpc_steps) + 1]
    if verbose:
      print(f"Final average dynamics loss: {dyn_losses[-1] / hor}")
      print(f"Final average action violation: {act_losses[-1] / hor}")
      print(f"Final average reward: {rewards[-1] / hor}")
    if save_images:
      img_preds = self._decode(feat_plan).mode()
      self.logger.log_graph('losses', {f'rewards/{step}': rewards,
                                       f'dynamics/{step}': dyn_losses,
                                       f'action_violation/{step}': act_losses})
      self.visualize_colloc(img_preds, act_plan, init_feat)
    else:
      img_preds = None
    return act_plan, img_preds, feat_plan

  def collocation_so_goal_1(self, obs, goal_obs, save_images, step, init_feat=None, verbose=True):
    hor = self._c.planning_horizon
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = feat_size + self._actdim
    damping = 1e-3

    if init_feat is None:
      init_feat, _ = self.get_init_feat(obs)
    goal_feat, _ = self.get_init_feat(goal_obs)
    plan = tf.random.normal((hor * var_len_step,), dtype=self._float)
    plan = tf.concat([init_feat[0], plan[feat_size:]], 0)
    plan = tf.reshape(plan, [1, hor, var_len_step])

    def pair_residual_func(x_a, x_b):
      actions_a = x_a[:, feat_size:]
      feats_a = x_a[:, :feat_size]
      states_a = self._dynamics.from_feat(feats_a[None])
      prior_a = self._dynamics.img_step(states_a, actions_a[None])
      feats_b_pred = tf.concat([prior_a['mean'], prior_a['deter']], -1)[0]
      dyn_res = x_b[:, :feat_size] - feats_b_pred
      act_res = tf.clip_by_value(tf.square(actions_a) - 1, 0, np.inf)
      return tf.concat([dyn_res, act_res], 1)

    init_residual_func = lambda x: (x[:, :-self._actdim] - init_feat) * 1000
    # final_residual_func = lambda x: (x[:, :-self._actdim] - goal_feat) * 1000
    final_residual_func = lambda x_a: pair_residual_func(x_a, goal_feat)
    # final_residual_func = lambda x: tf.zeros((1,0))

    # Run second-order solver
    dyn_losses, rewards, act_losses = [], [], []
    for i in range(self._c.gd_steps):
      # Run Gauss-Newton step
      with timing("Single Gauss-Newton step time: "):
        plan = gn_solver_goal.solve_step_inference(pair_residual_func, init_residual_func, final_residual_func, plan, damping=damping)
      # Compute and record dynamics loss and reward
      feat_plan, act_plan = tf.split(plan[0], [feat_size, self._actdim], 1)
      reward = tf.reduce_sum(self._reward(feat_plan).mode())
      states = self._dynamics.from_feat(feat_plan[None, :-1])
      next_states = self._dynamics.img_step(states, act_plan[None, :-1])
      next_feats = tf.squeeze(tf.concat([next_states['mean'], next_states['deter']], axis=-1))
      dyn_loss = tf.reduce_sum(tf.square(next_feats - feat_plan[1:]))
      act_loss = tf.reduce_sum(tf.clip_by_value(tf.square(act_plan) - 1, 0, np.inf))
      dyn_losses.append(dyn_loss)
      rewards.append(reward)
      act_losses.append(act_loss)

    import pdb; pdb.set_trace()
    act_plan = act_plan[:min(hor, self._c.mpc_steps)]
    feat_plan = feat_plan[1:min(hor, self._c.mpc_steps) + 1]
    if verbose:
      print(f"Final average dynamics loss: {dyn_losses[-1] / hor}")
      print(f"Final average action violation: {act_losses[-1] / hor}")
      print(f"Final average reward: {rewards[-1] / hor}")
    if save_images:
      img_preds = self._decode(feat_plan).mode()
      self.logger.log_graph('losses', {f'rewards/{step}': rewards,
                                       f'dynamics/{step}': dyn_losses,
                                       f'action_violation/{step}': act_losses})
      self.visualize_colloc(img_preds, act_plan, init_feat)
    else:
      img_preds = None
    return act_plan, img_preds, feat_plan

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
        if i % self._c.lam_int == self._c.lam_int - 1:
          lambdas += self._c.lam_lr * log_prob_frame
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
    var_len_step = self._actdim + feat_size

    # Initialize decision variables
    init_feat, _ = self.get_init_feat(obs)
    t = tf.Variable(tf.random.normal((horizon, var_len_step), dtype=self._float))
    lambdas = tf.ones(horizon)
    nus = tf.ones([horizon, self._actdim])
    dyn_loss, act_loss, rewards = [], [], []
    dyn_loss_frame = []
    opt = tf.keras.optimizers.Adam(learning_rate=self._c.gd_lr)
    # Gradient descent loop
    for i in range(self._c.gd_steps):
      print("Gradient descent step {0}".format(i + 1))
      with tf.GradientTape() as g:
        g.watch(t)
        actions = tf.expand_dims(t[:, :self._actdim], 0)
        feats = tf.concat([init_feat, t[:, self._actdim:]], axis=0)
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
      if i % self._c.lam_int == self._c.lam_int - 1:
        lambdas += self._c.lam_lr * log_prob_frame
        nus += self._c.nu_lr * (actions_viol)
        print(tf.reduce_mean(log_prob_frame, axis=0))
        print(f"Lambdas: {lambdas}\n Nus: {nus}")

    act_pred = t[:min(horizon, mpc_steps), :self._actdim]
    feat_pred = t[:min(horizon, mpc_steps), self._actdim:]
    img_pred = self._decode(feat_pred).mode()
    print(f"Final average dynamics loss: {dyn_loss[-1] / horizon}")
    print(f"Final average action violation: {act_loss[-1] / horizon}")
    print(f"Final average reward: {rewards[-1] / horizon}")
    if save_images:
      self.logger.log_graph(
        'losses', {f'rewards/{step}': rewards, f'dynamics/{step}': dyn_loss, f'action_violation/{step}': act_loss})
      self.visualize_colloc(img_pred, act_pred, init_feat)
    return act_pred, img_pred, feat_pred

  def collocation_gd_inverse_model(self, obs, save_images, step):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = feat_size

    # Initialize decision variables
    init_feat, _ = self.get_init_feat(obs)
    t = tf.Variable(tf.random.normal((horizon, var_len_step), dtype=self._float))
    lambdas = tf.ones(horizon)
    nus = tf.ones([horizon, self._actdim])
    dyn_loss, act_loss, rewards = [], [], []
    dyn_loss_frame = []
    opt = tf.keras.optimizers.Adam(learning_rate=self._c.gd_lr)
    # Gradient descent loop
    for i in range(self._c.gd_steps):
      print("Gradient descent step {0}".format(i + 1))
      with tf.GradientTape() as g:
        g.watch(t)
        feats = tf.concat([init_feat, t], axis=0)
        actions = self._inverse(feats[:-1], feats[1:]).mean()[None]
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
      if i % self._c.lam_int == self._c.lam_int - 1:
        lambdas += self._c.lam_lr * log_prob_frame
        nus += self._c.nu_lr * (actions_viol)
        print(tf.reduce_mean(log_prob_frame, axis=0))
        print(f"Lambdas: {lambdas}\n Nus: {nus}")

    feat_pred = t[:min(horizon, mpc_steps)]
    act_pred = self._inverse(tf.concat([init_feat, feat_pred[:-1]], axis=0), feat_pred).mean()
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
    init_feat, _ = self.get_init_feat(obs)

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
      if self._c.mppi:
        elite_fitness = tf.gather(fitness, elite_inds)
        weights = tf.expand_dims(tf.nn.softmax(self._c.mppi_gamma * elite_fitness), axis=1)
        means = tf.reduce_sum(weights * elite_samples, axis=0)
        stds = tf.sqrt(tf.reduce_sum(weights * tf.square(elite_samples - means), axis=0))
      else:
        means = tf.reduce_mean(elite_samples, axis=0)
        stds = tf.math.reduce_std(elite_samples, axis=0)
      # Lagrange multiplier
      if i % self._c.lam_int == self._c.lam_int - 1:
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

  def shooting_gd(self, obs, step, min_action=-1, max_action=1):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps

    # Initialize decision variables
    init_feat, _ = self.get_init_feat(obs)
    t = tf.Variable(tf.random.normal((horizon, self._actdim), dtype=self._float))
    lambdas = tf.ones([horizon, self._actdim])
    act_loss, rewards = [], []
    opt = tf.keras.optimizers.Adam(learning_rate=self._c.gd_lr)
    # Gradient descent loop
    for i in range(self._c.gd_steps):
      # print("Gradient descent step {0}".format(i + 1))
      with tf.GradientTape() as g:
        g.watch(t)
        actions = t[None, :]
        feats = self._dynamics.imagine_feat(actions, init_feat, deterministic=True)
        reward = tf.reduce_sum(self._reward(feats).mode(), axis=1)
        actions_viol = tf.clip_by_value(tf.square(actions) - 1, 0, np.inf)
        actions_constr = tf.reduce_sum(lambdas * actions_viol)
        fitness = - self._c.rew_loss_scale * reward + self._c.act_loss_scale * actions_constr
      grad = g.gradient(fitness, t)
      opt.apply_gradients([(grad, t)])
      t.assign(tf.clip_by_value(t, min_action, max_action)) # Prevent OOD preds
      act_loss.append(tf.reduce_sum(actions_viol))
      rewards.append(tf.reduce_sum(reward))
      if i % self._c.lam_int == self._c.lam_int - 1:
        lambdas += self._c.lam_lr * actions_viol

    act_pred = t[:min(horizon, mpc_steps)]
    feat_pred = feats
    img_pred = self._decode(feat_pred).mode()
    print(f"Final average action violation: {act_loss[-1] / horizon}")
    print(f"Final average reward: {rewards[-1] / horizon}")
    curves = dict(rewards=rewards, action_violation=act_loss)
    self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in curves.items()})
    if self._c.visualize:
      img_pred = self._decode(feat_pred[:min(horizon, mpc_steps)]).mode()
      import matplotlib.pyplot as plt
      plt.title("Reward Curve")
      plt.plot(range(len(rewards)), rewards)
      plt.savefig('./lr.jpg')
      plt.show()
    else:
      img_pred = None
    return act_pred, img_pred, feat_pred

  def shooting_cem(self, obs, step, min_action=-1, max_action=1):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    elite_size = int(self._c.cem_batch_size * self._c.cem_elite_ratio)
    var_len = self._actdim * horizon
    batch = self._c.cem_batch_size

    # Get initial states
    init_feat, _ = self.get_init_feat(obs)

    def eval_fitness(t):
      init_feats = tf.tile(init_feat, [batch, 1])
      actions = tf.reshape(t, [batch, horizon, -1])
      feats = self._dynamics.imagine_feat(actions, init_feats, deterministic=True)
      rewards = tf.reduce_sum(self._reward(feats).mode(), axis=1)
      return rewards, feats

    # CEM loop:
    rewards = []
    act_losses = []
    means = tf.zeros(var_len, dtype=self._float)
    stds = tf.ones(var_len, dtype=self._float)
    for i in range(self._c.cem_steps):
      # print("CEM step {0} of {1}".format(i + 1, self._c.cem_steps))
      # Sample action sequences and evaluate fitness
      samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[batch])
      samples = tf.clip_by_value(samples, min_action, max_action)
      fitness, feats = eval_fitness(samples)
      rewards.append(tf.reduce_mean(fitness).numpy())
      # Refit distribution to elite samples
      _, elite_inds = tf.nn.top_k(fitness, elite_size, sorted=False)
      elite_samples = tf.gather(samples, elite_inds)
      if self._c.mppi:
        elite_fitness = tf.gather(fitness, elite_inds)
        weights = tf.expand_dims(tf.nn.softmax(self._c.mppi_gamma * elite_fitness), axis=1)
        means = tf.reduce_sum(weights * elite_samples, axis=0)
        stds = tf.sqrt(tf.reduce_sum(weights * tf.square(elite_samples - means), axis=0))
      else:
        means, vars = tf.nn.moments(elite_samples, 0)
        stds = tf.sqrt(vars + 1e-6)
      # Log action violations
      means_pred = tf.reshape(means, [horizon, -1])
      act_pred = means_pred[:min(horizon, mpc_steps)]
      act_loss = tf.reduce_sum(tf.clip_by_value(tf.square(act_pred) - 1, 0, np.inf))
      act_losses.append(act_loss)

    means_pred = tf.reshape(means, [horizon, -1])
    act_pred = means_pred[:min(horizon, mpc_steps)]
    feat_pred = feats[elite_inds[0]]
    print("Final average reward: {0}".format(rewards[-1] / horizon))
    # Log curves
    curves = dict(rewards=rewards, action_violation=act_losses)
    self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in curves.items()})
    if self._c.visualize:
      img_pred = self._decode(feat_pred[:min(horizon, mpc_steps)]).mode()
      import matplotlib.pyplot as plt
      plt.title("Reward Curve")
      plt.plot(range(len(rewards)), rewards)
      plt.savefig('./lr.jpg')
      plt.show()
    else:
      img_pred = None
    return act_pred, img_pred, feat_pred

  @tf.function()
  def train(self, data, log_images=False):
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
      if self._c.state_regressor:
        states_pred = self._state(tf.stop_gradient(feat))
        likes.state_regressor = tf.reduce_mean(states_pred.log_prob(data['state']))
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
      # TODO figure out why this breaks or make a hotfix...
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
  is_goal_based = 'goal' in pt or config.goal_based
  if is_goal_based:
    goal_obs = env.render_goal()
    imageio.imwrite('goal_img.jpg', goal_obs['image'])
    goal_obs['image'] = [goal_obs['image']]
  else:
    goal_obs = None

  num_iter = config.time_limit // config.action_repeat
  img_preds, act_preds, frames = [], [], []
  total_reward, total_sparse_reward = 0, None
  if config.sparse_reward:
    total_sparse_reward = 0
  start = time.time()
  for i in range(0, num_iter, config.mpc_steps):
    print("Planning step {0} of {1}".format(i + 1, num_iter))
    # Run single planning step
    if pt == 'colloc_cem':
      act_pred, img_pred = agent.collocation_cem(obs)
    elif pt == 'colloc_gd':
      if config.inverse_model:
        act_pred, img_pred, feat_pred = agent.collocation_gd_inverse_model(obs, save_images, i)
      else:
        act_pred, img_pred, feat_pred = agent.collocation_gd(obs, save_images, i)
    elif pt == 'colloc_second_order':
      act_pred, img_pred, feat_pred, _ = agent.collocation_so(obs, goal_obs, save_images, i)
    elif pt == 'colloc_second_order_goal':
      act_pred, img_pred, feat_pred = agent.collocation_so_goal_1(obs, goal_obs, save_images, i)
    elif pt == 'colloc_gd_goal':
      act_pred, img_pred = agent.collocation_goal(obs, goal_obs, 'gd')
    elif pt == 'shooting_cem':
      act_pred, img_pred, feat_pred = agent.shooting_cem(obs, i)
    elif pt == 'shooting_gd':
      act_pred, img_pred, feat_pred = agent.shooting_gd(obs, i)
    elif pt == 'random':
      act_pred = tf.random.uniform((config.mpc_steps,) + actspace.shape, actspace.low[0], actspace.high[0])
      img_pred = None
    else:
      raise ValueError("Unimplemented planning task")
    # Simluate in environment
    act_pred_np = act_pred.numpy()
    for j in range(len(act_pred_np)):
      obs, reward, done, info = env.step(act_pred_np[j])
      total_reward += reward
      if config.sparse_reward:
          total_sparse_reward += info['success'] # float(info['goalDist'] < 0.15)
      frames.append(obs['image'])
    obs['image'] = [obs['image']]
    # Logging
    act_preds.append(act_pred_np)
    if img_pred is not None:
      img_preds.append(img_pred.numpy())
      agent.logger.log_video(f"plan/{i}", img_pred.numpy())
    agent.logger.log_video(f"execution/{i}", frames[-len(act_pred_np):])
  end = time.time()
  goal_dist = info['goalDist'] if 'goalDist' in info else 0 # info['reachDist']
  success = info['success'] if 'success' in info else 0 # float(goal_dist < 0.15)
  print(f"Planning time: {end - start}")
  print(f"Total reward: {total_reward}")
  agent.logger.log_graph('true_reward', {'rewards/true': [total_reward]})
  if config.sparse_reward:
    print(f"Total sparse reward: {total_sparse_reward}")
    agent.logger.log_graph('true_sparse_reward', {'rewards/true': [total_sparse_reward]})
  print(f"Success: {success}")
  if save_images:
    if img_pred is not None:
      img_preds = np.vstack(img_preds)
      # TODO mark beginning in the gif
      agent.logger.log_video("plan/full", img_preds)
    agent.logger.log_video("execution/full", frames)
  return total_reward, total_sparse_reward, success, goal_dist


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

  rew_meter, sp_rew_meter = AverageMeter(), AverageMeter()
  tot_rews, tot_sp_rews, tot_succ = [], [], 0
  goal_dists = []
  for i in range(config.eval_tasks):
    save_images = (i % 1 == 0) and config.visualize
    tot_rew, tot_sp_rew, succ, goal_dist = colloc_simulate(agent, config, env, save_images)
    rew_meter.update(tot_rew)
    tot_rews.append(tot_rew)
    if config.sparse_reward:
      tot_sp_rews.append(tot_sp_rew)
      sp_rew_meter.update(tot_sp_rew)
    tot_succ += succ
    goal_dists.append(goal_dist)
  print(f'Average reward across {config.eval_tasks} tasks: {rew_meter.avg}')
  agent.logger.log_graph('total_reward', {'total_reward/dense': tot_rews})
  agent.logger.log_graph('reward_std',
      {'total_reward/dense_std': [tf.math.reduce_std(tot_rews)]})
  if config.sparse_reward:
    print(f'Average sparse reward across {config.eval_tasks} tasks: {sp_rew_meter.avg}')
    agent.logger.log_graph('total_sparse_reward', {'total_reward/sparse': tot_sp_rews})
    agent.logger.log_graph('sparse_reward_std',
        {'total_reward/sparse_std': [tf.math.reduce_std(tot_sp_rews)]})
  print(f'Success rate: {tot_succ / config.eval_tasks}')
  agent.logger.log_graph('success_rate', {'total_reward/success': [tot_succ / config.eval_tasks]})
  agent.logger.log_hist('total_reward/goal_dist', goal_dists)
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

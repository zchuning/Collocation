import argparse
import os
import pathlib
import sys

import imageio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import tensorflow as tf
from blox.utils import AverageMeter, timing
from blox.basic_types import map_dict
from blox import AttrDefaultDict
import time

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

from dreamer import Dreamer, preprocess, make_bare_env
import dreamer
from planners import gn_solver, gn_solver_goal
from utils import logging, wrappers, tools


def define_config():
  config = dreamer.define_config()

  # Planning
  config.planning_task = 'colloc_second_order'
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
    x_b_pred = self._dynamics.get_mean_feat(prior_a)[0]
    dyn_res = x_b[:, :-self._actdim] - x_b_pred
    act_res = tf.clip_by_value(tf.math.abs(x_a[:, -self._actdim:]) - 1, 0, np.inf)
    # act_res = tf.clip_by_value(tf.square(x_a[:, -self._actdim:]) - 1, 0, np.inf)
    if self._c.ssq_reward:
      rew_res = self._reward.get_residual(x_b[:, :-self._actdim])
    else:
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
    objective = normalize * tf.concat([dyn_c * dyn_res, act_c * act_res, rew_c * rew_res], 1)
    return objective, dyn_res, act_res, rew_res, dyn_c * dyn_res, act_c * act_res, rew_c * rew_res

  @tf.function
  def opt_step(self, plan, init_feat, goal_feat, lam, nu, mu):
    init_residual_func = lambda x: (x[:, :-self._actdim] - init_feat) * 1000
    pair_residual_func = lambda x_a, x_b : self.pair_residual_func_body(x_a, x_b, goal_feat, lam, nu, mu)[0]
    plan = gn_solver.solve_step_inference(pair_residual_func, init_residual_func, plan, damping=self._c.gn_damping)
    return plan

  def collocation_so(self, obs, goal_obs, save_images, step, init_feat=None, verbose=True, log_extras=False):
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
    plans = [plan]
    metrics = AttrDefaultDict(list)
    # Can use tf.range for TF control flow here
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
      states = self._dynamics.from_feat(feat_preds[None, :-1])
      priors = self._dynamics.img_step(states, act_preds[None, :-1])
      priors_feat = tf.squeeze(self._dynamics.get_mean_feat(priors))
      dyn_viol = tf.reduce_sum(tf.square(priors_feat - feat_preds[1:]), 1)
      act_viol = tf.reduce_sum(tf.clip_by_value(tf.square(act_preds[:-1]) - 1, 0, np.inf), 1)
      # act_viol = tf.reduce_sum(tf.square(tf.clip_by_value(tf.abs(act_preds[:-1]) - 1, 0, np.inf)), 1)

      # Record losses and effective coefficients
      metrics.dynamics.append(tf.reduce_sum(dyn_viol))
      metrics.action_violation.append(tf.reduce_sum(act_viol))
      metrics.rewards.append(tf.reduce_sum(rew_raw))
      metrics.dynamics_coeff.append(self._c.dyn_res_wt**2 * tf.reduce_sum(lam))
      metrics.action_coeff.append(self._c.act_res_wt**2 * tf.reduce_sum(nu))

      # Record model rewards
      model_feats = self._dynamics.imagine_feat(act_preds[None, :], init_feat, deterministic=True)
      model_rew = self._reward(model_feats).mode()
      metrics.model_rewards.append(tf.reduce_sum(model_rew))
      
      if log_extras:
        # Record model sample rewards
        model_feats = self._dynamics.imagine_feat(act_preds[None, :], init_feat, deterministic=False)
        model_rew = self._reward(model_feats).mode()
        metrics.model_sample_rewards.append(tf.reduce_sum(model_rew))
        
        # Record residuals
        _, dyn_res, act_res, rew_res, dyn_resw, act_resw, rew_resw = \
          self.pair_residual_func_body(plan[0,:-1], plan[0,1:], goal_feat, lam, nu, mu)
        metrics.residual_dynamics.append(tf.reduce_sum(dyn_res ** 2))
        metrics.residual_actions.append(tf.reduce_sum(act_res ** 2))
        metrics.residual_rewards.append(tf.reduce_sum(rew_res ** 2))
        metrics.weighted_residual_dynamics.append(tf.reduce_sum(dyn_resw ** 2))
        metrics.weighted_residual_actions.append(tf.reduce_sum(act_resw ** 2))
        metrics.weighted_residual_rewards.append(tf.reduce_sum(rew_resw ** 2))

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
    if tf.reduce_any(tf.math.is_nan(act_preds)) or tf.reduce_any(tf.math.is_inf(act_preds)):
      act_preds = tf.zeros_like(act_preds)
    feat_preds = feat_preds[:min(hor, self._c.mpc_steps)]
    if verbose:
      print(f"Final average dynamics loss: {metrics.dynamics[-1] / hor}")
      print(f"Final average action violation: {metrics.action_violation[-1] / hor}")
      print(f"Final total reward: {metrics.rewards[-1]}")
      print(f"Final average initial state violation: {init_loss}")
    if save_images:
      img_preds = self._decode(feat_preds).mode()
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in metrics.items()})
      self.visualize_colloc(img_preds, act_preds, init_feat, step)
    else:
      img_preds = None
    info = {'metrics': map_dict(lambda x: x[-1] / hor, dict(metrics)), 'plans': plans}
    return act_preds, img_preds, feat_preds, info

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
  total_reward, total_sparse_reward = 0, 0
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
      act_pred, img_pred, feat_pred, _ = agent.collocation_so(obs, goal_obs, save_images, i, log_extras=True)
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
      act_pred, img_pred, feat_pred, _ = agent._plan(obs, goal_obs, save_images, i, log_extras=True)
    # Simluate in environment
    act_pred_np = act_pred.numpy()
    for j in range(len(act_pred_np)):
      obs, reward, done, info = env.step(act_pred_np[j])
      total_reward += reward
      if 'success' in info:
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
  if 'goalDist' in info:
    goal_dist = info['goalDist']
  elif 'reachDist' in info:
    goal_dist = info['reachDist']
  else:
    goal_dist = np.nan
  print(f"Planning time: {end - start}")
  print(f"Total reward: {total_reward}")
  agent.logger.log_graph('true_reward', {'rewards/true': [total_reward]})
  success = np.nan
  if 'success' in info:
    success = float(total_sparse_reward > 0) # info['success']
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


def build_agent(config, env):
  # Build an appropriate agent and load any existing checkpoint
  actspace = env.action_space
  datadir = config.logdir / 'episodes'

  if config.planning_task == 'shooting_cem':
    from planners.shooting_cem import ShootingCEMAgent
    agent = ShootingCEMAgent(config, datadir, actspace)
  elif config.planning_task == 'shooting_gd':
    from planners.shooting_gd import ShootingGDAgent
    agent = ShootingGDAgent(config, datadir, actspace)
  elif config.planning_task == 'colloc_cem':
    from planners.colloc_cem import CollocCEMAgent
    agent = CollocCEMAgent(config, datadir, actspace)
  elif config.planning_task == 'colloc_gd':
    from planners.colloc_gd import CollocGDAgent
    agent = CollocGDAgent(config, datadir, actspace)
  elif config.planning_task == 'colloc_inverse_model':
    from planners.colloc_inverse_model import CollocInverseAgent
    agent = CollocInverseAgent(config, datadir, actspace)
  elif 'goal' in config.planning_task:
    from planners.colloc_goal import CollocGoalAgent
    agent = CollocGoalAgent(config, datadir, actspace)
  elif config.planning_task == 'colloc_shoot':
    from planners.colloc_shoot import CollocShootAgent
    agent = CollocShootAgent(config, datadir, actspace)
  elif config.planning_task == 'shoot_colloc':
    from planners.shoot_colloc import ShootCollocAgent
    agent = ShootCollocAgent(config, datadir, actspace)
  elif config.planning_task == 'shootcem_colloc':
    from planners.shootcem_colloc import ShootCEMCollocAgent
    agent = ShootCEMCollocAgent(config, datadir, actspace)
  else:
    agent = DreamerColloc(config, datadir, actspace)
  
  agent.load(config.logdir / 'variables.pkl')
  return agent


def main(config):
  dreamer.setup(config, config.logdir_colloc)
  env = make_env(config)
  agent = build_agent(config, env)

  rew_meter, sp_rew_meter = AverageMeter(), AverageMeter()
  tot_rews, tot_sp_rews, tot_succ = [], [], 0
  goal_dists = []
  for i in range(config.eval_tasks):
    save_images = (i % 1 == 0) and config.visualize
    tot_rew, tot_sp_rew, succ, goal_dist = colloc_simulate(agent, config, env, save_images)
    rew_meter.update(tot_rew)
    tot_rews.append(tot_rew)
    if config.collect_sparse_reward:
      tot_sp_rews.append(tot_sp_rew)
      sp_rew_meter.update(tot_sp_rew)
    tot_succ += succ
    goal_dists.append(goal_dist)
  print(f'Average reward across {config.eval_tasks} tasks: {rew_meter.avg}')
  print(f'Average goal distance across {config.eval_tasks} tasks: {tf.reduce_mean(goal_dists)}')
  agent.logger.log_graph('total_reward', {'total_reward/dense': tot_rews})
  agent.logger.log_graph('reward_std', {'total_reward/dense_std': [np.std(tot_rews)]})
  if config.collect_sparse_reward:
    print(f'Average sparse reward across {config.eval_tasks} tasks: {sp_rew_meter.avg}')
    agent.logger.log_graph('total_sparse_reward', {'total_reward/sparse': tot_sp_rews})
    agent.logger.log_graph('sparse_reward_std', {'total_reward/sparse_std': [np.std(tot_sp_rews)]})
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

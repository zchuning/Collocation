from dreamer_colloc import DreamerColloc
import tensorflow as tf
from blox import AttrDefaultDict
from blox.basic_types import map_dict
import numpy as np
from tensorflow_probability import distributions as tfd


class ShootCEMCollocAgent(DreamerColloc):
  def _plan(self, obs, goal_obs, save_images, step, init_feat=None, verbose=True, log_extras=False):
    horizon = hor = self._c.planning_horizon
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = feat_size + self._actdim
    var_len_shoot = self._actdim * horizon
    dyn_threshold = self._c.dyn_threshold
    act_threshold = self._c.act_threshold
    rew_threshold = self._c.rew_threshold
    batch = self._c.cem_batch_size
    elite_size = int(self._c.cem_batch_size * self._c.cem_elite_ratio)

    if init_feat is None:
      init_feat, _ = self.get_init_feat(obs)
    if goal_obs is not None:
      goal_feat, _ = self.get_init_feat(goal_obs)
    else:
      goal_feat = None
      
    ## Shooting
    def eval_fitness(t):
      init_feats = tf.tile(init_feat, [batch, 1])
      actions = tf.reshape(t, [batch, horizon, -1])
      feats = self._dynamics.imagine_feat(actions, init_feats, deterministic=True)
      rewards = tf.reduce_sum(self._reward(feats).mode(), axis=1)
      return rewards, feats

    # CEM loop:
    act_losses = []
    means = tf.zeros(var_len_shoot, dtype=self._float)
    stds = tf.ones(var_len_shoot, dtype=self._float)
    metrics = AttrDefaultDict(list)
    for i in range(self._c.cem_steps):
      # print("CEM step {0} of {1}".format(i + 1, self._c.cem_steps))
      # Sample action sequences and evaluate fitness
      samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[batch])
      samples = tf.clip_by_value(samples, -1, 2)
      fitness, feats = eval_fitness(samples)
      metrics.rewards.append(tf.reduce_mean(fitness).numpy())
      metrics.model_rewards.append(tf.reduce_mean(fitness).numpy())
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
      act_pred = means_pred[:min(horizon, self._c.mpc_steps)]
      act_loss = tf.reduce_sum(tf.clip_by_value(tf.square(act_pred) - 1, 0, np.inf))
      act_losses.append(act_loss)

    feat_pred = feats[elite_inds[0]]
    img_preds = self._decode(feat_pred).mode().numpy()
    self.logger.log_video(f"shooting_plan/{step}", img_preds)
  
    plan = tf.random.normal(((hor + 1) * var_len_step,), dtype=self._float)
    # Set the first state to be the observed initial state
    plan = tf.concat([init_feat[0], plan[feat_size:]], 0)
    plan = tf.Variable(tf.reshape(plan, [1, hor + 1, var_len_step]))
    # tf.Variable necessary for assignment to work
    plan[0, 1:, -self._actdim:].assign(act_pred)
    plan[0, 1:, :-self._actdim].assign(feat_pred)
    lam = tf.ones(hor)
    nu = tf.ones(hor)
    mu = tf.ones(hor)
  
    plans = [plan]
    ## Collocation
    for i in range(self._c.gd_steps):
      # Run Gauss-Newton step
      # with timing("Single Gauss-Newton step time: "):
      plan = self.opt_step(plan, init_feat, goal_feat, lam, nu, mu)
      plan_res = tf.reshape(plan, [hor + 1, -1])
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
      act_loss = tf.reduce_sum(act_viol)
      dyn_coeff = self._c.dyn_res_wt ** 2 * tf.reduce_sum(lam)
      act_coeff = self._c.act_res_wt ** 2 * tf.reduce_sum(nu)
      metrics.dynamics.append(tf.reduce_sum(dyn_viol))
      metrics.action_violation.append(act_loss)
      metrics.rewards.append(reward)
      metrics.dynamics_coeff.append(dyn_coeff)
      metrics.action_coeff.append(act_coeff)
    
      # Record model sample rewards
      model_feats = self._dynamics.imagine_feat(act_preds[None, :], init_feat, deterministic=False)
      model_rew = self._reward(model_feats).mode()
      metrics.model_rewards.append(tf.reduce_sum(model_rew))
    
      # Update lagrange multipliers
      if i % self._c.lam_int == self._c.lam_int - 1:
        lam_delta = lam * 0.1 * tf.math.log((dyn_viol + 0.1 * dyn_threshold) / dyn_threshold) / tf.math.log(10.0)
        nu_delta = nu * 0.1 * tf.math.log((act_viol + 0.1 * act_threshold) / act_threshold) / tf.math.log(10.0)
        lam = lam + lam_delta  # * self._c.lam_lr
        nu = nu + nu_delta  # * self._c.nu_lr

    act_preds = act_preds[:min(hor, self._c.mpc_steps)]
    feat_preds = feat_preds[:min(hor, self._c.mpc_steps)]
    
    img_preds = self._decode(feat_preds).mode()
    self.logger.log_video(f"collocation_plan/{step}", img_preds.numpy())
  
    if verbose:
      print(f"Final average dynamics loss: {metrics.dynamics[-1] / hor}")
      print(f"Final average action violation: {metrics.action_violation[-1] / hor}")
      print(f"Final total reward: {metrics.rewards[-1]}")
      print(f"Final average initial state violation: {init_loss}")
    if save_images:
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in metrics.items()})
      # self.visualize_colloc(img_preds, act_preds, init_feat, step)
    else:
      img_preds = None
    info = {'metrics': map_dict(lambda x: x[-1] / hor, dict(metrics)), 'plans': plans}
    return act_preds, img_preds, feat_preds, info
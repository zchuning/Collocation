import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd

from dreamer_colloc import DreamerColloc, preprocess
from planners import gn_solver_goal, gn_solver
from blox import AttrDefaultDict
from blox.basic_types import map_dict
from blox.utils import timing


class CollocGoalAgent(DreamerColloc):
  def compute_rewards(self, feats, goal):
    """ Defines what reward is used for collocation """
    if self._c.goal_distance == 'latent':
      return -tf.reduce_sum((feats - goal) ** 2)
    elif self._c.goal_distance == 'embed':
      return -tf.reduce_sum((self._embed(feats).mode() - self._embed(goal).mode()) ** 2)
  
  def reward_residual(self, feats, goal):
    """ Defines what reward residual is used for collocation """
    if self._c.goal_distance == 'latent':
      return feats - goal
    elif self._c.goal_distance == 'embed':
      return self._embed(feats).mode() - self._embed(goal).mode()
      
  
  def pair_residual_func_body(self, x_a, x_b, goal,
                              lam=np.ones(1, np.float32), nu=np.ones(1, np.float32), mu=np.ones(1, np.float32)):
    actions_a = x_a[:, -self._actdim:][None]
    feats_a = x_a[:, :-self._actdim][None]
    states_a = self._dynamics.from_feat(feats_a)
    prior_a = self._dynamics.img_step(states_a, actions_a)
    x_b_pred = self._dynamics.get_mean_feat(prior_a)[0]
    dyn_res = x_b[:, :-self._actdim] - x_b_pred
    act_res = tf.clip_by_value(tf.math.abs(x_a[:, -self._actdim:]) - 1, 0, np.inf)
    # rew = self.compute_rewards(x_b[:, :-self._actdim], goal).mode()
    # rew_res = tf.math.softplus(-rew)  # 1.0 - tf.math.sigmoid(rew) # tf.math.softplus(-rew) # -tf.math.log_sigmoid(rew) # Softplus
    rew_res = self.reward_residual(x_b[:, :-self._actdim], goal)

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
    pair_residual_func = lambda x_a, x_b: self.pair_residual_func_body(x_a, x_b, goal_feat, lam, nu, mu)[0]
    plan = gn_solver.solve_step_inference(pair_residual_func, init_residual_func, plan, damping=self._c.gn_damping)
    return plan

  def collocation_so_goal(self, obs, goal_obs, save_images, step, init_feat=None, verbose=True, log_extras=False):
    # Standard LatCo with redefined reward
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
      plan_res = tf.reshape(plan, [hor + 1, -1])
      feat_preds, act_preds = tf.split(plan_res, [feat_size, self._actdim], 1)
      plans.append(plan)
      act_preds_clipped = tf.clip_by_value(act_preds, -1, 1)
      # plan = tf.reshape(tf.concat([feat_preds, act_preds_clipped], -1), plan.shape)
  
      # Compute and record dynamics loss and reward
      init_loss = tf.linalg.norm(feat_preds[0:1] - init_feat)
      rew_raw = self.compute_rewards(feat_preds, goal_feat)
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
      metrics.dynamics_coeff.append(self._c.dyn_res_wt ** 2 * tf.reduce_sum(lam))
      metrics.action_coeff.append(self._c.act_res_wt ** 2 * tf.reduce_sum(nu))
  
      # Record model rewards
      model_feats = self._dynamics.imagine_feat(act_preds_clipped[None, :], init_feat, deterministic=True)
      model_rew = self.compute_rewards(model_feats, goal_feat)
      metrics.model_rewards.append(tf.reduce_sum(model_rew))
  
      if log_extras:
        # Record model sample rewards
        model_feats = self._dynamics.imagine_feat(act_preds[None, :], init_feat, deterministic=False)
        model_rew = self.compute_rewards(model_feats, goal_feat)
        metrics.model_sample_rewards.append(tf.reduce_sum(model_rew))
    
        # Record residuals
        _, dyn_res, act_res, rew_res, dyn_resw, act_resw, rew_resw = \
          self.pair_residual_func_body(plan[0, :-1], plan[0, 1:], goal_feat, lam, nu, mu)
        metrics.residual_dynamics.append(tf.reduce_sum(dyn_res ** 2))
        metrics.residual_actions.append(tf.reduce_sum(act_res ** 2))
        metrics.residual_rewards.append(tf.reduce_sum(rew_res ** 2))
        metrics.weighted_residual_dynamics.append(tf.reduce_sum(dyn_resw ** 2))
        metrics.weighted_residual_actions.append(tf.reduce_sum(act_resw ** 2))
        metrics.weighted_residual_rewards.append(tf.reduce_sum(rew_resw ** 2))
  
      # Update lagrange multipliers
      if i % self._c.lam_int == self._c.lam_int - 1:
        lam_delta = lam * 0.1 * tf.math.log((dyn_viol + 0.1 * dyn_threshold) / dyn_threshold) / tf.math.log(10.0)
        nu_delta = nu * 0.1 * tf.math.log((act_viol + 0.1 * act_threshold) / act_threshold) / tf.math.log(10.0)
        lam = lam + lam_delta  # * self._c.lam_lr
        nu = nu + nu_delta  # * self._c.nu_lr

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
    
  # def collocation_so_goal_old(self, obs, goal_obs, save_images, step, init_feat=None, verbose=True):
  #   hor = self._c.planning_horizon
  #   feat_size = self._c.stoch_size + self._c.deter_size
  #   var_len_step = feat_size + self._actdim
  #   damping = 1e-3
  #
  #   if init_feat is None:
  #     init_feat, _ = self.get_init_feat(obs)
  #   goal_feat, _ = self.get_init_feat(goal_obs)
  #   plan = tf.random.normal((1, hor + 1, var_len_step,), dtype=self._float)
  #
  #   def fix_first_last(plan):
  #     # Set the first state to be the observed initial state
  #     plan = tf.reshape(plan, [1, (hor + 1) * var_len_step])
  #     plan = tf.concat([init_feat, plan[:, feat_size:-feat_size], goal_feat], 1)
  #     return tf.reshape(plan, [1, hor + 1, var_len_step])
  #   plan = fix_first_last(plan)
  #
  #   def pair_residual_func(x_a, x_b):
  #     actions_a = x_a[:, -self._actdim:]
  #     feats_a = x_a[:, :-self._actdim]
  #     states_a = self._dynamics.from_feat(feats_a[None])
  #     prior_a = self._dynamics.img_step(states_a, actions_a[None])
  #     feats_b_pred = tf.concat([prior_a['mean'], prior_a['deter']], -1)[0]
  #     dyn_res = x_b[:, :-self._actdim] - feats_b_pred
  #     act_res = tf.clip_by_value(tf.square(actions_a) - 1, 0, np.inf)
  #     return tf.concat([dyn_res, act_res], 1)
  #
  #   init_residual_func = lambda x : (x[:, :-self._actdim] - init_feat) * 1000
  #   # init_residual_func = lambda x: tf.zeros((1,0))
  #
  #   # Run second-order solver
  #   dyn_losses, rewards, act_losses = [], [], []
  #   for i in range(self._c.gd_steps):
  #     # Run Gauss-Newton step
  #     with timing("Single Gauss-Newton step time: "):
  #       plan = gn_solver.solve_step_inference(pair_residual_func, init_residual_func, plan, damping=damping)
  #       plan = fix_first_last(plan)
  #     # Compute and record dynamics loss and reward
  #     feat_plan, act_plan = tf.split(plan[0], [feat_size, self._actdim], 1)
  #     reward = tf.reduce_sum(self._reward(feat_plan).mode())
  #     states = self._dynamics.from_feat(feat_plan[None, :-1])
  #     next_states = self._dynamics.img_step(states, act_plan[None, :-1])
  #     next_feats = tf.squeeze(tf.concat([next_states['mean'], next_states['deter']], axis=-1))
  #     dyn_loss = tf.reduce_sum(tf.square(next_feats - feat_plan[1:]))
  #     act_loss = tf.reduce_sum(tf.clip_by_value(tf.square(act_plan) - 1, 0, np.inf))
  #     dyn_losses.append(dyn_loss)
  #     rewards.append(reward)
  #     act_losses.append(act_loss)
  #
  #   import pdb; pdb.set_trace()
  #   act_plan = act_plan[:min(hor, self._c.mpc_steps)]
  #   feat_plan = feat_plan[1:min(hor, self._c.mpc_steps) + 1]
  #   if verbose:
  #     print(f"Final average dynamics loss: {dyn_losses[-1] / hor}")
  #     print(f"Final average action violation: {act_losses[-1] / hor}")
  #     print(f"Final average reward: {rewards[-1] / hor}")
  #   if save_images:
  #     img_preds = self._decode(feat_plan).mode()
  #     self.logger.log_graph('losses', {f'rewards/{step}': rewards,
  #                                      f'dynamics/{step}': dyn_losses,
  #                                      f'action_violation/{step}': act_losses})
  #     self.visualize_colloc(img_preds, act_plan, init_feat)
  #   else:
  #     img_preds = None
  #   return act_plan, img_preds, feat_plan

  def collocation_so_goal_boundary(self, obs, goal_obs, save_images, step, init_feat=None, verbose=True):
    # Using final_residual_func to make sure the final state matches the goal
    hor = self._c.planning_horizon
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = feat_size + self._actdim
    act_threshold = self._c.act_threshold

    if init_feat is None:
      init_feat, _ = self.get_init_feat(obs)
    goal_feat, _ = self.get_init_feat(goal_obs)
    plan = tf.random.normal((hor * var_len_step,), dtype=self._float)
    plan = tf.concat([init_feat[0], plan[feat_size:]], 0)
    plan = tf.reshape(plan, [1, hor, var_len_step])
    nu = tf.ones(hor - 1)

    def pair_residual_func(x_a, x_b, nu=np.ones(1, np.float32)):
      actions_a = x_a[:, feat_size:]
      feats_a = x_a[:, :feat_size]
      states_a = self._dynamics.from_feat(feats_a[None])
      prior_a = self._dynamics.img_step(states_a, actions_a[None])
      feats_b_pred = tf.concat([prior_a['mean'], prior_a['deter']], -1)[0]
      dyn_res = x_b[:, :feat_size] - feats_b_pred
      act_res = tf.clip_by_value(tf.square(actions_a) - 1, 0, np.inf)
      act_c = tf.sqrt(nu)[:, None] * self._c.act_res_wt
      return tf.concat([dyn_res, act_c * act_res], 1)

    @tf.function
    def opt_step(plan, nu):
      # final_residual_func = lambda x: (x[:, :-self._actdim] - goal_feat) * 1000
      # final_residual_func = lambda x: tf.zeros((1,0))
      init_residual_func = lambda x: (x[:, :-self._actdim] - init_feat) * 1000
      final_residual_func = lambda x_a: pair_residual_func(x_a, goal_feat)
      pair_residual_func_ = lambda x_a, x_b: pair_residual_func(x_a, x_b, nu)
      plan = gn_solver_goal.solve_step_inference(
        pair_residual_func_, init_residual_func, final_residual_func, plan, damping=self._c.gn_damping)
      return plan
    
    # Run second-order solver
    metrics = AttrDefaultDict(list)
    for i in range(self._c.gd_steps):
      # Run Gauss-Newton step
      # with timing("Single Gauss-Newton step time: "):
      plan = opt_step(plan, nu)
      # Compute and record dynamics loss and reward
      feat_plan, act_plan = tf.split(plan[0], [feat_size, self._actdim], 1)
      reward = tf.reduce_sum(self._reward(feat_plan).mode())
      states = self._dynamics.from_feat(feat_plan[None, :-1])
      next_states = self._dynamics.img_step(states, act_plan[None, :-1])
      next_feats = tf.squeeze(tf.concat([next_states['mean'], next_states['deter']], axis=-1))
      dyn_loss = tf.reduce_sum(tf.square(next_feats - feat_plan[1:]))
      act_loss = tf.reduce_sum(tf.clip_by_value(tf.square(act_plan) - 1, 0, np.inf))
      metrics.dynamics.append(dyn_loss)
      metrics.rewards.append(reward)
      metrics.action_violation.append(act_loss)

      if i % self._c.lam_int == self._c.lam_int - 1:
        nu_delta = nu * 0.1 * tf.math.log((act_loss + 0.1 * act_threshold) / act_threshold) / tf.math.log(10.0)
        nu = nu + nu_delta  # * self._c.nu_lr

    act_plan = act_plan[:min(hor, self._c.mpc_steps)]
    feat_plan = feat_plan[1:min(hor, self._c.mpc_steps) + 1]
    if verbose:
      print(f"Final average dynamics loss: {metrics.dynamics[-1] / hor}")
      print(f"Final average action violation: {metrics.action_violation[-1] / hor}")
      print(f"Final average reward: {metrics.rewards[-1] / hor}")
    if save_images:
      img_preds = self._decode(feat_plan).mode()
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in metrics.items()})
      self.visualize_colloc(img_preds, act_plan, init_feat, step)
    else:
      img_preds = None
    return act_plan, img_preds, feat_plan, None

  def collocation_goal_gd(self, init, goal, optim):
    #
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
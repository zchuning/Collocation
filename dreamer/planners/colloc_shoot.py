from dreamer_colloc import DreamerColloc
import tensorflow as tf
from blox import AttrDefaultDict
from blox.basic_types import map_dict


class CollocShootAgent(DreamerColloc):
  def _plan(self, obs, goal_obs, save_images, step, init_feat=None, verbose=True, log_extras=False):
    hor = self._c.planning_horizon
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = feat_size + self._actdim
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
  
    plans = [plan]
    metrics = AttrDefaultDict(list)
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
      dyn_loss = tf.reduce_sum(dyn_viol)
      act_loss = tf.reduce_sum(act_viol)
      dyn_coeff = self._c.dyn_res_wt ** 2 * tf.reduce_sum(lam)
      act_coeff = self._c.act_res_wt ** 2 * tf.reduce_sum(nu)
      metrics.dynamics.append(dyn_loss)
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
    # TODO mark as colloc
    img_preds = self._decode(feat_preds).mode()
    self.visualize_colloc(img_preds, act_preds, init_feat, step)
  
    ## Shooting
    t = tf.Variable(act_preds)
    lambdas = tf.ones([hor, self._actdim])
    opt = tf.keras.optimizers.Adam(learning_rate=self._c.gd_lr)
    for i in range(self._c.gd_steps):
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
      t.assign(tf.clip_by_value(t, -1, 1)) # Prevent OOD preds
      metrics.action_violation.append(tf.reduce_sum(actions_viol))
      metrics.rewards.append(tf.reduce_sum(reward))
      if i % self._c.lam_int == self._c.lam_int - 1:
        lambdas += self._c.lam_lr * actions_viol
  
    act_pred = t[:min(hor, self._c.mpc_steps)]
    feat_pred = feats
    img_pred = self._decode(feat_pred).mode()
  
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
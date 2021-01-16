import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd

from dreamer_colloc import DreamerColloc, preprocess
from planners import gn_solver_goal, gn_solver
from blox import AttrDefaultDict
from blox.basic_types import map_dict
from blox.utils import timing


class ShootingGDAgent(DreamerColloc):
  def shooting_gd(self, obs, step, min_action=-1, max_action=1, init_feat=None, verbose=True):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    batch = self._c.n_parallel_plans

    # Initialize decision variables
    if init_feat is None:
      init_feat, _ = self.get_init_feat(obs)
    init_feat = tf.repeat(init_feat, batch, 0)
    action_plan = tf.Variable(tf.random.normal((batch, horizon, self._actdim), dtype=self._float))
    lambdas = tf.ones([batch, horizon, self._actdim])
    act_loss, rewards = [], []
    opt = tf.keras.optimizers.Adam(learning_rate=self._c.gd_lr)
    # Gradient descent loop
    for i in range(self._c.gd_steps):
      # print("Gradient descent step {0}".format(i + 1))
      with tf.GradientTape() as g:
        g.watch(action_plan)
        feats = self._dynamics.imagine_feat(action_plan, init_feat, deterministic=True)
        reward = tf.reduce_sum(self._reward(feats).mode(), 1)
        actions_viol = tf.clip_by_value(tf.square(action_plan) - 1, 0, np.inf)
        actions_constr = tf.reduce_sum(lambdas * actions_viol)
        fitness = - self._c.rew_loss_scale * tf.reduce_sum(reward) + self._c.act_loss_scale * actions_constr
      grad = g.gradient(fitness, action_plan)
      opt.apply_gradients([(grad, action_plan)])
      action_plan.assign(tf.clip_by_value(action_plan, min_action, max_action)) # Prevent OOD preds
      act_loss.append(tf.reduce_sum(actions_viol))
      rewards.append(tf.reduce_sum(reward))
      if i % self._c.lam_int == self._c.lam_int - 1:
        lambdas += self._c.lam_lr * actions_viol

    best_plan = tf.argmax(reward)
    if batch > 1:
      print(f'plan rewards: {reward}, best plan: {best_plan}')
    act_pred = action_plan[best_plan, :min(horizon, mpc_steps)]
    feat_pred = feats
    curves = dict(rewards=rewards, action_violation=act_loss)
    if verbose:
      print(f"Final average action violation: {act_loss[-1] / horizon}")
      print(f"Final average reward: {rewards[-1] / horizon}")
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in curves.items()})
    if self._c.visualize:
      img_pred = self._decode(feat_pred[[best_plan], :min(horizon, mpc_steps)]).mode()
      import matplotlib.pyplot as plt
      plt.title("Reward Curve")
      plt.plot(range(len(rewards)), rewards)
      plt.savefig('./lr.jpg')
      plt.show()
    else:
      img_pred = None
    return act_pred, img_pred, feat_pred, {'metrics': map_dict(lambda x: x[-1] / horizon, curves),
                                           'predicted_rewards': tf.reduce_max(reward)}

import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd

from dreamer_colloc import DreamerColloc, preprocess
from planners import gn_solver_goal, gn_solver
from blox import AttrDefaultDict
from blox.basic_types import map_dict
from blox.utils import timing


class ShootingCEMAgent(DreamerColloc):
  def shooting_cem(self, obs, step, min_action=-1, max_action=1, init_feat=None, verbose=True):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    elite_size = int(self._c.cem_batch_size * self._c.cem_elite_ratio)
    var_len = self._actdim * horizon
    batch = self._c.cem_batch_size

    # Get initial statesi
    if init_feat is None:
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
    if verbose:
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

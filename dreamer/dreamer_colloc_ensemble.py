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
import time

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import wrappers


def define_config():
  config = tools.AttrDict()
  # General.
  config.logdir = pathlib.Path('.')
  config.seed = 0
  config.steps = 5e6
  config.eval_every = 1e4
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = True
  config.gpu_growth = True
  config.precision = 32
  # Environment.
  config.task = 'dmc_walker_walk'
  config.envs = 1
  config.parallel = 'none'
  config.action_repeat = 2
  config.time_limit = 1000
  config.prefill = 5000
  config.eval_noise = 0.0
  config.clip_rewards = 'none'
  # Model.
  config.deter_size = 200
  config.stoch_size = 30
  config.num_units = 400
  config.dense_act = 'elu'
  config.cnn_act = 'relu'
  config.cnn_depth = 32
  config.pcont = False
  config.free_nats = 3.0
  config.kl_scale = 1.0
  config.pcont_scale = 10.0
  config.weight_decay = 0.0
  config.weight_decay_pattern = r'.*'
  # Training.
  config.batch_size = 50
  config.batch_length = 50
  config.train_every = 1000
  config.train_steps = 100
  config.pretrain = 100
  config.model_lr = 6e-4
  config.value_lr = 8e-5
  config.actor_lr = 8e-5
  config.grad_clip = 100.0
  config.dataset_balance = False
  # Behavior.
  config.discount = 0.99
  config.disclam = 0.95
  config.horizon = 15
  config.action_dist = 'tanh_normal'
  config.action_init_std = 5.0
  config.expl = 'additive_gaussian'
  config.expl_amount = 0.3
  config.expl_decay = 0.0
  config.expl_min = 0.0
  # Planning
  config.planning_task = 'colloc_cem'
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
  config.var_loss_scale = 5000
  config.visualize = False
  config.prefix = ''
  config.num_ensembles = 5
  return config


class Dreamer(tools.Module):

  def __init__(self, config, datadir, actspace, writer):
    self._c = config
    self._actspace = actspace
    self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
    self._writer = writer
    self._random = np.random.RandomState(config.seed)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
    self._should_pretrain = tools.Once()
    self._should_train = tools.Every(config.train_every)
    self._should_log = tools.Every(config.log_every)
    self._last_log = None
    self._last_time = time.time()
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    self._metrics['expl_amount']  # Create variable for checkpoint.
    self._float = prec.global_policy().compute_dtype
    self._strategy = tf.distribute.MirroredStrategy()
    with self._strategy.scope():
      self._dataset = iter(self._strategy.experimental_distribute_dataset(
          load_dataset(datadir, self._c)))
      self._build_model()

  def __call__(self, obs, reset, state=None, training=True):
    step = self._step.numpy().item()
    tf.summary.experimental.set_step(step)
    if state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      state = tf.nest.map_structure(lambda x: x * mask, state)
    if self._should_train(step):
      log = self._should_log(step)
      n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
      print(f'Training for {n} steps.')
      with self._strategy.scope():
        for train_step in range(n):
          log_images = self._c.log_images and log and train_step == 0
          self.train(next(self._dataset), log_images)
      if log:
        self._write_summaries()
    action, state = self.policy(obs, state, training)
    if training:
      self._step.assign_add(len(reset) * self._c.action_repeat)
    return action, state

  def compute_rewards(self, feats):
    return self._reward(feats)

  def forward_dynamics(self, states, actions):
    return self._dynamics.img_step(states, actions)

  def decode_feats(self, feats):
    return self._decode(feats).mode()


  def get_init_feat(self, obs):
    latent = self._dynamics.initial(len(obs['image']))
    action = tf.zeros((len(obs['image']), self._actdim), self._float)
    embedding = self._encode(preprocess(obs, self._c))
    latent, _ = self._dynamics.obs_step(latent, action, embedding)
    init_feat = self._dynamics.get_feat(latent)
    return init_feat

  def visualize_colloc(self, rewards, dyn_loss, img_pred, act_pred, init_feat):
    # Plot loss curves
    import matplotlib.pyplot as plt
    plt.title("Learning Curves")
    plt.plot(range(len(rewards)), rewards, label='Rewards')
    plt.plot(range(len(dyn_loss)), dyn_loss, label='Dynamics Loss')
    plt.legend()
    plt.savefig(f"./lr_{self._c.prefix}.jpg")
    plt.show()
    # Decode states predicted by collocation
    colloc_imgs = img_pred.numpy().reshape(-1, 64, 3)
    # Decode states generated by the model using predicted actions
    model_imgs = []
    curr_feat = tf.squeeze(init_feat)
    for j in range(len(act_pred)):
      states = {'stoch': tf.reshape(curr_feat[:self._c.stoch_size], [1, 1, -1]),
                'deter': tf.reshape(curr_feat[self._c.stoch_size:], [1, 1, -1])}
      curr_actions = tf.reshape(act_pred[j], [1, 1, -1])
      prior = self._dynamics.img_step(states, curr_actions)
      curr_feat = tf.squeeze(tf.concat([prior['mean'], prior['deter']], axis=-1))
      model_img = self._decode(tf.expand_dims(curr_feat, 0)).mode()[0]
      model_imgs.append(model_img.numpy())
    model_imgs = np.array(model_imgs)
    # Write images
    imageio.imwrite(f"./colloc_imgs_{self._c.prefix}.jpg", colloc_imgs)
    imageio.mimsave(f"./model_{self._c.prefix}.gif", model_imgs, fps=60)
    imageio.imwrite(f"./model_imgs_{self._c.prefix}.jpg", model_imgs.reshape(-1, 64, 3))
    sys.exit()

  def visualize_model_preds(self, init_feat, act_pred):
    model_imgs = []
    curr_feat = tf.squeeze(init_feat)
    for j in range(act_pred.shape[0]):
      states = {'stoch': tf.reshape(curr_feat[:self._c.stoch_size], [1, 1, -1]),
                'deter': tf.reshape(curr_feat[self._c.stoch_size:], [1, 1, -1])}
      curr_actions = tf.reshape(act_pred[j], [1, 1, -1])
      prior = self._dynamics.img_step(states, curr_actions)
      curr_feat = tf.squeeze(tf.concat([prior['mean'], prior['deter']], axis=-1))
      model_img = self._decode(tf.expand_dims(curr_feat, 0)).mode()[0]
      model_imgs.append(model_img.numpy())
    model_imgs = np.array(model_imgs)
    return model_imgs

  def collocation_gd(self, obs):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len = (self._actdim + feat_size) * self._c.planning_horizon

    init_feat = self.get_init_feat(obs)
    means = tf.zeros(var_len, dtype=self._float)
    stds = tf.ones(var_len, dtype=self._float)
    t = tf.Variable(tfd.MultivariateNormalDiag(means, stds).sample())
    opt = tf.keras.optimizers.Adam(learning_rate=self._c.gd_lr)
    dyn_loss, act_loss, rewards, variances = self.collocation_gd_body(obs, init_feat, t, opt)
    t_res = tf.reshape(t, [horizon, -1])
    act_pred = t_res[:min(horizon, mpc_steps), :self._actdim]
    img_pred = self._decode(t_res[:min(horizon, mpc_steps), self._actdim:]).mode()
    # print(act_pred)
    print(f"Dynamics loss: {dyn_loss[-1]}")
    print(f"Action violation: {act_loss[-1]}")
    print(f"Reward: {rewards[-1]}")
    print(f"Variance: {variances[-1]}")

    # Visualize model forward predictions
    model_imgs = self.visualize_model_preds(init_feat, act_pred)
    if self._c.visualize:
      # self.visualize_colloc(rewards, dyn_loss, img_pred, act_pred, init_feat)
      import matplotlib.pyplot as plt
      plt.title("Learning Curves")
      plt.plot(range(len(rewards)), rewards, label='Rewards')
      plt.plot(range(len(dyn_loss)), dyn_loss, label='Dynamics Loss')
      plt.plot(range(len(act_loss)), act_loss, label='Action Loss')
      plt.legend()
      plt.savefig(f"./lr_{self._c.prefix}.jpg")
      colloc_imgs = img_pred.numpy().reshape(-1, 64, 3)
      imageio.imwrite(f"./colloc_imgs_{self._c.prefix}.jpg", colloc_imgs)
      sys.exit()
    return act_pred, img_pred, model_imgs

  def collocation_gd_body(self, obs, init_feat, t, opt):
    horizon = self._c.planning_horizon
    # Gradient descent parameters
    dyn_loss, rewards, act_loss, variances = [], [], [], []
    lambdas, nus = tf.ones(horizon), tf.ones([horizon, self._actdim])
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
        stoch = tf.expand_dims(feats[:-1, :self._c.stoch_size], 0)
        deter = tf.expand_dims(feats[:-1, self._c.stoch_size:], 0)
        states = {'stoch': stoch, 'deter': deter}
        priors = self._dynamics.img_step(states, actions)
        feats_pred = tf.squeeze(tf.concat([priors['mean'], priors['deter']], axis=-1))
        log_prob_seq = tf.reduce_sum(tf.square(feats_pred - feats[1:]), axis=1)
        log_prob = tf.reduce_sum(lambdas * log_prob_seq)
        # Compute actions constraint
        actions_viol = tf.clip_by_value(tf.square(actions) - 1, 0, np.inf)
        actions_constr = tf.reduce_sum(nus * actions_viol)
        # Compute disagreement
        feats_ens = []
        for m in self._ensembles:
          pr = m.img_step(states, actions)
          feats_ens.append(tf.squeeze(tf.concat([pr['mean'], pr['deter']], axis=-1)))
        ens_variance = tf.reduce_sum(tf.math.reduce_variance(feats_ens, axis=0))
        fitness = - reward + self._c.dyn_loss_scale * log_prob + \
            self._c.act_loss_scale * actions_constr + \
            self._c.var_loss_scale * ens_variance
      grad = g.gradient(fitness, t)
      opt.apply_gradients([(grad, t)])
      if i % self._c.lambda_int == self._c.lambda_int - 1:
        lambdas += self._c.lambda_lr * log_prob_seq
        nus += self._c.nu_lr * actions_viol
        print(f"Lambdas: {lambdas}\n Nus: {nus}")
      dyn_loss.append(tf.reduce_sum(log_prob_seq))
      act_loss.append(tf.reduce_sum(actions_viol))
      rewards.append(reward)
      variances.append(ens_variance)

    return dyn_loss, act_loss, rewards, variances

  def shooting_cem(self, obs, min_action=-1, max_action=1):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    elite_size = int(self._c.cem_batch_size * self._c.cem_elite_ratio)
    var_len = self._actdim * horizon
    batch = self._c.cem_batch_size

    # Get initial states
    init_feat = self.get_init_feat(obs)
    def eval_fitness(t):
      init_feats = tf.tile(init_feat, [batch, 1])
      actions = tf.reshape(t, [batch, horizon, -1])
      init_states = {'stoch': init_feats[:, :self._c.stoch_size],
                     'deter': init_feats[:, self._c.stoch_size:]}
      priors = self._dynamics.imagine(actions, init_states)
      feats = tf.squeeze(tf.concat([priors['stoch'], priors['deter']], axis=-1))
      rewards = tf.reduce_sum(self._reward(feats).mode(), axis=1)
      return rewards

    # CEM loop:
    rewards = []
    means = tf.zeros(var_len, dtype=self._float)
    stds = tf.ones(var_len, dtype=self._float)
    for i in range(self._c.cem_steps):
      print("CEM step {0} of {1}".format(i + 1, self._c.cem_steps))
      # Sample action sequences and evaluate fitness
      samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[batch])
      samples = tf.clip_by_value(samples, min_action, max_action)
      fitness = eval_fitness(samples)
      rewards.append(tf.reduce_mean(fitness).numpy())
      # Refit distribution to elite samples
      _, elite_inds = tf.nn.top_k(fitness, elite_size, sorted=False)
      elite_samples = tf.gather(samples, elite_inds)
      means, vars = tf.nn.moments(elite_samples, 0)
      stds = tf.sqrt(vars + 1e-6)

    means_pred = tf.reshape(means, [horizon, -1])
    act_pred = means_pred[:min(horizon, mpc_steps)]
    print("Final average reward: {0}".format(rewards[-1] / horizon))
    if self._c.visualize:
      import matplotlib.pyplot as plt
      plt.title("Reward Curve")
      plt.plot(range(len(rewards)), rewards)
      plt.savefig(f"./lr_{self._c.prefix}.jpg")
      plt.show()
      sys.exit()
    return act_pred

  @tf.function
  def policy(self, obs, state, training):
    if state is None:
      latent = self._dynamics.initial(len(obs['image']))
      action = tf.zeros((len(obs['image']), self._actdim), self._float)
    else:
      latent, action = state
    embed = self._encode(preprocess(obs, self._c))
    latent, _ = self._dynamics.obs_step(latent, action, embed)
    feat = self._dynamics.get_feat(latent)
    if training:
      action = self._actor(feat).sample()
    else:
      action = self._actor(feat).mode()
    action = self._exploration(action, training)
    state = (latent, action)
    return action, state

  def load(self, filename):
    super().load(filename)
    self._should_pretrain()

  @tf.function()
  def train(self, data, log_images=False):
    self._strategy.experimental_run_v2(self._train, args=(data, log_images))

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
      model_loss /= float(self._strategy.num_replicas_in_sync)

    with tf.GradientTape() as ensemble_tape:
      ensemble_loss = 0
      prev_state = {k: v[:, :-1] for k, v in post.items()}
      next_state = {k: v[:, 1: ] for k, v in post.items()}
      next_state_dist = self._dynamics.get_dist(next_state)
      for e in self._ensembles:
        pred_state = e.img_step(prev_state, data['action'][:, :-1])
        pred_state_dist = self._dynamics.get_dist(pred_state)
        div_e = tf.reduce_mean(tfd.kl_divergence(next_state_dist, pred_state_dist))
        mse_e = tf.reduce_mean(tf.square(next_state['deter'] - pred_state['deter']))
        ensemble_loss += self._c.kl_scale * div_e + mse_e
      ensemble_loss /= float(self._strategy.num_replicas_in_sync)

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
      actor_loss /= float(self._strategy.num_replicas_in_sync)

    with tf.GradientTape() as value_tape:
      value_pred = self._value(imag_feat)[:-1]
      target = tf.stop_gradient(returns)
      value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
      value_loss /= float(self._strategy.num_replicas_in_sync)

    model_norm = self._model_opt(model_tape, model_loss)
    actor_norm = self._actor_opt(actor_tape, actor_loss)
    value_norm = self._value_opt(value_tape, value_loss)
    ensemble_norm = self._ensemble_opt(ensemble_tape, ensemble_loss)

    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      if self._c.log_scalars:
        self._scalar_summaries(
            data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, ensemble_loss,
            model_norm, value_norm, actor_norm, ensemble_norm)
      if tf.equal(log_images, True):
        self._image_summaries(data, embed, image_pred)

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
    self._dynamics = models.RSSM(
        self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    self._ensembles = []
    for _ in range(self._c.num_ensembles):
      self._ensembles.append(models.RSSMCell(
        self._c.stoch_size, self._c.deter_size, self._c.deter_size))
    self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
    self._actor = models.ActionDecoder(
        self._actdim, 4, self._c.num_units, self._c.action_dist,
        init_std=self._c.action_init_std, act=act)
    model_modules = [self._encode, self._dynamics, self._decode, self._reward]
    if self._c.pcont:
      model_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
    self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
    self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
    self._ensemble_opt = Optimizer('ensemble', self._ensembles, self._c.model_lr)
    # Do a train step to initialize all variables, including optimizer
    # statistics. Ideally, we would use batch size zero, but that doesn't work
    # in multi-GPU mode.
    self.train(next(self._dataset))

  def _exploration(self, action, training):
    if training:
      amount = self._c.expl_amount
      if self._c.expl_decay:
        amount *= 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
      if self._c.expl_min:
        amount = tf.maximum(self._c.expl_min, amount)
      self._metrics['expl_amount'].update_state(amount)
    elif self._c.eval_noise:
      amount = self._c.eval_noise
    else:
      return action
    if self._c.expl == 'additive_gaussian':
      return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
    if self._c.expl == 'completely_random':
      return tf.random.uniform(action.shape, -1, 1)
    if self._c.expl == 'epsilon_greedy':
      indices = tfd.Categorical(0 * action).sample()
      return tf.where(
          tf.random.uniform(action.shape[:1], 0, 1) < amount,
          tf.one_hot(indices, action.shape[-1], dtype=self._float),
          action)
    raise NotImplementedError(self._c.expl)

  def _imagine_ahead(self, post):
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in post.items()}
    policy = lambda state: self._actor(
        tf.stop_gradient(self._dynamics.get_feat(state))).sample()
    states = tools.static_scan(
        lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
        tf.range(self._c.horizon), start)
    imag_feat = self._dynamics.get_feat(states)
    return imag_feat

  def _scalar_summaries(
      self, data, feat, prior_dist, post_dist, likes, div,
      model_loss, value_loss, actor_loss, ensemble_loss,
      model_norm, value_norm, actor_norm, ensemble_norm):
    self._metrics['model_grad_norm'].update_state(model_norm)
    self._metrics['value_grad_norm'].update_state(value_norm)
    self._metrics['actor_grad_norm'].update_state(actor_norm)
    self._metrics['ensemble_grad_norm'].update_state(ensemble_norm)
    self._metrics['prior_ent'].update_state(prior_dist.entropy())
    self._metrics['post_ent'].update_state(post_dist.entropy())
    for name, logprob in likes.items():
      self._metrics[name + '_loss'].update_state(-logprob)
    self._metrics['div'].update_state(div)
    self._metrics['model_loss'].update_state(model_loss)
    self._metrics['value_loss'].update_state(value_loss)
    self._metrics['actor_loss'].update_state(actor_loss)
    self._metrics['action_ent'].update_state(self._actor(feat).entropy())
    self._metrics['ensemble_loss'].update_state(ensemble_loss)

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

  def _write_summaries(self):
    step = int(self._step.numpy())
    metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
    if self._last_log is not None:
      duration = time.time() - self._last_time
      self._last_time += duration
      metrics.append(('fps', (step - self._last_log) / duration))
    self._last_log = step
    [m.reset_states() for m in self._metrics.values()]
    with (self._c.logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
    [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
    print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
    self._writer.flush()


def preprocess(obs, config):
  dtype = prec.global_policy().compute_dtype
  obs = obs.copy()
  with tf.device('cpu:0'):
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
    obs['reward'] = clip_rewards(obs['reward'])
  return obs


def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config):
  episode = next(tools.load_episodes(directory, 1))
  types = {k: v.dtype for k, v in episode.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
  generator = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length,
      config.dataset_balance)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.map(functools.partial(preprocess, config=config))
  dataset = dataset.prefetch(10)
  return dataset


def summarize_episode(episode, config, datadir, writer, prefix):
  episodes, steps = tools.count_episodes(datadir)
  length = (len(episode['reward']) - 1) * config.action_repeat
  ret = episode['reward'].sum()
  print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
  metrics = [
      (f'{prefix}/return', float(episode['reward'].sum())),
      (f'{prefix}/length', len(episode['reward']) - 1),
      (f'episodes', episodes)]
  step = count_steps(datadir, config)
  with (config.logdir / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
  with writer.as_default():  # Env might run in a different thread.
    tf.summary.experimental.set_step(step)
    [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
    if prefix == 'test':
      tools.video_summary(f'sim/{prefix}/video', episode['image'][None])


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

  # Create environment.
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
  elif suite == 'mw':
    if task == 'sawyer_SawyerPushXYEnv':
      env = wrappers.SawyerPushXYEnv(False, config.action_repeat)
    else:
      env = wrappers.MetaWorld(task, False, config.action_repeat)
  else:
    raise ValueError("Unsuported environment")
  env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  env = wrappers.RewardObs(env)
  obs = env.reset()
  obs['image'] = [obs['image']]
  init_obs = obs.copy()

  # Define task-related variables
  pt = config.planning_task
  is_shooting = pt == 'shooting'
  is_goal_based = (len(config.planning_task.split('_')) == 3)

  # Create agent.
  actspace = env.action_space
  datadir = config.logdir / 'episodes'
  writer = tf.summary.create_file_writer(
      str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  agent = Dreamer(config, datadir, actspace, writer)
  agent.load(config.logdir / 'variables.pkl')
  # if pt == 'colloc_ensemble':
  #   agents = []
  #   for i in range(config.num_ensembles):
  #     agent_temp = Dreamer(config, datadir, actspace, writer)
  #     agent_temp.load(config.logdir / f'variables{i}.pkl')
  #     agents.append(agent_temp)

  # Obtain goal observation for goal-based collocation
  if is_goal_based:
    goal_obs = env.render_goal()
    goal_obs['image'] = [goal_obs['image']]

  # Run planning loop
  num_iter = config.time_limit // config.action_repeat
  img_preds, act_preds, frames = [], [], []
  total_reward = 0
  start = time.time()
  model_frames = []
  for i in range(0, num_iter, config.mpc_steps):
    print("Planning step {0} of {1}".format(i + 1, num_iter))
    # Run single planning step
    if pt == 'colloc_cem':
      act_pred, img_pred = agent.collocation_cem(obs)
    elif pt == 'colloc_gd':
      act_pred, img_pred, model_imgs = agent.collocation_gd(obs)
      model_frames.append(model_imgs)
    elif pt == 'colloc_gd_goal':
      act_pred, img_pred = agent.collocation_goal(obs, goal_obs, 'gd')
    elif pt == 'shooting':
      act_pred = agent.shooting_cem(obs)
    elif pt == 'colloc_ensemble':
      act_pred, img_pred, model_imgs = agent.collocation_gd(obs)
    else:
      raise ValueError("Unimplemented planning task")
    act_pred_np = act_pred.numpy()
    act_preds.append(act_pred_np)
    if not is_shooting:
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
  print("Total reward: " + str(total_reward))
  # Save videos
  imageio.mimsave(f"./frames_{config.prefix}.gif", frames, fps=60)
  if not is_shooting:
    img_preds = np.vstack(img_preds)
    imageio.mimsave(f"./preds_{config.prefix}.gif", img_preds, fps=60)
  if len(model_frames) > 0:
    model_frames = np.vstack(model_frames)
    imageio.mimsave(f"./model_{config.prefix}.gif", model_frames, fps=60)
  # Save executions of an ensemble of models
  # act_preds = np.vstack(act_preds)
  # np.save('actions.npy', act_preds)
  # act_preds = np.load('actions.npy')
  # if len(act_preds) > 0:
  #   for i in range(config.num_ensembles):
  #     print(f"Simulating in model {i}")
  #     agent.load(config.logdir / f'variables{i}.pkl')
  #     init_feat = agent.get_init_feat(init_obs)
  #     model_frames = agent.visualize_model_preds(init_feat, tf.convert_to_tensor(act_preds))
  #     imageio.mimsave(f"./model_e{i}_{config.prefix}.gif", model_frames, fps=60)


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

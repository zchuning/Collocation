import gym
import math
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

class Collocation():
  def __init__(self, env, time):
    self.env = env
    self.env.reset()
    self.time = time
    self.optim_iter = 0

  def guess(self):
    x0 = np.random.rand(self.time) - 0.5
    v0 = np.random.rand(self.time) - 0.5
    u0 = np.random.rand(self.time) - 0.5
    z0 = np.concatenate((x0, v0, u0))
    return z0

  def objective(self, z):
    return np.sum(np.square(z[-self.time:]))

  def dynamics_constraint(self, z):
    # Dynamics:
    # v += u * self.env.power - 0.0025 * math.cos(3 * x)
    # x += v
    x, u, v = np.split(z, 3)

    # Displacement is equal to velocity
    x_diff = x[1:] - x[:-1]
    position_constraint = x_diff - v[:-1]

    # Acceleration is equal to force (unit mass)
    v_diff = v[1:] - v[:-1]
    acc = u[:-1] * self.env.power + 0.0025 * np.cos(3 * x[:-1])
    velocity_constraint = v_diff - acc

    return np.concatenate((position_constraint, velocity_constraint))

  def solve(self, method):
    # Define initial and goal states
    init_position = self.env.state[0]
    goal_position = self.env.goal_position
    init_velocity = self.env.state[1]
    goal_velocity = 0

    # Guess initial values of decision variables
    z0 = self.guess()

    # Dynamics constraints
    dyn_constr = NonlinearConstraint(self.dynamics_constraint, 0, 0)

    # Path and boundary constraints
    x_lb = np.ones(self.time) * self.env.min_position
    x_ub = np.ones(self.time) * self.env.max_position
    x_lb[0] = x_ub[0] = init_position
    x_lb[-1] = x_ub[-1] = goal_position

    v_lb = np.ones(self.time) * self.env.max_speed * -1
    v_ub = np.ones(self.time) * self.env.max_speed
    v_lb[0] = v_ub[0] = init_velocity
    v_lb[-1] = v_ub[-1] = goal_velocity

    u_lb = np.ones(self.time) * self.env.min_action
    u_ub = np.ones(self.time) * self.env.max_action

    z_lb = np.concatenate((x_lb, v_lb, u_lb))
    z_ub = np.concatenate((x_ub, v_ub, u_ub))
    z_bounds = Bounds(z_lb, z_ub)

    # Optimize
    print("Start optimization")
    res = None
    if method == 'SLSQP':
      res = minimize(self.objective, z0, method='SLSQP', bounds=z_bounds, \
                     constraints=[{'type': 'eq', 'fun':self.dynamics_constraint}], \
                     callback=self.optim_callback_slsqp)
    elif method == 'trust-constr':
      res = minimize(self.objective, z0, method='trust-constr', bounds=z_bounds, \
                     constraints=dyn_constr, callback=self.optim_callback_trust_constr)
    else:
      print("Unsupported optimization method")
    print("Optimization terminated")
    self.print_summary(res)
    return res

  def simulate(self, actions=None, render=True):
    total_reward = 0
    for i in range(self.time):
      if render:
        self.env.render()
      action = self.env.action_space.sample() if actions is None else [actions[i]]
      obs, reward, done, _ = self.env.step(action)
      total_reward += reward
    print("Total reward: " + str(total_reward))

  def print_summary(self, res):
    print("Status: {0}\n Message: {1}".format(res.status, res.message))
    z_opt = res.x
    x_opt, v_opt, u_opt = np.split(z_opt, 3)
    print("Positions: {0}\n Velocities: {1}\n Actions: {2}".format(x_opt, v_opt, u_opt))
    input("Press any key to watch simulation...")
    self.simulate(u_opt)

  def optim_callback_trust_constr(self, z, state):
    self.optim_iter += 1
    print("Iteration " + str(self.optim_iter) + ": " + str(state.constr_violation))
    if self.optim_iter % 50 == 0:
      self.simulate(z[-self.time:], False)
    return False

  def optim_callback_slsqp(self, z):
    self.optim_iter += 1
    print("Iteration " + str(self.optim_iter))
    if self.optim_iter % 50 == 0:
      self.simulate(z[-self.time:], False)


if __name__ == '__main__':
  env = gym.make('MountainCarContinuous-v0')
  colloc = Collocation(env, 150)
  colloc.solve('SLSQP')
  env.close()

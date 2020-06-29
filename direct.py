import gym
import math
import numpy as np
from pathlib import Path
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from utils import simulate

class DirectCollocation():
  def __init__(self, env, time):
    self.env = env
    self.t = time
    self.optim_iter = 0

  def guess(self):
    x0 = np.random.rand(self.t) - 0.5
    v0 = np.random.rand(self.t) - 0.5
    u0 = np.random.rand(self.t) - 0.5
    z0 = np.concatenate((x0, v0, u0))
    return z0

  def objective(self, z):
    # Minimize total force applied
    return np.sum(np.square(z[-self.t:]))

  def dynamics_constraint(self, z):
    # Dynamics: v += u * self.env.power - 0.0025 * math.cos(3 * x); x += v
    x, v, u = np.split(z, 3)
    # Displacement is equal to velocity
    dx = x[1:] - x[:-1]
    position_constraint = dx - v[:-1]
    # Acceleration is equal to force (unit mass)
    dv = v[1:] - v[:-1]
    force = u[:-1] * self.env.power + 0.0025 * np.cos(3 * x[:-1])
    velocity_constraint = dv - force
    return np.concatenate((position_constraint, velocity_constraint))

  def velocity_constraint(self, z):
    x, v, u = np.split(z, 3)
    dv = v[1:] - v[:-1]
    force = u[:-1] * self.env.power + 0.0025 * np.cos(3 * x[:-1])
    return dv - force

  def dynamics_constraint_alt(self):
    position_constraint_mat = np.concatenate((-np.eye(self.t) + np.eye(self.t, k=1), \
                                              -np.eye(self.t), \
                                               np.zeros((self.t, self.t))), 1)[:-1]
    position_constraint = LinearConstraint(position_constraint_mat, 0, 0)
    velocity_constraint = NonlinearConstraint(self.velocity_constraint, 0, 0)
    return [position_constraint, velocity_constraint]

  def solve(self, method):
    # Define initial and goal states
    init_position = self.env.state[0]
    init_velocity = self.env.state[1]
    goal_position = self.env.goal_position
    goal_velocity = 0
    print("Initial position: {0}. Initial velocity: {1}".format(init_position, init_velocity))

    # Guess initial values of decision variables
    z0 = self.guess()

    # Dynamics constraints
    # dyn_constr = [NonlinearConstraint(self.dynamics_constraint, 0, 0)]
    dyn_constr = self.dynamics_constraint_alt()

    # Path and boundary constraints
    x_lb = np.ones(self.t) * self.env.min_position
    x_ub = np.ones(self.t) * self.env.max_position
    x_lb[0] = x_ub[0] = init_position
    x_lb[-1] = x_ub[-1] = goal_position

    v_lb = np.ones(self.t) * self.env.max_speed * -1
    v_ub = np.ones(self.t) * self.env.max_speed
    v_lb[0] = v_ub[0] = init_velocity
    v_lb[-1] = v_ub[-1] = goal_velocity

    u_lb = np.ones(self.t) * self.env.min_action
    u_ub = np.ones(self.t) * self.env.max_action

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
      return res
    print("Optimization terminated")
    self.print_summary(res)
    return res

  def print_summary(self, res):
    print("Status: {0}\n Message: {1}".format(res.status, res.message))
    z_opt = res.x
    x_opt, v_opt, u_opt = np.split(z_opt, 3)
    print(x_opt[1:] - x_opt[:-1] - v_opt[:-1])
    print(v_opt[1:] - v_opt[:-1] - (u_opt[:-1] * self.env.power - 0.0025 * np.cos(3 * x_opt[:-1])))
    print("Positions: {0}\n Velocities: {1}\n Actions: {2}".format(x_opt, v_opt, u_opt))
    input("Press any key to watch simulation...")
    simulate(self.env, self.t, u_opt, "./log/direct_final.gif")

  def optim_callback_trust_constr(self, z, state):
    self.optim_iter += 1
    print("Iteration " + str(self.optim_iter) + ": " + str(state.constr_violation))
    if self.optim_iter % 50 == 0:
      np.save('./log/direct_action' + str(self.optim_iter) + '.npy', z[-self.t:])
      total_reward = simulate(self.env, self.t, actions=z[-self.t:])
      return total_reward >= 100
    return False

  def optim_callback_slsqp(self, z):
    self.optim_iter += 1
    print("Iteration " + str(self.optim_iter))
    if self.optim_iter % 50 == 0:
      np.save('./log/direct_action' + str(self.optim_iter) + '.npy', z[-self.t:])
      simulate(self.env, self.t, actions=z[-self.t:])

if __name__ == '__main__':
  Path("./log").mkdir(parents=True, exist_ok=True)
  env = gym.make('MountainCarContinuous-v0')
  env.reset()
  colloc = DirectCollocation(env, 150)
  colloc.solve('trust-constr')
  env.close()

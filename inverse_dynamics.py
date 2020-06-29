import gym
import math
import numpy as np
from pathlib import Path
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from utils import simulate

class InverseDynamicsCollocation():
  def __init__(self, env, time):
    self.env = env
    self.t = time
    self.optim_iter = 0
    # Define initial and goal states
    self.init_position = self.env.state[0]
    self.init_velocity = self.env.state[1]
    self.goal_position = self.env.goal_position
    self.goal_velocity = self.env.goal_velocity
    print("Initial position: {0}. Initial velocity: {1}".format(self.init_position, self.init_velocity))

  def guess(self):
    x0 = np.random.rand(self.t) - 0.5
    v0 = np.random.rand(self.t) - 0.5
    z0 = np.concatenate((x0, v0))
    return z0

  def get_actions(self, z):
    x, v = np.split(z, 2)
    u = np.zeros(self.t)
    u[:-1] = ((v[1:] - v[:-1]) + 0.0025 * np.cos(3 * x[:-1])) / self.env.power
    return u

  def objective(self, z):
    return np.sum(np.square(self.get_actions(z)))

  def dynamics_constraint(self):
    position_constraint_mat = np.concatenate((-np.eye(self.t) + np.eye(self.t, k=1), \
                                              -np.eye(self.t)), 1)[:-1]
    position_constraint = LinearConstraint(position_constraint_mat, 0, 0)
    velocity_constraint = NonlinearConstraint(self.get_actions, self.env.min_action, self.env.max_action)
    return [position_constraint, velocity_constraint]

  def solve(self):
    # Guess initial values of decision variables
    z0 = self.guess()

    # Dynamics constraints
    dyn_constr = self.dynamics_constraint()

    # Path and boundary constraints
    x_lb = np.ones(self.t) * self.env.min_position
    x_ub = np.ones(self.t) * self.env.max_position
    x_lb[0] = x_ub[0] = self.init_position
    x_lb[-1] = x_ub[-1] = self.goal_position

    v_lb = np.ones(self.t) * self.env.max_speed * -1
    v_ub = np.ones(self.t) * self.env.max_speed
    v_lb[0] = v_ub[0] = self.init_velocity
    v_lb[-1] = v_ub[-1] = self.goal_velocity

    z_lb = np.concatenate((x_lb, v_lb))
    z_ub = np.concatenate((x_ub, v_ub))
    z_bounds = Bounds(z_lb, z_ub)

    # Optimize
    print("Start optimization")
    res = minimize(self.objective, z0, method='trust-constr', bounds=z_bounds, \
                   constraints=dyn_constr, callback=self.optim_callback_trust_constr)
    print("Optimization terminated")
    self.print_summary(res)
    return res

  def reset_env(self):
    self.env.state[0] = self.init_position
    self.env.state[1] = self.init_velocity

  def print_summary(self, res):
    print("Status: {0}\n Message: {1}".format(res.status, res.message))
    z_opt = res.x
    x_opt, v_opt = np.split(z_opt, 2)
    u_opt = self.get_actions(z_opt)
    print("Positions: {0}\n Velocities: {1}\n Actions: {2}".format(x_opt, v_opt, u_opt))
    self.reset_env()
    simulate(self.env, u_opt, './log/inverse_final.gif')

  def optim_callback_trust_constr(self, z, state):
    self.optim_iter += 1
    print("Iteration " + str(self.optim_iter) + ": " + str(state.constr_violation))
    if self.optim_iter % 50 == 0:
      np.save('./log/inverse_action' + str(self.optim_iter) + '.npy', z[-self.t:])
      self.reset_env()
      total_reward = simulate(self.env, actions=z[-self.t:])
      return total_reward >= 100
    return False

if __name__ == '__main__':
  Path("./log").mkdir(parents=True, exist_ok=True)
  env = gym.make('MountainCarContinuous-v0')
  env.reset()
  colloc = InverseDynamicsCollocation(env, 200)
  colloc.solve()
  env.close()

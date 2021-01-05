import argparse
import numpy as np
import os
import pathlib
import sys

from glob import glob
from matplotlib import pyplot as plt

MW_PUSH_MAX_GOAL_DIST = 1.0
PM_OBSTACLE_MAX_GOAL_DIST = 7.0 # 3 + 1.5 + 2.5


def assign_pm_obstacle(init_state, res):
    # Assign to bin based on Manhattan distance
    goal_x, goal_y = 0.5, 1.5
    wall_y_min = -1
    x, y = init_state[:2]
    if x <= 0:
        goal_dist = (goal_x - x) + abs(y - wall_y_min) + (goal_y - wall_y_min)
    else:
        goal_dist = abs(goal_x - x) + abs(y - goal_y)
    interval = PM_OBSTACLE_MAX_GOAL_DIST / res
    return int(goal_dist % interval)


def assign_mw_push(init_state, res):
    # Returns an integer index
    pass


def generate_grid(filenames, assign_fn, res):
    # Returns array of length <res>
    rews = np.zeros(res)
    frqs = np.zeros(res)
    for filename in filenames:
        filename = pathlib.Path(filename).expanduser()
        with filename.open('rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
        init_state = episode['state'][0]
        ind = assign_fn(init_state, res)
        rews[ind] += episode['reward'].sum()
        frqs[ind] += 1
    # Avoid division by zero
    frqs[frqs == 0] = 1.0
    rews = rews / frqs
    return rews, frqs


def plot_grid(rews, frqs, title, figdir):
    plt.plot(rews)
    plt.title(title)
    plt.ylabel('Average rewards')
    plt.xlabel('Task difficulty')
    plt.savefig(figdir)


def eval_grid(config):
    filenames = sorted(glob(f'{config.logdir}/episodes/*.npz'))
    # Take the most recent episodes
    filenames = filenames[-config.num_episodes:]
    task = config.task
    res = config.resolution
    if 'pm_obstacle_long' in task:
        assign_fn = assign_pm_obstacle
    elif 'SawyerPushEnv' in task:
        assign_fn = assign_mw_push
    else:
        raise NotImplementedError(task)

    rews, frqs = generate_grid(filenames, assign_fn, res)
    plot_grid(rews, frqs, f'[{config.planner}] {task}', config.figdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dir = lambda x: pathlib.Path(x).expanduser()
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--logdir', type=dir, default='.')
    parser.add_argument('--figdir', type=dir, default='./out.jpg')
    parser.add_argument('--task', type=str, default='colloc_pm_obstacle_long_w1.5')
    parser.add_argument('--planner', type=str, default='')
    parser.add_argument('--resolution', type=int, default=10)
    eval_grid(parser.parse_args())

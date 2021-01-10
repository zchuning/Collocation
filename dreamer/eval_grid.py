import argparse
import numpy as np
import os
import pathlib
import sys

from glob import glob
from matplotlib import pyplot as plt

## TODO: replace max_dist with the maximum among episodes

MW_PUSH_MAX_GOAL_DIST = np.linalg.norm([0.2, 0.2]) + 0.2
MW_REACH_MAX_GOAL_DIST = 1.5 * np.linalg.norm([0.6, 0.4, 0.3])
PM_OBSTACLE_MAX_GOAL_DIST = 7.0 # 3 + 1.5 + 2.5

def get_task_config(task):
    if 'pm_obstacle_long' in task:
        assign_fn = assign_pm_obstacle
        ymax = PM_OBSTACLE_MAX_GOAL_DIST
        rew_key = 'reward'
    elif 'SawyerPushEnv' in task:
        assign_fn = assign_mw_push
        ymax = MW_PUSH_MAX_GOAL_DIST
        rew_key = 'sparse_reward'
    elif 'SawyerReachEnv' in task:
        assign_fn = assign_mw_reach
        ymax = MW_REACH_MAX_GOAL_DIST
        rew_key = 'sparse_reward'
    else:
        raise NotImplementedError(task)
    return assign_fn, ymax, rew_key


def assign_pm_obstacle(init_state, interval):
    # Assign to bin based on Manhattan distance
    goal_x, goal_y = 0.5, 1.5
    wall_y_min = -1
    x, y = init_state[:2]
    if x <= 0:
        goal_dist = (goal_x - x) + abs(y - wall_y_min) + (goal_y - wall_y_min)
    else:
        goal_dist = abs(goal_x - x) + abs(y - goal_y)
    return int(goal_dist / interval)


def assign_mw_push(init_state, interval):
    # Returns an integer index
    hand_pos, obj_pos, _, goal_pos = np.split(init_state, 4)
    reach_dist = np.linalg.norm(hand_pos - obj_pos)
    push_dist = np.linalg.norm(obj_pos[:2] - goal_pos[:2])
    goal_dist = reach_dist + push_dist
    return int(goal_dist / interval)


def assign_mw_reach(init_state, interval):
    hand_pos, _, goal_pos = np.split(init_state, [3, 9])
    goal_dist = np.linalg.norm(hand_pos - goal_pos)
    return int(goal_dist / interval)


def plot_grid(rew_list, frq_list, lbl_list, title, figdir, ymax):
    # rew_list, frq_list, lbl_list: n * resolution
    plt.title(title)
    bins = np.linspace(0, ymax, len(rew_list[0]))
    for rews, lbl in zip(rew_list, lbl_list):
        plt.plot(bins, rews, label=lbl)
    plt.legend()
    plt.ylabel('Success rate')
    plt.xlabel('Task difficulty')
    plt.savefig(figdir)


def create_grid(filenames, assign_fn, res, ymax, rew_key):
    # Returns array of length <res>
    rews = np.zeros(res)
    frqs = np.zeros(res)
    for filename in filenames:
        filename = pathlib.Path(filename).expanduser()
        with filename.open('rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
        init_state = episode['state'][0]
        interval = ymax / res
        ind = assign_fn(init_state, interval)
        rews[ind] += float(episode[rew_key].sum() > 0)
        frqs[ind] += 1
    # Avoid division by zero
    mask = (frqs != 0)
    rews[mask] = rews[mask] / frqs[mask]
    return rews, frqs


def eval_grid(config):
    task, res = config.task, config.resolution
    assign_fn, ymax, rew_key = get_task_config(task)
    rew_list, frq_list, lbl_list = [], [], []
    for logdir, method in zip(config.logdirs, config.methods):
        filenames = sorted(glob(f'{logdir}/episodes/*.npz'))
        assert config.num_episodes <= len(filenames), \
            'Inspected number of episodes greater than total number of episodes'
        # Take the most recent episodes
        filenames = filenames[-config.num_episodes:]
        rews, frqs = create_grid(filenames, assign_fn, res, ymax, rew_key)
        rew_list.append(rews)
        frq_list.append(frqs)
        lbl_list.append(method)
    plot_grid(rew_list, frq_list, lbl_list, task, config.figdir, ymax)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dir = lambda x: pathlib.Path(x).expanduser()
    parser.add_argument('--task', type=str, default='colloc_pm_obstacle_long')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--figdir', type=dir, default='./out.jpg')
    parser.add_argument('--resolution', type=int, default=10)
    parser.add_argument('--logdirs', type=dir, default='', nargs='+')
    parser.add_argument('--methods', type=str, default='', nargs='+')
    config = parser.parse_args()
    eval_grid(config)

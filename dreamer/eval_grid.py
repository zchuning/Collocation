import argparse
import numpy as np
import os
import pathlib
import sys

from glob import glob
from matplotlib import pyplot as plt

## TODO: replace max_dist with the maximum among episodes

MW_PUSH_MAX_HAND_HEIGHT = 0.2
MW_PUSH_MAX_OBJ_DIST = 0.32 # 0.25 # np.linalg.norm([0.2, 0.2])
MW_PUSH_MAX_GOAL_DIST = MW_PUSH_MAX_OBJ_DIST + MW_PUSH_MAX_HAND_HEIGHT
MW_REACH_MAX_GOAL_DIST = 0.72 # np.linalg.norm([0.6, 0.4, 0.3])
MW_HAMMER_MAX_GOAL_DIST = 0.7
MW_BUTTON_PRESS_MAX_GOAL_DIST = np.linalg.norm([0.5, 0.38, 0.38])
PM_OBSTACLE_MAX_GOAL_DIST = 7.0 # 3 + 1.5 + 2.5

def get_task_config(task):
    if 'pm_obstacle_long' in task:
        assign_fn = assign_pm_obstacle
        ymax = PM_OBSTACLE_MAX_GOAL_DIST
        rew_key = 'reward'
    elif 'SawyerPushEnv' in task:
        if 'hand' in task:
            assign_fn = assign_mw_push_hand
            ymax = MW_PUSH_MAX_HAND_HEIGHT
        elif 'obj' in task:
            assign_fn = assign_mw_push_obj
            ymax = MW_PUSH_MAX_OBJ_DIST
        else:
            assign_fn = assign_mw_push
            ymax = MW_PUSH_MAX_GOAL_DIST
        rew_key = 'sparse_reward'
    elif 'SawyerReachEnv' in task:
        assign_fn = assign_mw_reach
        ymax = MW_REACH_MAX_GOAL_DIST
        rew_key = 'sparse_reward'
    elif 'SawyerHammerEnv' in task:
        assign_fn = assign_mw_hammer
        ymax = MW_HAMMER_MAX_GOAL_DIST
        rew_key = 'sparse_reward'
    elif 'SawyerButtonPressEnv' in task:
        assign_fn = assign_mw_button_press
        ymax = MW_BUTTON_PRESS_MAX_GOAL_DIST
        rew_key = 'sparse_reward'
    else:
        raise NotImplementedError(task)
    return assign_fn, ymax, rew_key


def assign_pm_obstacle(init_state):
    # Assign to bin based on Manhattan distance
    goal_x, goal_y = 0.5, 1.5
    wall_y_min = -1
    x, y = init_state[:2]
    if x <= 0:
        goal_dist = (goal_x - x) + abs(y - wall_y_min) + (goal_y - wall_y_min)
    else:
        goal_dist = abs(goal_x - x) + abs(y - goal_y)
    return goal_dist


def assign_mw_push_hand(init_state):
    # Returns an integer index
    hand_pos, _, _, _ = np.split(init_state, 4)
    hand_height = hand_pos[2]
    return hand_height


def assign_mw_push_obj(init_state):
    # Returns an integer index
    #_, obj_pos, _, _ = np.split(init_state, 4)
    obj_pos = init_state[3:6]
    goal_pos = np.array([0.1, 0.8, 0.02])
    obj_dist = np.linalg.norm(obj_pos[:2] - goal_pos[:2])
    return obj_dist


def assign_mw_push(init_state):
    # Returns an integer index
    hand_pos, obj_pos, _, _ = np.split(init_state, 4)
    goal_pos = np.array([0.1, 0.8, 0.02])
    reach_dist = np.linalg.norm(hand_pos - obj_pos)
    push_dist = np.linalg.norm(obj_pos[:2] - goal_pos[:2])
    goal_dist = reach_dist + push_dist
    return goal_dist


def assign_mw_reach(init_state):
    # hand_pos, _, goal_pos = np.split(init_state, [3, 9])
    hand_pos = init_state[:3]
    goal_pos = np.array([-0.1, 0.8, 0.12])
    goal_dist = np.linalg.norm(hand_pos - goal_pos)
    return goal_dist

def assign_mw_button_press(init_state):
    hand_pos = init_state[:3]
    if hand_pos[0] < 0:
        return None
    goal_pos = np.array([0, 0.78, 0.2])
    goal_dist = np.linalg.norm(hand_pos - goal_pos)
    return goal_dist

def assign_mw_hammer(init_state):
    hammer_pos = init_state[3:6]
    nail_pos = init_state[6:9]
    hammer_dist = np.linalg.norm(hammer_pos - nail_pos)
    return hammer_dist


def plot_grid(rew_list, frq_list, lbl_list, title, figdir, ymax):
    # rew_list, frq_list, lbl_list: n * resolution
    # Cut off 0-frequency bins from the left
    cutoff = -1
    for i, frq_each in enumerate(zip(*frq_list)):
        if all(frq_each):
            cutoff = i
            break
    print(f'Cutoff : {cutoff}')

    # Plot success rates
    plt.figure(0)
    plt.title(title)
    bins = np.linspace(0, ymax, len(rew_list[0]))
    for rews, lbl in zip(rew_list, lbl_list):
        plt.plot(bins[cutoff:], rews[cutoff:], label=lbl)
    plt.legend()
    plt.ylabel('Success rate')
    plt.xlabel('Goal distance')
    plt.savefig(figdir)

    # Plot frequencies
    plt.figure(1)
    plt.title(title)
    bins = np.linspace(0, ymax, len(frq_list[0]))
    for frqs, lbl in zip(frq_list, lbl_list):
        plt.plot(bins[cutoff:], frqs[cutoff:], label=lbl)
    plt.legend()
    plt.ylabel('Frequency')
    plt.xlabel('Goal distance')
    plt.savefig(figdir.parent / (figdir.name[:-4] + '_frq.jpg'))


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
        difficulty = assign_fn(init_state)
        if difficulty is None:
            # print('Filtered out hard initialization')
            continue
        interval = ymax / res
        index = int(difficulty / interval)
        if index >= res:
            print('Omitted out-of-bound initialization')
            continue
        rews[index] += float(episode[rew_key].sum() > 0)
        frqs[index] += 1
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

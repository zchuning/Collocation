import os
import argparse
import gym
import numpy as np
import d4rl
from blox import AttrDict, rmap
import imageio
from scipy.signal import butter, filtfilt
from tqdm import tqdm


def get_video(qpos, qvel):
    frames = []
    env.reset()
    env.set_state(qpos[0], qvel[0])
    for t in range(qpos.shape[0]):
        env.set_state(qpos[t], qvel[t])
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
    # Here's some code to set up camera: run the human rendering, manipulate the camera manually, and save the result.
    # env.render()
    # import pdb; pdb.set_trace()
    # cam = env.env.sim_robot.renderer._window.camera
    # cam.distance, cam.lookat, cam.azimuth, cam.elevation
    
    return np.array(frames)


def smooth(data):
    """ butter_lowpass_filter,
    https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7 """
    # A bunch of magic constants
    # n = 300
    fs = 30
    cutoff = fs / 10
    nyq = 0.5 * fs
    order = 2
    
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='kitchen-partial-v0')
    parser.add_argument('--save_dir', type=str, default='./temp/episodes/')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    env = gym.make(args.env_name)
    
    dataset = env.get_dataset()
    qpos = dataset['observations'][:, :30]
    qvel = np.zeros_like(qpos)[:, :29]
    rewards = dataset['rewards']
    actions = dataset['actions']
    print(dataset['terminals'].shape[0] / dataset['terminals'].sum(), dataset['terminals'].sum())
    
    
    # import pdb; pdb.set_trace()
    episode_ends = tuple(dataset['terminals'].nonzero()[0] + 1)
    episode_starts = (0,) + episode_ends[:-1]
    
    episodes = []
    for e in tqdm(range(len(episode_starts))):
        start = episode_starts[e]
        end = episode_ends[e]
        episode = AttrDict()
        # end = 1
    
        # TODO float16
        episode.image = get_video(smooth(qpos[start:end]),  qvel[start:end])
        episode.state = np.concatenate((smooth(qpos[start:end]),  qvel[start:end]), 1)
        episode.reward, episode.action = rewards[start:end], actions[start:end]
        episode.discount = np.ones((end - start,))
        episodes.append(episode)
    
        # break
        # imageio.mimwrite('test_kitchen.gif', episode.image)
        episode = rmap(lambda x: x.astype(np.float16), episode)
        episode.image = episode.image.astype(np.uint8)
        np.savez_compressed(args.save_dir + str(e).zfill(5), **episode)
        
        # import pdb; pdb.set_trace()
        # break
        # import pdb; pdb.set_trace()
        
        
    
    # get_video()
    
        
        
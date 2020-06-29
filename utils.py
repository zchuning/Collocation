import gym
import imageio

def simulate(env, time, actions=None, path=None):
  total_reward = 0
  frames = []
  if not (actions is None):
    assert(time == len(actions))
  for i in range(time):
    if not (path is None):
      frames.append(env.render(mode='rgb_array'))
    action = env.action_space.sample() if actions is None else [actions[i]]
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
      break
  if not (path is None):
    imageio.mimsave(path, frames, fps=60)
  print("Total reward: " + str(total_reward))
  return total_reward

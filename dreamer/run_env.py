from wrappers import DreamerMujocoEnv
# from tools import AttrDict
import numpy as np
import skvideo.io
# import blox

class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__
  
  
if __name__ == "__main__":
    env = DreamerMujocoEnv()
    act_space = env.action_space
    
    # TODO Dreamer code has some more magic with Async and other wrappers
    
    def get_video(steps=10, a=np.zeros(2)):
        obs = env.reset()
        # goal_obs = env.render_goal() # No goal for pointmass
        
        execution = [obs['image']]
        
        for i in range(steps):
            obs, reward, done, _ = env.step(a)
            execution.append(obs['image'])
        
        video = np.stack(execution)
    
        skvideo.io.vwrite('./logdir/temp/pointmass.gif', video, outputdict={'-r': '10'})

    a = np.zeros(2)  # TODO
    get_video(10, a)
    
    
    env.close()
    
    
    def async_version():
        obs = env.reset(blocking=False)()
        a = 1
        obs, _, done = env.step(a)()
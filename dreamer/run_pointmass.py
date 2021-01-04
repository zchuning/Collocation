import numpy as np
import mujoco_py
import imageio
from envs.pointmass.pointmass_long import PointmassLong
from envs.pointmass.pointmass_smart_env import Pointmass as PointmassSmart
from envs.pointmass.pointmass_hard_env import PointmassHard

if __name__ == '__main__':
    env = PointmassLong(0.5, True)
    # env = PointmassSmart()
    # env = PointmassHard()
    env.reset()

    # Debug camera
    viewer = mujoco_py.MjViewer(env.sim)
    # viewer.cam.distance = 3
    # viewer.cam.azimuth = 135
    viewer.cam.azimuth = 0
    viewer.cam.elevation = 90
    viewer.cam.distance = 3
    viewer.cam.lookat[0] = 0
    viewer.cam.lookat[1] = 0
    viewer.cam.lookat[2] = 0
    viewer.render()
    for i in range(10000):
        # if i % 100 == 0:
        # env.reset()

        a = env.action_space.sample()
        a = a / np.abs(a)
        a = np.asarray([-1, -1])
        # print(a)
        # env.step(a)
        # print(env.get_endeff_pos())
        # env.sim.step()
        viewer.render()
        # print(viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance, viewer.cam.lookat)
    
    import pdb; pdb.set_trace()
    
    # Debug data collection
    for j in range(10):
        env.reset()
        
        images = []
        for i in range(100):
            a = np.random.rand(3) * 2 - 1
            env.step(a)
            image = env.render()[0]
            images.append(image)

        imageio.mimwrite(f'test_{j}.gif', images)


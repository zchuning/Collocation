from envs.franka_desk.franka_desk import FrankaDesk
import numpy as np
import mujoco_py
import imageio

if __name__ == '__main__':

    env_params = {
        # resolution sufficient for 16x anti-aliasing
        'viewer_image_height': 192,
        'viewer_image_width': 256,
        'textured': True,
    }
    env = FrankaDesk(env_params)
    env.reset()
    

    # Debug camera
    viewer = mujoco_py.MjViewer(env.sim)
    viewer.cam.distance = 3
    viewer.cam.azimuth = 135
    for i in range(10000):
        if i % 100 == 0:
            env.reset()
        env.sim.step()
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


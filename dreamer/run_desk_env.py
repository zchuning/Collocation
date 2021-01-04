import numpy as np
import mujoco_py
import imageio
from tqdm import tqdm
import blox
import blox.mujoco


class GtPointmassModel():
    def __init__(self, env):
        self.env = env
        self._env = env._env
        self.std = 0.001
    
    def _dynamics(self, s, a):
        # Use mujoco
        self._env.set_state(s, np.zeros_like(s))
        self._env.do_simulation(a, self._env.frame_skip)
        s_prime = self._env.data.qpos
        
        from tensorflow_probability import distributions as tfd
        import tensorflow as tf
        # Approximate with a linear model
        return tfd.Independent(tfd.Normal(s - tf.stop_gradient(s) + s_prime, self.std), 1)
    
    def _reward(self, s):
        goal = self._env.model.site_pos[self._env.target_sid][:self._env.goal_dim]
        import tensorflow as tf
        from tensorflow_probability import distributions as tfd
        return tfd.Normal(tf.reduce_sum((s - goal) ** 2), 1)
    
    def _decode(self, s):
        self._env.set_state(s, np.zeros_like(s))
        return self.env._get_obs(None)['image']

if __name__ == '__main__':

    env_params = {
        # resolution sufficient for 16x anti-aliasing
        'viewer_image_height': 192,
        'viewer_image_width': 256,
        'textured': True,
        # 'frame_skip': 5,
        # 'action_scale': 1./5
    }
    # from envs.franka_desk.franka_desk import FrankaDesk
    # env = FrankaDesk(env_params)

    # Multiworld
    # from utils.wrappers import MultiWorld
    # env_p = MultiWorld('push', 1)
    # env = env_p._env

    from utils.wrappers import MultiTaskMetaWorld
    from utils.wrappers import MetaWorld
    # env = MultiTaskMetaWorld('sawyer_SawyerPushEnvV2', 1)
    env = MetaWorld('sawyer_SawyerHammerEnvV2', 1)
    env._env.max_path_length = np.inf
    env.reset()
    
    
    # import matplotlib.pyplot as plt
    # for i in tqdm(range(1000)):
    #     env.reset()
    #     goal = env.render_goal()['image']
    #     plt.subplot(4, 4, i+1)
    #     plt.imshow(goal)
    # plt.show()

    # env._env.hand_init_pos = env._env.goal = np.array((-0.2, 0.4, 0.2))
    # env._env.hand_init_pos = env._env.goal
    # env._env.reset()
    # env._env._reset_hand()
    # env._env._reset_hand()
    # env._env._reset_hand()
    # env._env._reset_hand()
    
    # import pdb; pdb.set_trace()
    
    # Debug camera
    viewer = mujoco_py.MjViewer(env._env.sim)
    # viewer.cam.azimuth = -90
    # viewer.cam.elevation = -22
    # viewer.cam.distance = 0.82
    # viewer.cam.lookat[0] = 0.
    # viewer.cam.lookat[1] = 0.55
    # viewer.cam.lookat[2] = 0.
    # viewer.cam.distance = 3
    # viewer.cam.azimuth = 135
    # blox.mujoco.set_camera(
    #     viewer.cam, azimuth=90, elevation=41 + 180, distance=0.61, lookat=[0., 0.55, 0.])
    viewer.render()
    viewer._paused = True
    for i in range(10000):
        # if i % 100 == 0:
        #     env.reset()

        a = env.action_space.sample()

        env.step(a)
        # if i == 0:
        #     env.remove_markers()
        # env.sim.step()
        viewer.render()
        # import pdb; pdb.set_trace()
        # env._env.data.site_xpos[env._env.model.site_name2id('goal_reach'), 2] = (-1000)
        # import pdb; pdb.set_trace()
        # print(viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance, viewer.cam.lookat)
    #
    # import pdb; pdb.set_trace()
    #
    # # Debug data collection
    # for j in range(10):
    #     env.reset()
    #
    #     images = []
    #     for i in range(100):
    #         a = np.random.rand(3) * 2 - 1
    #         env.step(a)
    #         image = env.render()[0]
    #         images.append(image)
    #
    #     imageio.mimwrite(f'test_{j}.gif', images)



    if False:
        # Multiworld
        from utils.wrappers import MultiWorld
    
        env_p = MultiWorld('push', 1)
        env = env_p._env
    
        env.reset()
    
        goal = env_p.render_goal()['image']
        import matplotlib.pyplot as plt
    
        plt.imshow(goal)
        plt.show()
        
        
    if False:
        # Ground truth pointmass
        # from gt_models.gt_pointmass_mj import GtPointmassMj
        # env = GtPointmassMj()
        from utils.wrappers import DreamerMujocoEnv
        from colloc_gt import GtPointmassModel
    
        env = DreamerMujocoEnv('pm_obstacle_long')
        model = GtPointmassModel(env)
    
        import tensorflow as tf
    
        with tf.GradientTape() as tape:
            st = tf.convert_to_tensor(np.array((0.5, 0)))
            tape.watch(st)
            rew = model._reward(st).mode()
        tape.gradient(rew, st)
    
        ac = np.array((1, 1))
        s_p = model._dynamics(st, ac).mode()
        for i in range(10):
            s_p = model._dynamics(s_p, ac).mode()
        rew = model._reward(s_p)
        im_p = model._decode(s_p)
        rew = model._reward(s_p.numpy())
        im_p = model._decode(s_p.numpy())
        # im = model._decode(st)
    
        import tensorflow as tf
    
        with tf.GradientTape() as tape:
            st = tf.convert_to_tensor(np.array((0.5, 0)))
            tape.watch(st)
            s_p = model._dynamics(st, ac).mode()
        tape.gradient(s_p, st)
    
        import matplotlib.pyplot as plt
        # plt.imshow(im)
        # plt.show()
    
        # plt.imshow(im_p)
        # plt.show()
        import pdb;
    
        pdb.set_trace()
        
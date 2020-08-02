import imageio
import numpy as np


def npy2gif(dir, file, imgs):
    # Mark beginning
    imgs = np.concatenate([np.zeros_like(imgs[[0]]), imgs], 0)
    imageio.mimsave(dir / file, imgs, fps=10)
    
    # TODO log strips
    
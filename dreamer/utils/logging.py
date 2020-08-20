import tensorflow as tf
import numpy as np

import imageio
import tools


class Logger():
  pass


class TBLogger(Logger):
  def __init__(self, dir):
    writer = tf.summary.create_file_writer(str(dir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    self.writer = writer
  
  def log_graph(self, name, curves):
    for curve_name, curve in curves.items():
      for i, v in enumerate(curve):
        tf.summary.scalar(curve_name, v, i)
    self.writer.flush()
  
  def log_image(self, name, image):
    """
    
    :param name:
    :param image: HxWxC
    :return:
    """
    if np.issubdtype(image.dtype, np.floating):
      image = image + 0.5
    tools.image_summary(name, image[None])
    # tools.graph_summary(self.writer, tools.image_summary, name, image)
    self.writer.flush()
  
  def log_video(self, name, video):
    """
    
    :param name:
    :param video: (B x) T x H x W x C
    :return:
    """
    if isinstance(video, list):
      video = np.array(video)
    if np.issubdtype(video.dtype, np.floating):
      video = video + 0.5
    tools.video_summary(name, video)
    # tools.graph_summary(self.writer, tools.video_summary, name, video)
    self.writer.flush()


class DiskLogger(Logger):
  def __init__(self, dir):
    self.dir = dir
  
  def log_image(self, name, image):
    imageio.imwrite(self.dir / f'{name}.jpg', image)
  
  def log_video(self, name, video):
    imageio.mimwrite(self.dir / f'{name}.gif', video, fps=10)
  
  def log_graph(self, name, curves):
    import matplotlib.pyplot as plt
    plt.title(name)
    for k, v in curves.items():
      plt.plot(range(len(v)), v, label=k)
    plt.legend()
    plt.savefig(self.dir / f'{name}.jpg')
    plt.show()
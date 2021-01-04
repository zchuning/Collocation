# import tensorflow as tf

import pathlib
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os


# Debug CSV

data = np.loadtxt('colloc_push_return2.csv', delimiter=",",skiprows=1)
rew = data[:,2]

trained_rew = rew[np.argmax(rew>0):]
print((trained_rew > 0).mean())

import pdb; pdb.set_trace()
plt.plot(trained_rew)
plt.show()

import pdb; pdb.set_trace()

if False:
  # debug episodes
  directory = pathlib.Path('./logdir/mw_reach/random/latcog_online/episodes').expanduser()
  filenames = directory.glob('*.npz')
  
  for filename in tqdm.tqdm(directory.glob('*.npz')):
    with filename.open('rb') as f:
      episode = np.load(f)
      episode = {k: episode[k] for k in episode.keys()}
      
      print(episode['state'].shape)
      # if episode['state'].shape[-1] != 9:
      #   # import pdb; pdb.set_trace()
      #   os.remove(filename)
      # import pdb; pdb.set_trace()
    
    
if False:
  # debug autograph
  def f(x):
    # for x in range(10):
    y = 0
    # for x in tf.constant((1,1,1,1,)):
    l = []
    for x in tf.range(10):
      print('hi')
      x = x + 1
      y = x + 1
      l.append(x)
    return y, l
  
  tf_f = tf.function(f)
  tf_f(0)
  print('hhi')
  tf_f(0)
  
  # print(tf.autograph.to_code(f))
  
  import pdb; pdb.set_trace()
  
  
  # Debug learning a constant
  # class X(tf.Module):
  #   def __init__(self):
  #     self.x = tf.Variable(0., name='bias')
  #
  # x = X()
  #
  # with tf.GradientTape() as tape: y = x.x + 1; print(tape.gradient(y, x.x))
  #
  # import pdb; pdb.set_trace()
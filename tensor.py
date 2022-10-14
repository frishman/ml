import tensorflow as tf
import numpy as np

rank_0_tensor = tf.constant(4)
print(rank_0_tensor.numpy())
rank_1_tensor = tf.constant([2.1, 3.3, 4.9])
print(rank_1_tensor.numpy()[0])
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor.numpy()[0][0])
print(rank_2_tensor.shape)



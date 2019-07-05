# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This tutorial uses deep learning to compose one image in the style of
another image (ever wish you could paint like Picasso or Van Gogh?).
This is known as neural style transfer and the technique is outlined in
A Neural Algorithm of Artistic Style (Gatys et al.)."""

import time
import functools

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

content_path = tf.keras.utils.get_file('turtle.jpg', 'https://storage.googleapis.com/download.tensorflow.org'
                                                     '/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
style_path = tf.keras.utils.get_file('kandinsky.jpg', 'https://storage.googleapis.com/download.tensorflow.org'
                                                      '/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


# Visualize the input
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, dtype=tf.float32)

  # get img size.
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  # update img shape
  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


# Create a simple function to display an image
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)




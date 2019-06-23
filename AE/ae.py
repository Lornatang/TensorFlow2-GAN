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

""" Autoencoder (AE)
A simple autoencoder network.
"""

import glob
import os
import time

import tensorflow as tf
from tensorflow.python.keras import layers

import imageio
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display

# check tf version
assert tf.__version__ == '2.0.0-beta1'
"""TensorFlow version must equal '2.0.0-beta1', 
   please run `pip install -q tensorflow-gpu==2.0.0-beta1"""

# Load and prepare the dataset
"""You will use the MNIST dataset to train the generator and the discriminator. 
The generator will generate handwritten digits resembling the MNIST data."""

BUFFER_SIZE = 60000
BATCH_SIZE = 256

EPOCHS = 50
noise_dim = 64
num_examples_to_generate = 16

img_shape = (28, 28, 1)

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def load_data(buffer_size, batch_size):
  """

  Returns:
    tf.keras.datasets.fashion_mnist

  """

  # load datasets
  (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

  # split datasets
  train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
  train_images = train_images / 255.0  # Normalize the images to [0, 1]

  # Batch and shuffle the data
  train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
      .shuffle(buffer_size)
      .batch(batch_size)
  )
  return train_dataset


class AE(tf.keras.Model):
  """ a basic autoencoder class for tensorflow.

    Returns
      tf.keras.Model
  """

  def __init__(self, **kwargs):
    super(AE, self).__init__()
    self.__dict__.update(kwargs)

    self.dims = img_shape
    self.z = noise_dim

    self.optimizer = tf.optimizers.Adam()

    self.enc = tf.keras.Sequential([
      layers.InputLayer(input_shape=self.dims),

      layers.Conv2D(32, (3, 3), strides=(2, 2)),
      layers.ReLU(),
      layers.Conv2D(64, (3, 3), strides=(2, 2)),
      layers.ReLU(),

      layers.Flatten(),
      layers.Dense(units=self.z)]
    )
    self.dec = tf.keras.Sequential([
      layers.Dense(units=7 * 7 * 64, input_shape=(self.z,)),
      layers.ReLU(),

      layers.Reshape(target_shape=(7, 7, 64)),

      layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'),
      layers.ReLU(),
      layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'),
      layers.ReLU(),
      layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same',
                             activation='sigmoid')]
    )

  @tf.function
  def encoder(self, x):
    return self.enc(x)

  @tf.function
  def decoder(self, z):
    return self.dec(z)

  # This annotation causes the function to be "compiled".
  @tf.function
  def train_step(self, x):
    """ break it down into training steps.

    Args:
      x: input images.

    """
    with tf.GradientTape() as tape:
      # computer loss
      z = self.encoder(x)
      _x = self.decoder(z)
      loss = tf.reduce_mean(tf.square(x - _x))

    # compute gradients
    gradients = tape.gradient(loss, self.trainable_variables)
    # optimizer loss
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


# Create the models
model = AE()

# Save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(decoder=model.decoder,
                                 encoder=model.encoder,
                                 decoder_optimizer=model.optimizer,
                                 encoder_optimizer=model.optimizer)


def train(dataset, epochs):
  """ train op

    Args:
      dataset: mnist dataset or cifar10 dataset.
      epochs: number of iterative training.

    """
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      model.train_step(image_batch)

    # Produce images for the GIF as we go
    generate_and_save_images(model.decoder,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'Time for epoch {epoch + 1} is {time.time() - start:.3f} sec.')

  # Generate after the final epoch
  generate_and_save_images(model.decoder,
                           epochs,
                           seed)


# Generate and save images
def generate_and_save_images(model, epoch, test_input):
  """ Notice `training` is set to False.
  This is so all layers run in inference mode (batchnorm).

  Args:
    model: Models to be saved during training.
    epoch: How many training cycles?
    test_input: The test generates model output.

  Returns:
    matplotlib.pyplot.figure.

  """
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

  plt.savefig(checkpoint_dir + '/' + 'image_at_epoch_{:04d}.png'.format(epoch))
  plt.close(fig)


# Create a GIF
# Display a single image using the epoch number
def display_image(epoch_no):
  return Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def create_gif(file_name):
  with imageio.get_writer(file_name, mode='I') as writer:
    filenames = glob.glob(checkpoint_dir + '/' + 'image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
      frame = 2 * (i ** 0.5)
      if round(frame) > round(last):
        last = frame
      else:
        continue
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

  import IPython
  if IPython.version_info > (6, 2, 0, ''):
    display.Image(filename=file_name)


if __name__ == '__main__':
  train_images = load_data(BUFFER_SIZE, BATCH_SIZE)
  train(train_images, epochs=50)

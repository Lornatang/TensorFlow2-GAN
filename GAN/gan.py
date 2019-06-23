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

""" Generative Adversarial Network (GAN)
GANs are a form of neural network in which two sub-networks (the encoder and decoder)
are trained on opposing loss functions:
an encoder that is trained to produce data which is indiscernable from the true data,
and a decoder that is trained to discern between the data and generated data."""

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
noise_dim = 100
num_examples_to_generate = 16

img_shape = (28, 28, 1)

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def load_data(buffer_size, batch_size):
  """

  Returns:
    tf.keras.datasets.fashion_mnist

  """

  # load datasets
  (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

  # split datasets
  train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
  train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

  # Batch and shuffle the data
  train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
      .shuffle(buffer_size)
      .batch(batch_size)
  )
  return train_dataset


def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss


class GAN(tf.keras.Model):
  """ a basic gan class for tensorflow.

    Returns
      tf.keras.Model
  """

  def __init__(self, **kwargs):
    super(GAN, self).__init__()
    self.__dict__.update(kwargs)

    self.dims = img_shape
    self.z = noise_dim

    self.generator_optimizer = tf.optimizers.Adam(lr=3e-1, beta_1=0.5)
    self.discriminator_optimizer = tf.optimizers.RMSprop(lr=5e-1)

    self.gen = tf.keras.Sequential([
      layers.Dense(units=7 * 7 * 64, input_shape=(noise_dim, )),
      layers.ReLU(),

      layers.Reshape(target_shape=(7, 7, 64)),

      layers.Conv2DTranspose(64, (3, 3), strides=(2, 2),
                             padding='same'),
      layers.ReLU(),
      layers.Conv2DTranspose(32, (3, 3), strides=(2, 2),
                             padding='same'),
      layers.ReLU(),
      layers.Conv2DTranspose(1, (3, 3), strides=(1, 1),
                             padding='same')]
    )
    self.disc = tf.keras.Sequential([
      layers.InputLayer(input_shape=self.dims),
      layers.Conv2D(32, (3, 3),
                    strides=(2, 2),
                    padding='same'),
      layers.ReLU(),
      layers.Conv2D(64, (3, 3),
                    strides=(2, 2),
                    padding='same'),
      layers.ReLU(),

      layers.Flatten(),
      layers.Dense(units=1, activation=None)]
    )

  def generator(self, z):
    return self.gen(z)

  def discriminator(self, x):
    return self.disc(x)

  # This annotation causes the function to be "compiled".
  @tf.function
  def train_step(self, images):
    """ break it down into training steps.

    Args:
      images: input images.

    """
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.generator(noise)

      real_output = self.discriminator(images)
      fake_output = self.discriminator(generated_images)

      gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

      real_loss = cross_entropy(tf.ones_like(real_output), real_output)
      fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
      disc_loss = real_loss + fake_loss

    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               self.gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                    self.disc.trainable_variables)

    self.generator_optimizer.apply_gradients(
      zip(gradients_of_generator, self.gen.trainable_variables))
    self.discriminator_optimizer.apply_gradients(
      zip(gradients_of_discriminator, self.disc.trainable_variables))


# Create the models
model = GAN()

# Save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=model.generator_optimizer,
                                 discriminator_optimizer=model.discriminator_optimizer,
                                 generator=model.gen,
                                 discriminator=model.disc)


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
    generate_and_save_images(model.generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'Time for epoch {epoch + 1} is {time.time() - start:.3f} sec.')

  # Generate after the final epoch
  generate_and_save_images(model.generator,
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

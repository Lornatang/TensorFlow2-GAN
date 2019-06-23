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


# Both the generator and discriminator are defined using the Keras Sequential API.
def make_generator_model():
  """

  Returns:
    tf.keras.Model
  """
  model = tf.keras.Sequential()
  model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((7, 7, 256)))
  assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 7, 7, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 14, 14, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  assert model.output_shape == (None, 28, 28, 1)

  return model


generator = make_generator_model()


# The discriminator is a CNN-based image classifier.
def make_discriminator_model():
  """

    Returns:
      tf.keras.Model
    """
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                          input_shape=[28, 28, 1]))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model


discriminator = make_discriminator_model()

# Define the loss and optimizers
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Generator loss
def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)


# Discriminator loss
def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss


# The discriminator and the generator optimizers are different since
# we will train two networks separately.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


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
      layers.Dense(units=7 * 7 * 256, input_shape=(noise_dim,)),
      layers.ReLU(),

      layers.Reshape(target_shape=(7, 7, 256)),

      layers.Conv2DTranspose(256, (3, 3), strides=(2, 2),
                             padding='same'),
      layers.ReLU(),
      layers.Conv2DTranspose(128, (3, 3), strides=(2, 2),
                             padding='same'),
      layers.ReLU(),
      layers.Conv2DTranspose(1, (3, 3), strides=(1, 1),
                             padding='same',
                             activation='sigmoid')]
    )
    self.disc = tf.keras.Sequential([
      layers.InputLayer(input_shape=self.dims),
      layers.Conv2D(64, (3, 3),
                    strides=(2, 2),
                    padding='same'),
      layers.ReLU(),
      layers.Conv2D(128, (3, 3),
                    strides=(2, 2),
                    padding='same'),
      layers.ReLU(),

      layers.Flatten(),
      layers.Dense(units=1, activation=None)]
    )

  @tf.function
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


# Save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# create dir
if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
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
  predictions = model(test_input)

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
  create_gif('gan.gif')

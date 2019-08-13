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

"""Auto Encoder (AE) In 1986 Rumelhart put forward the concept of automatic
encode rand applied it to high dimensional complex data processing, which
promoted the development of neural network. Self-coding neural network is
an unsupervised learning algorithm, which USES the back-propagation algorithm
and makes the target value equal to the input value."""

import glob
import os
import time

import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import utils

from IPython import display
from PIL import Image

# check tf version
if not tf.__version__ == '2.0.0-beta1':
  raise Exception("TensorFlow version must equal '2.0.0-beta1', "
                  "please run `pip install -q tensorflow-gpu==2.0.0-beta1")

# Load and prepare the dataset
"""You will use the MNIST dataset to train the generator and the discriminator.
The generator will generate handwritten digits resembling the MNIST data."""

BUFFER_SIZE = 60000
BATCH_SIZE = 128

EPOCHS = 200
noise_dim = 64
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def load_data(buffer_size, batch_size):
  """

  Returns:
    tf.keras.datasets.fashion_mnist

  """

  # load datasets
  (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

  # split datasets
  train_images = train_images.reshape(
      train_images.shape[0], 28, 28, 1).astype('float32')
  # Normalize the images to [-1, 1]
  train_images = (train_images - 127.5 / 127.5)

  # Batch and shuffle the data
  train_dataset = (
      tf.data.Dataset.from_tensor_slices(train_images)
      .shuffle(buffer_size)
      .batch(batch_size)
  )
  return train_dataset


# Both the generator and discriminator are defined using the Keras
# Sequential API.
def make_encoder_model(input_tensor=None,
                       input_shape=(noise_dim,)):
  """ Create a build model
  Args:
    input_tensor: It's going to be tensor, which is going to be the array
    input_shape:  Noise data that conforms to normal distribution
  Returns:
    tf.keras.Model
  """
  if input_tensor is None:
      img_input = layers.Input(shape=input_shape)
  else:
      if not backend.is_keras_tensor(input_tensor):
          img_input = layers.Input(tensor=input_tensor, shape=input_shape)
      else:
          img_input = input_tensor
  x = layers.Conv2D(32, (3, 3),
                    strides=(2, 2),
                    activation=tf.nn.relu,
                    padding="SAME",
                    name="conv1")(img_input)
  x = layers.Conv2D(64, (3, 3),
                    strides=(2, 2),
                    activation=tf.nn.relu,
                    padding="SAME",
                    name="conv2")(x)
  x = layers.Flatten(name="flat")(x)
  x = layers.Dense(units=noise_dim, name="den1")(x)

  if input_tensor is not None:
      inputs = utils.get_source_inputs(input_tensor)
  else:
      inputs = img_input

  model = models.Model(inputs, x, name='Encoder_model')
  return model


encoder = make_encoder_model()
encoder.summary()


# The discriminator is a CNN-based image classifier.
def make_decoder_model(input_tensor=None,
                       input_shape=(28, 28, 1)):
  """ Create a decoder model
  Args:
    input_tensor: It's going to be tensor, which is going to be the array
    input_shape:  image tensor size.
  Returns:
    tf.keras.Model
  """
  if input_tensor is None:
      img_input = layers.Input(shape=input_shape)
  else:
      if not backend.is_keras_tensor(input_tensor):
          img_input = layers.Input(tensor=input_tensor, shape=input_shape)
      else:
          img_input = input_tensor

  x = layers.Dense(7 * 7 * 64, activation=tf.nn.relu, name="den1")(img_input)
  x = layers.Reshape(target_shape=(7, 7, 64), name="reshape1")(x)
  x = layers.Conv2DTranspose(64, (3, 3),
                             strides=(2, 2),
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="convt1")(x),
  x = layers.Conv2DTranspose(32, (3, 3),
                             strides=(2, 2),
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="convt2")(x),
  x = layers.Conv2DTranspose(1, (3, 3),
                             strides=(1, 1),
                             padding="SAME",
                             activation=tf.nn.sigmoid,
                             name="convt3")(x),

  if input_tensor is not None:
      inputs = utils.get_source_inputs(input_tensor)
  else:
      inputs = img_input

  model = models.Model(inputs, x, name='Decoder_model')
  return model


decoder = make_decoder_model()
decoder.summary()


# AE loss
def compute_loss(real_output, fake_output):
  """ compute autoencoder loss.
  Args:
     real_output:
     fake_output
  Returns:
  """
  ae_loss = tf.reduce_mean(real_output - fake_output)
  return ae_loss


# The discriminator and the generator optimizers are different since
# we will train two networks separately.
generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

# Save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=decoder,
                                 discriminator=encoder)

# create dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    """

    Args:
        images:
    """
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = decoder(noise, training=True)

        real_output = encoder(images, training=True)
        fake_output = encoder(generated_images, training=True)

        ae_loss = compute_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        ae_loss, decoder.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        ae_loss, encoder.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, decoder.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, encoder.trainable_variables))


def train(dataset, epochs):
    """

    Args:
        dataset:
        epochs:
    """
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        generate_and_save_images(decoder,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Time for epoch {epoch + 1} is {time.time() - start:.4f} sec')

    # Generate after the final epoch
    generate_and_save_images(decoder,
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
    save_path = os.path.join(checkpoint_dir, f"image_at_epoch_{epoch:04d}.png")
    plt.savefig(save_path)
    plt.close(fig)


# Create a GIF
# Display a single image using the epoch number
def display_image(epoch_no):
    """

    Args:
        epoch_no:

    Returns:

    """
    return Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def create_gif(file_name):
    """

    Args:
        file_name:
    """
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
    dataset = load_data(BUFFER_SIZE, BATCH_SIZE)
    train(dataset, epochs=EPOCHS)
    create_gif('gan.gif')

# This is the main deep learning library we're using.
import shutil
import time
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# Print TensorFlow version to ensure compatibility
print(tf.__version__)
# List and print if any GPUs are available for training
print(tf.config.list_physical_devices('GPU'))
# Function to test for GPU availability
print(tf.test.is_gpu_available())
# Used to create plots of outputted images


def build_generator(z_dim):
    """
    Constructs the generator model with a given latent dimension (z_dim).

    Args:
    z_dim (int): Size of the latent space vector.

    Returns:
    tf.keras.Model: The generator model, which takes a latent space vector as
    input and outputs a 64x64 RGB image.
    """

    # The generator model is a Sequential model, which means that the output
    # of each layer is the input to the next layer.
    model = tf.keras.Sequential()

    # The first layer is a Dense layer that takes a latent space vector of
    # size z_dim (a vector of size z_dim full of random nosie) and maps it to
    # a larger, flattened, dense representation (of size 8*8*512). This is the
    # foundation that we'll build upon to eventually form an image.
    # The reason for not using a bias is to manually control the output and
    # let the batch normalization handle the offset. Bias terms are additional
    # parameters in neural network layers that allow the model to learn an
    # offset value for each neuron's output. This helps the model fit more
    # complex patterns in the data by allowing it to shift and scale the
    # activation functions. Without bias terms, the model's capacity to learn
    # might be limited, as it would only be able to learn patterns that pass
    # through the origin (i.e., have a zero output when all inputs are zero).
    model.add(
        tf.keras.layers.Dense(
            8 * 8 * 512,
            use_bias=False,
            input_shape=(
                z_dim,
            )))

    # BatchNormalization normalizes the output of the previous layer
    # by adjusting and scaling the activations to have a mean of zero
    # and a standard deviation of one, thereby stabilizing the learning
    # process. It includes learnable parameters that scale (gamma) and shift
    # (beta) these normalized values, which effectively allows the network
    # to undo the normalization if that's beneficial. This negates the need
    # for a separate bias term because the shift (beta) can serve the same
    # purpose. Applying Batch Normalization before non-linear activation
    # functions, such as LeakyReLU, helps prevent the network activations
    # from saturating during training when inputs could become very large or
    # very small. Essentially, Batch Normalization reduces the
    # model's dependence on the initial scale and distribution of inputs,
    # which are typically compensated for by bias terms. This makes the
    # network less sensitive to input scaling and
    # shifting, and the internal shift (beta) parameter allows the network
    # to still produce non-zero outputs when all inputs are zero,
    # maintaining the ability to learn complex patterns without the explicit
    # use of bias terms.
    model.add(tf.keras.layers.BatchNormalization())

    # LeakyReLU is a variant of the ReLU activation function. It allows
    # a small gradient (slope) when the unit is inactive, which is
    # defined by values less than zero. This non-zero slope for negative
    # values helps to maintain gradient flow during backpropagation,
    # which can prevent the vanishing gradient problem often encountered
    # with ReLU activations. It is applied after the weights and
    # batch normalization, allowing the network to learn from all data points,
    # including those that would otherwise have no gradient.
    model.add(tf.keras.layers.LeakyReLU())

    # We now reshape the output from the Dense layer into a format
    # that can be worked with by Conv2DTranspose layers.
    # This is essentially an 8x8 grid with 512 channels that we will
    # "upsample" to a larger image. So now we have 512
    # 8x8 grids.
    model.add(tf.keras.layers.Reshape((8, 8, 512)))

    # Conv2DTranspose layers are used to upsample the input. These
    # are sometimes called deconvolutional layers and they
    # perform the opposite of a Conv2D layer. Instead of reducing the
    # dimensions, they increase the dimensions. The parameters
    # define the number of filters, kernel size, and the stride.
    # Padding='same' ensures the output size matches the input size multipled
    # by the stride length (in this case 2). Again, we do not use a bias term
    # because it will be handled by the subsequent batch normalization layer.
    # So simply put we have 512 8x8 grids and to all 512 of these grids we
    # apply 256 kernals/filters which are our weights essentially, we move
    # each kernal across the whole grid for all 512 grids and take the total
    # affect (sum up) of each input node when the kernal is applied to it.
    # All 256 of the kernals create one channel in the output and have their
    # own weights.
    # Upsample to 16x16 grids
    model.add(
        tf.keras.layers.Conv2DTranspose(
            256, (5, 5), strides=(
                2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Further upsampling to 32x32. The number of filters is reduced compared
    # to the previous layer as we get closer to the
    # final image size. This is a common practice in generator design to
    # progressively reduce the number of channels.
    model.add(
        tf.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(
                2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # This layer adds more detail to the image. It does not change the size of
    # the image but refines the features that have been learned.
    # We know this since padding is "same" and strides are of length 1, once
    # again we are just adding detail to the image.
    model.add(
        tf.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(
                1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Final upsampling to 64x64, which is the size of the images we want to
    # generate.
    model.add(
        tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(
                2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())

    # ReLU is used here for the final upsampling. This can be a design
    # choice to experiment with LeakyReLU or ReLU to see
    # which gives better results. I found ReLU gave better results.
    model.add(tf.keras.layers.ReLU())

    # The final Conv2DTranspose layer has 3 filters because we are
    # creating 64x64 RGB images, which require 3 color channels.
    # The 'tanh' activation function is used as it outputs values in a
    # range of [-1, 1], which matches the preprocessing
    # that is typically applied to the images before training a GAN (scaling
    # pixel values to this range).
    model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(
        1, 1), padding='same', use_bias=False, activation='tanh'))

    return model


def build_discriminator(img_shape):
    """
    Constructs the discriminator model with a given input image shape.

    Args:
    img_shape (tuple): Shape of the input images.

    Returns:
    tf.keras.Model: The discriminator model, which takes an image as input
    and outputs a single scalar score indicating the authenticity of the image.
    """

    # Initialize the Sequential model which will hold the layers in sequence.
    model = tf.keras.Sequential()

    # Start with a Conv2D layer which will reduce the dimensionality
    # of the input image. The discriminator's first layer
    # uses a set of learnable filters that will extract features
    # from the input image (e.g., edges, textures).
    # Padding is set to 'same' to keep the output size the same as
    # the input size when stride is (1, 1).
    # this Conv2D layer layer mirrors the last Conv2DTranspose layer of the
    # generator, however now we go from 3 channels
    # with 64x64 grids to 64 channels with 64x64 grids, rather than the other
    # way around.
    model.add(
        tf.keras.layers.Conv2D(
            64, (5, 5), strides=(
                1, 1), padding='same', input_shape=img_shape))

    # LeakyReLU is used for the same reason as in the generator -
    # it allows gradients to flow even when the neuron's
    # output is less than zero, which is crucial for the discriminator to
    # learn to classify images effectively.
    model.add(tf.keras.layers.LeakyReLU())

    # Dropout is added as a form of regularization; it randomly sets a
    # fraction of input units to 0 during training,
    # which helps prevent overfitting. The dropout rate of 0.3 means that
    # each unit has a 30% chance of being excluded
    # from the update during each training iteration.
    model.add(tf.keras.layers.Dropout(0.3))

    # Reverse the generator's architecture
    # From 64x64 to 32x32
    # The discriminator progressively increases the number of filters and
    # reduces the spatial dimensionality of the input volume,
    # effectively focusing on more abstract features and reducing
    # computational requirements.
    # This layer downsamples the image from 64x64 to 32x32.
    model.add(
        tf.keras.layers.Conv2D(
            128, (5, 5), strides=(
                2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    # Refining Conv2D layer similar to generator's equivalent
    # Additional convolutional layers continue to increase the
    # depth while compressing spatial dimensions,
    # forcing the network to retain only the most essential
    # information for the classification task.
    # This layer does not change the size of the volume, it's meant to refine
    # the features extracted.
    model.add(
        tf.keras.layers.Conv2D(
            128, (5, 5), strides=(
                1, 1), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    # Continue to intensify the depth of the network while halving the spatial
    # dimensions. This layer downsamples the image from 32x32 to 16x16.
    model.add(
        tf.keras.layers.Conv2D(
            256, (5, 5), strides=(
                2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    # This layer downsamples the image from 16x16 to 8x8 and continues to add
    # depth to the network. It is the last convolutional layer and it prepares
    # the features for final classification.
    model.add(
        tf.keras.layers.Conv2D(
            512, (5, 5), strides=(
                2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    # Flatten the output of the last convolutional layer to a
    # 1D vector to be fed into the Dense layer.
    # This is necessary because Dense layers expect inputs to
    # be flat vectors, but our convolutional layers
    # output a high-dimensional tensor.
    model.add(tf.keras.layers.Flatten())

    # The final Dense layer outputs a single value. No activation
    # function is used here; thus, it outputs a raw score (logit)
    # without bounding it to a specific range. For binary classification
    # tasks, an activation function like sigmoid would be used
    # to constrain the output between 0 and 1, representing the probability
    # of a class. However, in GANs, the discriminator's
    # output is used as a real-valued score that indicates the discriminator's
    # assessment of the authenticity of the input image
    # (with higher values more indicative of "real" images, and lower values
    # more indicative of "fake" ones), which is then used
    # in the computation of the loss during training.
    model.add(tf.keras.layers.Dense(1))

    return model


# Binary cross-entropy loss function is used for the discriminator and
# generator. Since the final layer of the discriminator does not use a
# sigmoid activation, 'from_logits=True' is set
# which means the function will internally apply the sigmoid before
# calculating the loss.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    """
    Calculates the discriminator's loss given the discriminator's predictions
    on real and generated images.

    Args:
    real_output (tf.Tensor): Discriminator output on real images.
    fake_output (tf.Tensor): Discriminator output on fake (generated) images.

    Returns:
    tf.Tensor: The total loss for the discriminator.
    """

    # Discriminator loss consists of two parts:
    # 1. How well the discriminator classifies real images as real.
    # 2. How well the discriminator classifies fake images as fake.
    # The discriminator's goal is to output 1 for real images and 0 for fake
    # ones.

    # Calculate the loss for real images:
    # The real_loss measures how well the discriminator classifies real images
    # as real. We compare the discriminator's output for real images
    # (real_output) with a tensor of ones, where each element is 1.0,
    # indicating the desired output (real) for each real image.
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    # Calculate the loss for fake images:
    # The fake_loss measures how well the discriminator classifies fake images
    # as fake. We compare the discriminator's output for fake images
    # (fake_output) with a tensor of zeros, where each element is 0.0,
    # indicating the desired output (fake) for each fake image.
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    # The total loss is the sum of the real and fake losses.
    total_loss = real_loss + fake_loss

    return total_loss


def generator_loss(fake_output):
    """
    Calculates the generator's loss given the discriminator's predictions on
    generated images.

    Args:
    fake_output (tf.Tensor): Discriminator output on fake (generated) images.

    Returns:
    tf.Tensor: The loss for the generator.
    """

    # Generator's loss depends on how well it tricks the discriminator.
    # Ideally, the generator wants the discriminator to classify all the fake
    # images as real (output as 1).
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Optimizers for the generator and discriminator.
# Adam optimizer is used for its adaptive learning rate capabilities,
# which makes it well-suited for GANs.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# The train_step function will perform one step of training on a batch of
# images.


# This decorator compiles the function into a callable TensorFlow graph,
# which can lead to performance gains.
@tf.function
def train_step(images, generator, discriminator, batch_size, z_dim):
    """
    Performs one step of training on a batch of images.

    Args:
        images (tf.Tensor): A batch of real images.
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
        batch_size (int): The batch size.
        z_dim (int): The dimensionality of the latent noise vector.
    """

    # Random noise is generated for each image in the batch, which will be
    # used as input for the generator.
    noise = tf.random.normal([batch_size, z_dim])

    # TensorFlow's GradientTape is a context manager that
    # records operations for automatic differentiation.
    # We use two tapes to watch the respective operations
    # for the generator and discriminator separately
    # because we want to calculate the gradients of each
    # model's loss with respect to its own variables.
    # If we used one tape, we would not be able to
    # calculate these two sets of
    # gradients independently.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Generator produces images from random noise.
        # We are recording the operations inside this context
        # so that TensorFlow can later compute the gradients of the generator's
        # loss with respect to its weights.
        generated_images = generator(noise, training=True)

        # The discriminator evaluates both
        # real images and the generated images.
        # We record these operations on both tapes since both affect the
        # generator and discriminator losses.
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Compute the generator and discriminator losses.
        # These computations are also being watched by the respective tapes, so
        # that the gradients can be computed.
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Once we exit the 'with' block, no further operations are recorded, and
    # we can now compute the gradients.
    # We compute the gradient of the generator's loss with respect to the
    # generator's variables (weights and biases).
    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)

    # Similarly, we compute the gradient of the discriminator's loss with
    # respect to the discriminator's variables.
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    # Now that we have the gradients, we can use them
    # to perform a step of gradient descent.
    # The generator's optimizer uses the generator's
    # gradients to update its variables,
    # moving them in a direction that will reduce the generator's loss.
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))

    # Similarly, the discriminator's optimizer uses the
    # discriminator's gradients to update its variables,
    # moving them in a direction that will reduce the discriminator's loss.
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


# Initialize the generator model with a specified latent dimension (z_dim).
generator = build_generator(z_dim=100)

# Initialize the discriminator model with the shape of the images it will
# classify.
discriminator = build_discriminator(img_shape=(64, 64, 3))


# Function to preprocess the training images.
def preprocess_image(image):
    """
    Preprocesses an image by normalizing its pixel values to the range [-1, 1].

    Args:
    image (tf.Tensor): The input image.

    Returns:
    tf.Tensor: The normalized image.
    """

    # Normalize the image pixels to the range [-1, 1], as the
    # generator's output layer uses tanh activation
    # which also produces outputs in this range. This normalization is crucial
    # for the model to learn correctly.
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image

# Function to load and preprocess the dataset from a given directory.


def load_dataset(directory, batch_size):
    """
    Loads and preprocesses the dataset from a directory.

    Args:
    directory (str): Path to the dataset directory.
    batch_size (int): Size of the batches to load.

    Returns:
    DirectoryIterator: An iterator over the dataset with the specified batch
    size.
    """

    # Use the ImageDataGenerator to augment and preprocess images on-the-fly,
    # which is efficient on memory.
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_image)

    # Create a data generator that will fetch batches of images from
    # the directory, applying the defined preprocessing.
    # The images are resized to the target_size (64x64) and loaded in
    # color_mode 'rgb'.
    return datagen.flow_from_directory(
        directory,
        target_size=(64, 64),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode=None)


# Load the dataset of cat images located at the specified path, with a
# batch size of 32.
dataset = load_dataset('./cats', batch_size=32)


# Function to save generated images for visualization
def generate_and_save_images(
        model,
        epoch,
        test_input,
        folder='training_images'):
    """
    Generates images using the generator model and saves them to disk.

    Args:
    model (tf.keras.Model): The generator model.
    epoch (int): The current epoch, used for naming the output file.
    test_input (tf.Tensor): A batch of noise vectors to generate images from.
    folder (str, optional): The directory to save the images. Defaults to
    'training_images'.
    """

    # Generate images from the noise vector (test_input) using the generator
    # model.
    predictions = model(test_input, training=False)
    # Plot the results in a 4x4 grid.
    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # Rescale the pixel values from [-1, 1] (due to the tanh activation)
        # back to [0, 1]
        img = (predictions[i, :, :, :] + 1) / 2.0
        plt.imshow(img)
        plt.axis('off')  # Hide the axis to emphasize the images.

    # If the output folder doesn't exist, create it.
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Save the full grid image to the specified folder, naming it according to
    # the current epoch.
    plt.savefig(os.path.join(folder, f'image_at_epoch_{epoch:04d}.png'))
    # Close the plotting window to free up memory.
    plt.close()


# Directory and file prefix for saving model checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# Checkpoint object creation to save the state of the generator,
# discriminator, and their optimizers.
# This allows us to resume training from the last saved state or use the
# trained models for inference later.
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)


def train(
        dataset,
        epochs,
        generator,
        discriminator,
        batch_size,
        z_dim,
        total_images):
    """
    Trains the GAN models.

    Args:
    dataset (tf.data.Dataset): The dataset to train on.
    epochs (int): Number of epochs to train for.
    generator (tf.keras.Model): The generator model.
    discriminator (tf.keras.Model): The discriminator model.
    batch_size (int): The batch size for training.
    z_dim (int): The dimensionality of the latent space.
    total_images (int): The total number of images in the dataset.
    """

    noise_dim = z_dim  # Dimensionality of the noise vector for the generator.

    # Number of examples to generate for visualization.
    num_examples_to_generate = 16

    # Generate a fixed seed noise for visualizing progress in the output
    # generation.
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Calculate the number of steps per epoch.
    steps_per_epoch = total_images // batch_size

    # Loop over the dataset for a defined number of epochs.
    for epoch in range(epochs):

        start_time = time.time()  # Record the start time of the epoch.

        i = 0  # Initialize a counter to keep track of batches processed.

        # Iterate over batches of images in the dataset.
        for image_batch in dataset:
            i += 1

            # Perform a training step with the current batch of images.
            train_step(
                image_batch,
                generator,
                discriminator,
                batch_size,
                z_dim)

            # Every 200 batches, print the time taken to process those
            # batches. Just to know how the training is going.
            if i % 200 == 0:
                run_time = time.time() - start_time
                minutes, seconds = divmod(run_time, 60)
                print(f'{i} {int(minutes)}m {int(seconds)}s')
                start_time = time.time()  # Reset the start time.

            # Stop the epoch once all batches have been processed. I had to
            # add this in as a work around to avoid infinite loops.
            if i >= steps_per_epoch - 1:
                break
        i = 0

        # After an epoch, calculate and print the time taken for the entire
        # epoch.
        epoch_time = time.time() - start_time
        minutes, seconds = divmod(epoch_time, 60)

        # Generate and save images at the end of the epoch for visualization.
        print(f'Epoch {epoch + 1}: {int(minutes)}m {int(seconds)}s')

        # Generate and save images at the end of the epoch for visualization.
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every epoch. I know it's not the most efficient but
        # my computer had crashed after 9
        # epochs so I didn't want to risk wasted computation.
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save the current state of the models and optimizers.
        checkpoint.save(file_prefix=checkpoint_prefix)


# Restore the latest checkpoint before continuing training or for
# inference. Comment this out if no check points present
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Update the learning rates for both generator and discriminator
# optimizers if you want to, comment out if not.
new_learning_rate = 1.5e-4  # Increased learning rate

# Update the learning rate on the optimizers
generator_optimizer.learning_rate.assign(new_learning_rate)
discriminator_optimizer.learning_rate.assign(new_learning_rate)

# Start or continue training with the updated learning rate, dataset, and
# model parameters.
train(
    dataset,
    epochs=300,
    generator=generator,
    discriminator=discriminator,
    batch_size=32,
    z_dim=100,
    total_images=31494)

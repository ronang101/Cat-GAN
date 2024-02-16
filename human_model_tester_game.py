# Import the Tkinter library for creating GUI applications
import tkinter as tk

# Import TensorFlow library for deep learning tasks
import tensorflow as tf

# Import the NumPy library for numerical operations and array manipulation
import numpy as np

# Import the random module for generating random numbers and sequences
import random

# Import the OS module for interacting with the operating system, such as
# file operations
import os

# Import the PIL (Python Imaging Library) module for image processing tasks
from PIL import Image, ImageTk


# Load the trained generator model
saved_model_path = r'.\models'
generator = tf.keras.models.load_model(saved_model_path)

# Path to the real cat images
real_images_path = r'.\cats\cats'

# Initialize score
score = {'right': 0, 'wrong': 0}


def load_real_image():
    """
    Selects and loads a random real image from the specified directory.

    Returns:
    PIL.Image: A real image resized to 64x64 pixels.
    """
    # Randomly select a real image
    # Get a random filename from the directory
    image_name = random.choice(os.listdir(real_images_path))
    # Construct the full path to the image
    image_path = os.path.join(real_images_path, image_name)
    # Open the image using PIL, then resize it to match the GAN's output
    # dimensions.
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize to match your GAN output size
    return img


def generate_image():
    """
    Generates a fake image using the trained generator model.

    Returns:
    PIL.Image: A generated image converted from a tensor to a PIL Image and
    resized to 64x64 pixels.
    """
    # Generate random noise as input
    noise = tf.random.normal([1, 100])
    # Use the generator model to generate an image from the noise vector.
    generated_image = generator(noise, training=False)

    # Post-process the generated image for display:
    # 1. Normalize the pixel values to [0, 1] by bringing the tanh
    # activation output from [-1, 1] to [0, 1].
    # 2. Clip values to ensure they are within [0, 1] after any
    # potential numerical instability during processing.
    # 3. Convert to a NumPy array, scale to [0, 255], and cast to uint8.
    # 4. Convert the NumPy array to a PIL Image object for easy display
    # and manipulation in the Tkinter GUI.

    # Step 1: Normalize the pixel values to [0, 1] by bringing
    # the tanh activation output from [-1, 1] to [0, 1].
    # The output of the generator is typically in the range
    # [-1, 1] due to the tanh activation function.
    # We shift and scale it to the range [0, 1] by adding 1
    # (to shift) and dividing by 2 (to scale).
    # [0, :, :, :] get's the first generated image from a batch.
    img = (generated_image[0, :, :, :] + 1) / 2.0

    # Step 2: Clip values to ensure they are within [0, 1] after
    # any potential numerical instability during processing.
    # Sometimes, due to numerical instability or model errors,
    # values may exceed the desired range [0, 1].
    # We use np.clip() to ensure that all values in the array
    # are within the specified range [0, 1].
    img = np.clip(img, 0, 1)

    # Step 3: Convert to a NumPy array, scale to [0, 255], and cast to uint8.
    # We scale the pixel values from [0, 1] to [0, 255] to match the
    # expected range for image data.
    # Then, we cast the array to uint8 (unsigned 8-bit integer),
    # which is the data type expected by PIL for image data.

    # Step 4: Convert the NumPy array to a PIL Image object for easy
    # display and manipulation in the Tkinter GUI.
    # Finally, we convert the NumPy array representing the image to a
    # PIL Image object,
    # which is compatible with Tkinter for display within a GUI.
    img = Image.fromarray((img * 255).astype(np.uint8))
    return img


def update_score(is_correct):
    """
    Updates the player's score based on whether their guess was correct or not
    and updates the score display.

    Args:
    is_correct (bool): True if the player's guess was correct, False otherwise.
    """
    # Update the score dictionary based on the correctness of the player's
    # guess.
    if is_correct:
        score['right'] += 1
    else:
        score['wrong'] += 1

    # Update the text of the score_label widget to reflect the current score.
    # This uses the global score_label variable, which should be a Tkinter
    # Label widget.
    score_label.config(
        text=f"Score: Right - {score['right']}, Wrong - {score['wrong']}")


def show_image():
    """
    Displays a randomly selected real or generated (fake) image on the GUI and
    stores its authenticity.
    """

    # Randomly choose whether to display a real or generated image.
    if random.choice([True, False]):
        # Generate a fake image using the generator model.
        img = generate_image()
        is_real = False  # Mark the image as generated (fake).
    else:
        img = load_real_image()  # Load a real image from the dataset.
        is_real = True  # Mark the image as real.

    # Convert the PIL Image to a format that tkinter can use (PhotoImage)
    # and display it.
    img = ImageTk.PhotoImage(img)
    # Update the panel widget to display the new image.
    panel.config(image=img)
    panel.image = img  # Keep a reference to avoid garbage collection
    panel.is_real = is_real  # Store whether the image is real or not


def guess_real():
    """
    Handles the player's guess that the currently displayed image is real.
    """
    # Update the score based on whether the image was actually real.
    update_score(
        panel.is_real)


def guess_fake():
    """
    Handles the player's guess that the currently displayed image is fake
    (generated).
    """
    # Update the score based on whether the image was actually fake.
    update_score(
        not panel.is_real)


# Set up the tkinter GUI
root = tk.Tk()
root.title("Real or GAN Cat Image Game")  # Set the window title.

# Create a panel widget to display images. This will be updated each time a
# new image is generated or loaded.
# The Label widget is used here as a simple way to display images.
panel = tk.Label(root)
panel.pack()  # Add the panel to the window.

# Create buttons for the user to interact with the game.
# Button to guess the image is real.
real_button = tk.Button(root, text="Real", command=guess_real)
# Button to guess the image is fake.
fake_button = tk.Button(root, text="Fake", command=guess_fake)
# Button to generate/load a new image.
generate_button = tk.Button(root, text="Generate Image", command=show_image)

# Add the buttons to the window.
real_button.pack()
fake_button.pack()
generate_button.pack()

# Create and display a label to show the player's current score.
score_label = tk.Label(root, text="Score: Right - 0, Wrong - 0")
score_label.pack()

# Start the GUI event loop
root.mainloop()

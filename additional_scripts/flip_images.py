from PIL import Image  # PIL is Python Imaging Library for image processing
import os

# When training GANs, data augmentation techniques like flipping images
# horizontally can help improve the robustness and diversity of the
# generated images.

# Define the path to your image directory
image_dir = r'..\cats\cats'  # Path to the directory containing input images
# Path to the directory where flipped images will be saved
output_dir = r'..\cats\cats'

# Make sure the output directory exists
# Creates the output directory if it doesn't already exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all files in the image directory
# Iterates over all files in the image directory
for filename in os.listdir(image_dir):
    # Check if the file ends with '.jpg' or '.png'
    # (add or remove file types as needed)
    # Add or remove file types as needed
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Construct the full path to the image file
        image_path = os.path.join(image_dir, filename)
        # Open the image
        img = Image.open(image_path)  # Opens and loads the image into memory
        # Flip the image horizontally
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Save the flipped image to the output directory with a new name
        # Construct the full path to save the flipped image
        # New filename with 'flipped_' prefix
        flipped_img.save(os.path.join(output_dir, f'flipped_{filename}'))

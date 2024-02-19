import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle



def simulate_transposed_conv_influence(input_size, kernel_size,
                                       stride, input_indices, padding="same"):
    """
    Simulate the influence of specific input pixels on the output of a
    transposed convolution.

    Args:
    - input_size (int): The size of one side of the square input
    (assuming a square input).
    - kernel_size (int): The size of one side of the square kernel.
    - stride (int): The stride with which the kernel is applied.
    - input_indices (list of tuples): The indices of input pixels whose
    influence we want to track.
    - padding (str): This string indicates if we should apply the padding
    as "same" or "valid".

    Returns:
    - influence_map (np.array): A 2D array representing the influence of the
    specified input pixels on the output.
    """

    # Determine the size of the output based on the input size and stride.
    # The output size increases with the stride, effectively reversing the
    # effect of a convolution.
    output_size = (input_size - 1) * stride + kernel_size

    # Initialize a matrix to track the influence of specified input
    # pixels across the entire output.
    # Initially, no influence is assumed, hence a matrix of zeros.
    influence_map = np.zeros((output_size, output_size))

    # Iterate over each input index to simulate its influence on the output
    # through the transposed convolution.
    for i, j in input_indices:

        # Determine where the influence of the
        # current input pixel starts in the output.
        # This is calculated based on the stride
        # and the position of the input pixel.
        i_start = (i - 1) * stride
        j_start = (j - 1) * stride

        # For each position that the kernel covers
        # in the output, mark its influence.
        # This loop simulates the kernel sliding over the output space.
        for k in range(kernel_size):
            for l in range(kernel_size):
                i_pos = i_start + k
                j_pos = j_start + l
                influence_map[i_pos, j_pos] = 1  # Mark influence.

    # This conditional block checks for the padding type and adjusts the
    # output accordingly.
    if padding == "same":
        # For 'same' padding in transposed convolutions, we
        # aim to have an output size that is the same as
        # the input size multiplied by the stride, as if
        # reversing a 'same' padding forward convolution.
        # However, due to the way transposed convolution works,
        # the initial output (without cropping) can be larger.
        # This is because transposed convolution applies the
        # kernel to an upsampled version of the input,
        # which can lead to additional border pixels beyond
        # the desired output size.

        # To simulate 'same' padding, we calculate the target crop size,
        # which is the input size scaled by the stride.
        crop_size = input_size * stride

        # We then determine the starting point for cropping by
        # calculating the offset from the edge of the
        # initial larger output to the beginning of the
        # target output size.
        start = (output_size - crop_size) // 2

        # The ending point for cropping is simply the starting point
        # plus the size of the target output.
        end = start + crop_size

        # The influence_map is then cropped to this central region,
        # effectively simulating 'same' padding
        # by only considering the central part of the output that
        # corresponds to the input dimensions scaled by the stride.
        # This is how padding "same" is calculated in keras as we
        # can see at the end of the code when we check our
        # example function against the keras output.
        cropped_influence_map = influence_map[start:end, start:end]

        # Return the cropped influence map which now simulates 'same' padding.
        return cropped_influence_map

    elif padding == 'valid':

        # For 'valid' padding, no cropping is necessary because the
        # output size is directly determined by the input size,
        # stride, and kernel size without any adjustments.
        return influence_map

    else:

        # If the padding parameter is not recognized, we return an error
        # message prompting the user to enter a valid option.
        return "Enter Valid Padding Parameter"


# Example usage: Simulate and print the influence map for specific input
# pixels.
influence_map = simulate_transposed_conv_influence(
    input_size=8, kernel_size=5, stride=2, input_indices=[(2, 6), (8, 1)],
    padding="valid")

print(influence_map)


def transposed_conv_output(input_size, kernel_size, stride, padding='same'):
    """
    Calculate the output of a transposed convolution given the input size,
    kernel size, stride, and padding.

    Args:
    - input_size (int): The size of one side of the square input (assuming a
      square input).
    - kernel_size (int): The size of one side of the square kernel.
    - stride (int): The stride with which the kernel is applied.
    - padding (str): The type of padding applied ('same' or 'valid').

    Returns:
    - output_size (int): The size of one side of the square output.
    """

    # Calculate the output size of a square input after a transposed
    # convolution operation.
    if padding == 'same':

        # 'Same' padding aims to produce an output of size that
        # matches the input when stride is 1.
        # With transposed convolution, it scales the input size by the stride.
        # For 'same' padding, output size is input size * stride.
        output_size = input_size * stride

    elif padding == 'valid':

        # 'Valid' padding produces an output where the entire kernel
        # fits within the bounds of the input.
        # This calculation adjusts the output size based on the kernel
        # size and stride, without padding.
        # For 'valid' padding, output size is (input_size - 1) * stride +
        # kernel_size.
        output_size = (input_size - 1) * stride + kernel_size
    else:
        raise ValueError("Padding must be 'same' or 'valid'.")

    return output_size


def simulate_detailed_transposed_conv_influence(input_matrix, kernel_matrix,
                                                stride, padding="same"):
    """
    Simulate the influence of transposed convolution on an input matrix
    using a given kernel and stride.

    Parameters:
    - input_matrix: A 2D numpy array representing the input matrix.
    - kernel_matrix: A 2D numpy array representing the convolutional kernel.
    - stride: An integer representing the stride of the convolution.
    - padding (str): This string indicates if we should apply the padding
    as "same" or "valid".

    Returns:
    - influence_map: A 2D numpy array representing the output matrix after
    applying transposed convolution.
    """

    # Determine the dimensions of the input and kernel matrices.
    # Size of one dimension of the input matrix.
    input_size = input_matrix.shape[0]

    # Size of one dimension of the kernel matrix.
    kernel_size = kernel_matrix.shape[0]

    # Calculate the size of the output matrix based on the input size,
    # kernel size, and stride.
    output_size = (input_size - 1) * stride + kernel_size

    # Initialize an output matrix filled with zeros, with dimensions
    # determined by the calculated output size.
    influence_map = np.zeros((output_size, output_size))

    # Iterate over each element in the input matrix to apply the
    # kernel and stride.
    for i in range(input_size):
        for j in range(input_size):

            # Update the corresponding region in the influence map by
            # adding the element-wise product of the kernel
            # and the current element from the input matrix, scaled
            # according to the stride and position.s
            influence_map[i * stride:i * stride + kernel_size, j * stride:j *
                          stride + kernel_size] += (
                              kernel_matrix * input_matrix[i, j])

    # This conditional block checks for the padding type and
    # adjusts the output accordingly.
    if padding == "same":

        # For 'same' padding in transposed convolutions, we aim to
        # have an output size that is the same as
        # the input size multiplied by the stride, as if reversing a
        # 'same' padding forward convolution.
        # However, due to the way transposed convolution works, the
        # initial output (without cropping) can be larger.
        # This is because transposed convolution applies the
        # kernel to an upsampled version of the input,
        # which can lead to additional border pixels
        # beyond the desired output size.

        # To simulate 'same' padding, we calculate the target
        # crop size, which is the input size scaled by the stride.
        crop_size = input_size * stride

        # We then determine the starting point for
        # cropping by calculating the offset from the edge of the
        # initial larger output to the beginning of the target output size.
        start = (output_size - crop_size) // 2

        # The ending point for cropping is simply the starting point
        # plus the size of the target output.
        end = start + crop_size

        # The influence_map is then cropped to this central region,
        # effectively simulating 'same' padding
        # by only considering the central part of the output that
        # corresponds to the input dimensions scaled by the stride.
        # This is how padding "same" is calculated in keras as
        # we can see at the end of the code when we check our
        # example function against the keras output.
        cropped_influence_map = influence_map[start:end, start:end]

        # Return the cropped influence map which now simulates 'same' padding.
        return cropped_influence_map
    elif padding == 'valid':
        # For 'valid' padding, no cropping is necessary because the
        # output size is directly determined by the input size,
        # stride, and kernel size without any adjustments.
        return influence_map
    else:
        # If the padding parameter is not recognized, we return an error
        # message prompting the user to enter a valid option.
        return "Enter Valid Padding Parameter"


# Create a figure and axes for plotting
fig, ax = plt.subplots(figsize=(10, 8))


def animate(frame_and_highlights):
    """
    This function is called repeatedly by the animation framework to visualize
    each frame of the transposed convolution simulation.
    It updates the plot with the current state of the matrix as it undergoes
    the simulated transposed convolution process.

    Parameters:
    - frame_and_highlights: A tuple containing two 2D numpy arrays.
    The first array represents the matrix at a
    specific step in the simulation process. Each call to this
    function visualizes the matrix at its current state by updating the plot
    accordingly.

    The second array represents a mask of the matrix where we see which cells
    are being affected in order to apply highlighting in the animation
    """

    # Unpack the tuple into the two numpy arrays.
    frame, highlight = frame_and_highlights

    # Clear the axes for the new frame.
    # This is necessary because we want to redraw the plot from
    # scratch on each animation frame, ensuring that the previous
    # frame's content is erased and replaced with the new frame's content.
    ax.clear()

    # Add a title to the plot. This will appear on every frame of the animation.
    ax.set_title("Transposed Convolution Output Simulation")

    # Set up grid lines to visually delineate individual cells in the matrix.
    # The grid lines are positioned at half-integer locations to
    # line up with the edges of matrix cells, not the center.
    # This makes it easier to see the boundaries between individual cells in
    # the matrix.
    ax.set_xticks(np.arange(-0.5, frame.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, frame.shape[0], 1), minor=True)

    # Draw the grid with specified styling.
    # The grid visually represents the "walls" or boundaries of
    # each cell in the matrix, using black lines with a specific
    # thickness. This styling choice helps to clearly define each cell's
    # space on the plot.
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

    # Hide the minor tick marks that were automatically
    # created by setting the minor ticks for the grid.
    # This is purely a visual preference to keep the focus
    # on the grid and the values within each cell, without extra
    # visual clutter from the tick marks.
    ax.tick_params(which="minor", size=0)

    # Set the limits of the axes to tightly enclose the grid.
    # This adjustment ensures that the plot area is exactly large
    # enough to contain the grid representing the matrix,
    # without unnecessary whitespace around the edges.
    ax.set_xlim(-0.5, frame.shape[1]-0.5)
    ax.set_ylim(-0.5, frame.shape[0]-0.5)

    # Remove the major tick labels (the default labels on the axes).
    # Since our focus is on the content of the matrix itself and
    # not the position within the plot space, the axis labels
    # are hidden to maintain a clean and focused visualization.
    ax.set_xticks([])
    ax.set_yticks([])

    # Annotate each cell with its numerical value.
    # This loop goes through each element (cell) in the frame (matrix)
    # and places a text annotation at that cell's location.
    # The text is centered in the cell and displays the integer value of
    # the cell, providing a clear visual representation
    # of the matrix's contents at this step in the simulation.
    for (j, i), val in np.ndenumerate(frame):
        ax.text(i, j, int(val), ha='center', va='center', color='black')

    # Highlight each cell which has been affected in this frame
    # This loop goes through each element (cell) in the highlight (matrix)
    # and colours the cell at the cell's location.
    for (j, i), val in np.ndenumerate(highlight):
        # Only highlight cells updated in the mask.
        if val:
            ax.add_patch(Rectangle((i-0.5, j-0.5), 1, 1, color='lightblue'))

    # Invert the y-axis to match the typical matrix representation.
    # Matrices are typically displayed with the (0,0) entry in the top-left
    # corner, but the default plotting behavior
    # places the (0,0) point at the bottom-left. This inversion ensures that
    # the matrix is visually represented in a way
    # that matches common expectations.
    ax.invert_yaxis()


def create_animation(input_matrix, kernel_matrix, stride, padding="same"):
    """
    Creates an animation illustrating the process of applying a transposed
    convolution operation to an input matrix.

    The function simulates the transposed convolution step-by-step. It applies
    a kernel to the input matrix, progressively revealing the influence of each
    cell in the input matrix on the output matrix. The resulting animation is
    saved as a GIF file.

    Please note a square matrix is assumed for the input and kernal.

    Parameters:
    - input_matrix (np.array): A 2D numpy array representing the input matrix
    to the transposed convolution operation.
    - kernel_matrix (np.array): A 2D numpy array representing the kernel used
    in the transposed convolution operation.
    - stride (int): The stride (step size) used in the transposed convolution
    operation.

    Returns:
    - gif_path (str): The file path to the saved GIF animation.

    The animation is created using matplotlib's FuncAnimation class. Each
    frame of the animation represents a step in the simulation where the
    kernel is applied to a part of the input matrix defined by a mask.
    The mask grows with each frame, simulating the transposed convolution
    process. The animation properties, such as the interval between frames
    and whether to use blitting, are set within the FuncAnimation call.

    The resulting GIF file provides a visual representation of how the
    transposed convolution operation affects the input matrix to produce
    the output matrix, making it a useful educational tool for understanding
    this operation in the context of neural networks and deep learning.
    """

    # Create the animation frames for each step of the simulation.
    frames = []

    # Create the highlight masks for each step of the simulation.
    highlights = []

    # Variable for knowing the dimensions of the matrix.
    size = input_matrix.shape[0]

    # Used for updating the input mask.
    current_cell = 0
    # Iterate over each row and column to create highlight coordinates.
    for row in range(1, size + 1):
        for col in range(1, size + 1):
            highlight_coords = (row, col)
            # Create a mask to simulate the step-by-step application
            # of the kernel over the input matrix.
            input_mask = (np.arange(input_matrix.size).reshape(
                input_matrix.shape) <= current_cell)
            current_cell += 1
            # Apply the transposed convolution simulation function to the
            # masked input matrix.
            step_matrix = simulate_detailed_transposed_conv_influence(
                input_matrix * input_mask, kernel_matrix, stride, padding)
            # Append the result to the list of frames for animation.
            highlight = simulate_transposed_conv_influence(
                input_matrix.shape[0], kernel_matrix.shape[0], stride, [
                    highlight_coords],
                padding)
            highlights.append(highlight)
            frames.append(step_matrix)

    # Create an animation using the frames and the animate function.
    # 'fig' is the object on which the animation will be drawn.
    # 'animate' is the function called the draw each frame.
    # 'frames' specify the data used for each frame of the animation.
    # 'blit = False' stops blitting which is a technique to only update the
    # parts
    # of the fram which updated, but this is a simple animation so is not
    # needed here.
    # 'interval' sets the delay between frames in ms.
    anim = FuncAnimation(fig, animate, frames=zip(frames, highlights),
                         blit=False, interval=500,
                         save_count=input_matrix.size)

    # Save the animation to a GIF file
    gif_path = './transposed_convolution.gif'
    anim.save(gif_path, writer=PillowWriter(fps=2))

    # Close the plot to prevent it from displaying inline if running in a
    # notebook.
    plt.close()

    # Return the path of the saved GIF file.
    gif_path


# Define input matrix and kernel matrix for demonstration.
input_matrix = np.array([
    [14,  75, 249, 185, 214,  40, 174, 249],
    [42,   7, 186, 218,  27, 230, 235, 101],
    [93, 172,  21, 116,  61, 174, 191,  56],
    [143,  30, 219, 177,   7,  35, 196, 194],
    [116, 204, 194, 170, 158, 173,  55, 185],
    [35, 237, 122,   2,   3,  32, 158, 196],
    [41, 107,  63, 122, 113, 205, 185, 118],
    [43,  30, 219, 216,  94,  59,  45, 204]
])

kernel_matrix = np.array([
    [2,  1,  -3,  4,  9],
    [-1,  1, -1,  -4,  1],
    [2,  -3, -1,  0, -1],
    [1,  0,  4,  1,  1],
    [0, -1,  4, -1, -1]
])

stride = 2


create_animation(input_matrix, kernel_matrix, stride)


def run_simulation_and_keras_model():
    """
    This function runs a simulation of a transposed
    convolution operation using a custom
    implementation and compares it with the result of a
    Keras model using the Conv2DTranspose layer.

    The input and kernel matrices are predefined within the
    function. The simulation and
    model predictions are printed for you to see that the
    outputs of the functions defined above
    are the same as the outputs from keras.
    """
    import tensorflow as tf


    input_matrix = np.array([
        [14,  75, 249, 185, 214,  40, 174, 249],
        [42,   7, 186, 218,  27, 230, 235, 101],
        [93, 172,  21, 116,  61, 174, 191,  56],
        [143,  30, 219, 177,   7,  35, 196, 194],
        [116, 204, 194, 170, 158, 173,  55, 185],
        [35, 237, 122,   2,   3,  32, 158, 196],
        [41, 107,  63, 122, 113, 205, 185, 118],
        [43,  30, 219, 216,  94,  59,  45, 204]
    ])

    kernel_matrix = np.array([
        [0,  0,  1,  0,  0],
        [-1,  1, -1,  0,  0],
        [0,  0, -1,  0, -1],
        [1,  0,  0,  1,  1],
        [0, -1,  0, -1, -1]
    ])

    stride = 2

    print("My Conv2D Transpose: \n {}".format(
        simulate_detailed_transposed_conv_influence(
            input_matrix, kernel_matrix, stride, padding="same")))
    print("\n")

    # Define your input matrix and kernel as numpy arrays.
    # Make sure to reshape them to fit the expected dimensions for Keras.
    # input_matrix should have dimensions (1, input_height, input_width, 1)
    # (batch_size, height, width, channels).
    # kernel_matrix should have dimensions (kernel_height, kernel_width,
    # input_channels, output_channels).
    input_matrix = input_matrix.reshape(
        1, input_matrix.shape[0], input_matrix.shape[1], 1)
    kernel_matrix = kernel_matrix.reshape(
        kernel_matrix.shape[0], kernel_matrix.shape[1], 1, 1)

    # Create a Keras model with a Conv2DTranspose layer.
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=kernel_matrix.shape[0:2],
            strides=(stride, stride),
            padding='same',
            # Accept any height and width but expect only one channel.
            input_shape=(None, None, 1)
        )
    ])

    # Set the weights of the Conv2DTranspose layer to your kernel matrix.
    # The np.array[0] sets the bias to 0.
    model.layers[0].set_weights([kernel_matrix, np.array([0])])

    # Predict the output with the input matrix.
    output = model.predict(input_matrix)

    # The output will have the shape (1, new_height, new_width, 1).
    output = output.reshape(output.shape[1], output.shape[2])

    print(output)

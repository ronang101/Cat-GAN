import numpy as np


def simulate_transposed_conv_influence(input_size, kernel_size,
                                       stride, input_indices):
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

    Returns:
    - influence_map (np.array): A 2D array representing the influence of the
    specified input pixels on the output.
    """

    # Determine the size of the output based on the input size and stride.
    # The output size increases with the stride, effectively reversing the
    # effect of a convolution.
    output_size = input_size * stride

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
                # Check to ensure the kernel's position is within the bounds
                # of the output.
                if 0 <= i_pos < output_size and 0 <= j_pos < output_size:
                    influence_map[i_pos, j_pos] = 1  # Mark influence.

    return influence_map


# Example usage: Simulate and print the influence map for specific input
# pixels.
influence_map = simulate_transposed_conv_influence(
    input_size=8, kernel_size=5, stride=2, input_indices=[(1, 1), (1, 8)])

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
        # For 'same' padding, output size is input size * stride
        output_size = input_size * stride
    elif padding == 'valid':

        # 'Valid' padding produces an output where the entire kernel
        # fits within the bounds of the input.
        # This calculation adjusts the output size based on the kernel
        # size and stride, without padding.
        # For 'valid' padding, output size is (input_size - 1) * stride +
        # kernel_size
        output_size = (input_size - 1) * stride + kernel_size
    else:
        raise ValueError("Padding must be 'same' or 'valid'.")

    return output_size


# Demonstrate how output sizes change with different kernel sizes under 'same'
# padding.
input_size = 8
kernel_sizes = [3, 4, 5, 7, 20]  # Different kernel sizes
stride = 2

# Compute and display output sizes for various kernel sizes.
output_sizes = {k: transposed_conv_output(
    input_size, k, stride, padding='same') for k in kernel_sizes}
output_sizes

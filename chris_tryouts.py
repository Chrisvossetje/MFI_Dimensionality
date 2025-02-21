import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from matplotlib.image import imread

import numpy as np
import matplotlib.pyplot as plt


def gaussian_2d(shape, center=None, sigma=1.0):
    """
    Generates a 2D Gaussian-distributed matrix.
    
    Parameters:
    - shape: tuple (rows, cols), size of the matrix
    - center: tuple (x, y), center of the Gaussian (default: center of the matrix)
    - sigma: float, standard deviation of the Gaussian
    
    Returns:
    - 2D NumPy array with Gaussian-distributed values
    """
    rows, cols = shape
    x = np.linspace(0, cols - 1, cols)
    y = np.linspace(0, rows - 1, rows)
    X, Y = np.meshgrid(x, y)
    
    if center is None:
        center = (cols // 2, rows // 2)
    
    cx, cy = center
    gaussian = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    
    return gaussian

def custom_2d_shape(size=(50, 50), center=None, sx=10, sy=5, theta=0, p=2, offset=0):
    """
    Generates a 2D shape similar to a Gaussian but with more control.

    Parameters:
    - size: tuple (height, width) -> Size of the output matrix.
    - center: tuple (cx, cy) -> Center of the shape. Default: middle of the matrix.
    - sx, sy: float -> Spread in X and Y directions (controls elongation).
    - theta: float (radians) -> Rotation angle of the shape.
    - p: float -> Power exponent (controls sharpness; p=2 is Gaussian).
    - offset: float -> Base offset to shift all values.

    Returns:
    - 2D NumPy array with the generated shape.
    """
    height, width = size
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)

    if center is None:
        cx, cy = width // 2, height // 2
    else:
        cx, cy = center

    # Rotation transformation
    Xr = (X - cx) * np.cos(theta) + (Y - cy) * np.sin(theta)
    Yr = -(X - cx) * np.sin(theta) + (Y - cy) * np.cos(theta)

    # Shape function
    shape = np.exp(-((Xr / sx) ** p + (Yr / sy) ** p)) + offset

    return shape

def gaussian_3d(shape, center=None, sigma=1.0):
    """
    Generates a 3D Gaussian-distributed tensor.

    Parameters:
    - shape: tuple (depth, height, width), size of the 3D matrix
    - center: tuple (cz, cy, cx), center of the Gaussian (default: center of the matrix)
    - sigma: float, standard deviation of the Gaussian

    Returns:
    - 3D NumPy array with Gaussian-distributed values
    """
    width, height, depth = shape
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    z = np.linspace(0, depth - 1, depth)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    if center is None:
        center = (depth // 2, height // 2, width // 2)

    cz, cy, cx = center
    
    gaussian = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * sigma ** 2))

    return gaussian

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_matrix(matrix):
    """
    Plots a 3D matrix as dots in a 3D grid.
    
    - Dot size and color represent intensity values.
    - Ignores very small values for cleaner visualization.
    
    Parameters:
    - matrix: 3D NumPy array
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get matrix shape
    depth, height, width = matrix.shape
    
    # Generate 3D coordinates
    z, y, x = np.indices(matrix.shape)
    
    # Flatten the coordinates and values
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = matrix.flatten()
    
    # # Filter out small values to avoid clutter
    # threshold = np.max(values) * 0.1  # Keep only significant values
    # mask = values > threshold
    
    # x, y, z, values = x[mask], y[mask], z[mask], values[mask]
    
    # Normalize values for color mapping
    colors = values / np.max(values)  # Normalize between 0 and 1
    
    # Scatter plot with intensity-based size and color
    scatter = ax.scatter(x, y, z, c=colors, cmap="viridis", s=values * 100, alpha=0.8)
    
    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label("Intensity")

    # Labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Matrix Visualization")

    plt.show()


def flatten(mat):
    m = np.zeros((mat.shape[0],0))
    for i in range(0,mat.shape[2]):
        print(m.shape)
        m = np.hstack((m,mat[:,:,i]))
    print(m.shape)
    return m

def restruct(mat, shape):
    m = np.zeros(shape)
    for i in range(0, shape[2]):
        m[:,:, i] += mat[:, i*shape[1] : ((i+1)*shape[1])]
    return m
    

# Define matrix size and standard deviation
shape = (71, 71) 



# Generate Gaussian matrix
guassian_tensor = custom_2d_shape(shape, None, sx=25, sy=5, theta=1, p=4)

# Plot the matrix
plt.imshow(guassian_tensor, )
plt.colorbar(label="Intensity")
plt.title("2D Gaussian Matrix")
plt.show()

# # plot_3d_matrix(guassian_tensor)
# flat = flatten(guassian_tensor) 
# lol = restruct(flat, shape)
# # plt.imshow(flat)
# # plt.colorbar(label="Intensity")
# # plt.show()

U, S, VT = np.linalg.svd(guassian_tensor,full_matrices=False)
S = np.diag(S)
# print(U)
print(S)
# print(VT)
for r in (1, 2,3, 5, 7): # Construct approximate image
    # print(U[:,:r] )
    print(S[0:r,:r])
    # print(VT[:r,:])
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.imshow(Xapprox, )
    plt.colorbar(label="Intensity")
    plt.title("2D Gaussian Matrix")
    plt.show()

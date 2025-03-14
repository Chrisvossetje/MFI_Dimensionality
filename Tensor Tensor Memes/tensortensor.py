import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from matplotlib.image import imread

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


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
        m = np.hstack((m,mat[:,:,i]))
    return m

def restruct(mat, shape):
    m = np.zeros(shape)
    for i in range(0, shape[2]):
        m[:,:, i] += mat[:, i*shape[1] : ((i+1)*shape[1])]
    return m

import numpy as np

def three_norm(A, M):
    res = np.zeros((A.shape[0],A.shape[1],M.shape[1]))
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):  # Fixed the loop over third axis
            res[x, y, :] = M @ A[x, y, :]
    return res

def star_m(A, B, M, Minv=None):
    if Minv is None:
        Minv = np.linalg.inv(M)

    Ahat = three_norm(A, M)
    Bhat = three_norm(B, M)

    if A.shape[2] != B.shape[2]:
        raise ValueError("something is wrong :(")

    C = np.zeros((A.shape[0], B.shape[1], A.shape[2]))
    for i in range(A.shape[2]):
        C[:, :, i] = Ahat[:, :, i] @ Bhat[:, :, i]

    return three_norm(C, Minv)

def non_square_diag(vector, rows, cols):
    if len(vector) > min(rows, cols):
        raise ValueError("Vector length exceeds matrix dimensions")
    
    matrix = np.zeros((rows, cols))
    for i in range(len(vector)):
        matrix[i, i] = vector[i]
    return matrix

def full_tsvdm(A, M):
    Minv = np.linalg.inv(M)
    Ahat = three_norm(A, M)

    n = A.shape[2]
    minn = min(A.shape[0], A.shape[1])
    U = np.zeros((A.shape[0], A.shape[0], A.shape[2]))
    S = np.zeros_like(A)
    V = np.zeros((A.shape[1], A.shape[1], A.shape[2]))

    for i in range(n):
        Us, Ss, Vs = np.linalg.svd(Ahat[:, :, i])
        U[:, :, i] = Us
        S[:, :, i] += non_square_diag(Ss, A.shape[0], A.shape[1])
        V[:, :, i] = Vs

    U = three_norm(U, Minv)
    S = three_norm(S, Minv)
    V = three_norm(V, Minv)

    return U, S, V

# TEST FUNCTIONS 


# Test function
def test_three_norm():
    A = np.random.rand(5, 7, 9)   # 3x3 matrices, 4 slices
    M = np.random.rand(9, 9)      # 3x3 matrix

    # Compute result
    result = three_norm(A, M)


    assert (A == three_norm(A, np.identity(9))).all()

    # Check shape
    assert result.shape == A.shape, f"Shape mismatch: expected {A.shape}, got {result.shape}"

    print("All tests passed!")

def test_star_m():
    """Test for star_m function."""
    A = np.random.rand(6, 3, 4)
    B = np.random.rand(3, 7, 4)
    M = np.random.rand(4, 4)

    # Compute result
    result = star_m(A, B, M)
    
    # Check shape
    assert result.shape == (A.shape[0], B.shape[1], B.shape[2]), f"Shape mismatch: expected (3, 3, 4), got {result.shape}"

    print("star_m test passed!")


def test_full_tsvdm():
    """Test for full t-SVDM function."""
    A = np.random.rand(5, 3, 7)
    M = np.identity(7)

    U, S, V = full_tsvdm(A, M)

    L = A - (star_m(star_m(U,S,M), V, M))
    print(norm(L))

    print("full_tsvdm test passed!")

def norm(L):
    return np.sqrt(np.sum(L**2))


nii_image = nib.load("Mock_lesion_0.nii.gz")
image_array = nii_image.get_fdata() # Convert to numpy array


def k_trunc_svd(U, S, V, k):
    k = max(1,min(U.shape[1], V.shape[0], k))
    return U[:, 0:k, :], S[0:k, 0:k, :], V[0:k,:, :]

def remake_svd(U, S, V, M):
    return star_m(star_m(U,S,M), V, M)

def k_trunc(A, M, k):
    U, S, V = full_tsvdm(A,M)
    Uu, Ss, Vv = k_trunc_svd(U,S,V,k)
    return remake_svd(Uu, Ss, Vv, M), Uu.shape[0] * Uu.shape[1] * Uu.shape[2] + Vv.shape[0] * Vv.shape[1] * Vv.shape[2] 

def calc_dimensions(U, S, V):
    return sum(U.shape) + sum(S.shape) + sum(V.shape)



def non_square_diag(vector, rows, cols):
    if len(vector) > min(rows, cols):
        raise ValueError("Vector length exceeds matrix dimensions")
    
    matrix = np.zeros((rows, cols))
    for i in range(len(vector)):
        matrix[i, i] = vector[i]
    return matrix

def k_trunc_2d(A, k):
    M = flatten(A)
    U, S, V = np.linalg.svd(M)
    
    S = non_square_diag(S, U.shape[1], V.shape[0])

    Uu = U[:, 0:k]
    Ss = S[0:k, 0:k]
    Vv = V[0:k, :]

    return restruct(Uu @ Ss @ Vv, A.shape), (M.shape[0]*k) + (M.shape[1]*k) + k 




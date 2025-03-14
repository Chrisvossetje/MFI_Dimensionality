import numpy as np
import scipy
import scipy.linalg
from scipy.linalg import dft

import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import seaborn as sns
import pywt

# NOTE: A LOT OF THIS IS DUPLICATE CODE, SINCE WE DID NOT WANT MERGE ERRORS. WILL SANITIZE LATER!!
# Calculate A \times_3 M
def three_prod(A, M):
    res = np.zeros((A.shape[0],A.shape[1],M.shape[1]), dtype=complex)
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):  # Fixed the loop over third axis
            res[x, y, :] = M @ A[x, y, :]
    return res

def star_m(A, B, M, Minv=None):
    if Minv is None:
        Minv = np.linalg.inv(M)

    Ahat = three_prod(A, M)
    Bhat = three_prod(B, M)

    if A.shape[2] != B.shape[2]:
        raise ValueError("something is wrong :(")

    C = np.zeros((A.shape[0], B.shape[1], A.shape[2]), dtype=complex)
    for i in range(A.shape[2]):
        C[:, :, i] = Ahat[:, :, i] @ Bhat[:, :, i]

    return three_prod(C, Minv)

def non_square_diag(vector, rows, cols):
    if len(vector) > min(rows, cols):
        raise ValueError("Vector length exceeds matrix dimensions")
    
    matrix = np.zeros((rows, cols), dtype=complex)
    for i in range(len(vector)):
        matrix[i, i] = vector[i]
    return matrix

def full_tsvdm(A, M):
    Minv = np.linalg.inv(M)
    Ahat = three_prod(A, M)

    n = A.shape[2]
    U = np.zeros((A.shape[0], A.shape[0], A.shape[2]), dtype=complex)
    S = np.zeros_like(A, dtype=complex)
    V = np.zeros((A.shape[1], A.shape[1], A.shape[2]), dtype=complex)

    for i in range(n):
        Us, Ss, Vs = np.linalg.svd(Ahat[:, :, i])
        U[:, :, i] = Us
        S[:, :, i] += non_square_diag(Ss, A.shape[0], A.shape[1])
        V[:, :, i] = Vs

    U = three_prod(U, Minv)
    S = three_prod(S, Minv)
    V = three_prod(V, Minv)

    return U, S, V

# def plot_3d_matrix(matrix):
#     """
#     Plots a 3D matrix as dots in a 3D grid.
    
#     - Dot size and color represent intensity values.
#     - Ignores very small values for cleaner visualization.
    
#     Parameters:
#     - matrix: 3D NumPy array
#     """
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Generate 3D coordinates
#     z, y, x = np.indices(matrix.shape)
    
#     # Flatten the coordinates and values
#     x = x.flatten()
#     y = y.flatten()
#     z = z.flatten()
#     values = matrix.flatten()
    
#     # # Filter out small values to avoid clutter
#     # threshold = np.max(values) * 0.1  # Keep only significant values
#     # mask = values > threshold
    
#     # x, y, z, values = x[mask], y[mask], z[mask], values[mask]
    
#     # Normalize values for color mapping
#     colors = values / np.max(values)  # Normalize between 0 and 1
    
#     # Scatter plot with intensity-based size and color
#     scatter = ax.scatter(x, y, z, c=colors, cmap="viridis", s=values * 100, alpha=0.8)
    
#     # Add color bar
#     cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
#     cbar.set_label("Intensity")

#     # Labels and title
#     ax.set_xlabel("X-axis")
#     ax.set_ylabel("Y-axis")
#     ax.set_zlabel("Z-axis")
#     ax.set_title("3D Matrix Visualization")

#     plt.show()

def plot_2_3D_matrices(matrix1, matrix2, title_1="Original image", title_2="tSVD"):
    # fig = plt.figure(figsize=(8, 8))
    # ax1 = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(111, projection='3d')
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    
    # Generate 3D coordinates
    z1, y1, x1 = np.indices(matrix1.shape)
    z2, y2, x2 = np.indices(matrix2.shape)
    
    # Flatten the coordinates and values
    x1 = x1.flatten()
    y1 = y1.flatten()
    z1 = z1.flatten()
    values1 = matrix1.flatten()

    x2 = x2.flatten()
    y2 = y2.flatten()
    z2 = z2.flatten()
    values2 = matrix2.flatten()
    
    # Normalize values for color mapping
    colors1 = values1 / np.max(values1)  # Normalize between 0 and 1
    colors2 = values2 / np.max(values2)  # Normalize between 0 and 1
    
    # Scatter plot with intensity-based size and color
    scatter1 = ax1.scatter(x1, y1, z1, c=colors1, cmap="viridis", s=values1 * 100, alpha=0.8)
    scatter2 = ax2.scatter(x2, y2, z2, c=colors2, cmap="viridis", s=values2 * 100, alpha=0.8)
    
    # Add color bar
    cbar = plt.colorbar(scatter1, ax=ax1, shrink=0.5, pad=0.1)
    cbar.set_label("Intensity")
    cbar = plt.colorbar(scatter2, ax=ax2, shrink=0.5, pad=0.1)
    cbar.set_label("Intensity")

    # Labels and title
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")
    ax1.set_title(title_1)
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_zlabel("Z-axis")
    ax2.set_title(title_2)
    
    # Adjust layout for better visualization
    plt.tight_layout()

    plt.show()


def matrix_norm(A):
    return np.sqrt(np.sum(np.abs(A) ** 2))


def process_lesion_tsvd(path, M=None, plot=False, max_rank=5):
    """
    Process a lesion image using tSVD and compute truncated approximations.

    Parameters:
    - path (str): Path to the NIfTI file of the lesion.
    - M (ndarray): The transformation matrix to use in tSVD. Defaults to a 10x10 identity matrix if None.
    - plot (bool): Whether to plot the original image and the approximations.
    - max_rank (int): Maximum truncation rank for the SVD approximation.

    Returns:
    - image_array (ndarray): The original lesion image array.
    - result (ndarray): The tSVD reconstructed image from the full decomposition.
    """
    # Assume full_tsvdm, star_m, and plot_2_3D_matrices are already defined elsewhere.

    # Set default transformation matrix if not provided
    if M is None:
        M = np.identity(10)

    # Import lesion
    nii_image = nib.load(path)
    image_array = nii_image.get_fdata()  # Convert to numpy array

    # Compute full tSVD decomposition and reconstruction
    U, S, V = full_tsvdm(image_array, M)
    result = star_m(star_m(U, S, M), V, M)

    # Plot the original image and the full tSVD approximation
    if plot:
        plot_2_3D_matrices(image_array, np.real(result))

    # Compute and print the overall difference between the original image and the tSVD result
    delta = np.sum((image_array - result)**2)
    if plot:
        print("Difference image and tSVD: " + str(delta))

    # Define a helper function for truncating the tSVD decomposition
    def svd_rank_r_truncation(U, S, V, k):
        k = max(1, min(U.shape[1], V.shape[0], k))
        return U[:, :k, :], S[:k, :k, :], V[:k, :, :]

    # Loop over truncation ranks from 1 to max_rank to compare approximations
    errors = []
    for rank in range(1, max_rank + 1):
        U_trunc, S_trunc, V_trunc = svd_rank_r_truncation(U, S, V, rank)
        result_trunc = star_m(star_m(U_trunc, S_trunc, M), V_trunc, M)

        # Compute and print the squared error between the full tSVD and the truncated approximation
        sqr_error = np.sum((result - result_trunc)**2)
        errors.append(sqr_error)
        if plot:
            print(f"Squared error for rank {rank}: {sqr_error}")


        # Plot the truncated approximation if plotting is enabled
        if plot:
            plot_2_3D_matrices(image_array, np.real(result_trunc), title_2=f"tSVD approx for rank {rank}")

    return image_array, result, delta, errors

def collect_errors(num_lesions=1000, M=None, plot=False, max_rank=5):
    """
    Process multiple lesion files and collect error metrics in a pandas dataframe.

    Parameters:
    - num_lesions (int): Number of lesion files to process.
    - M (ndarray): The transformation matrix for tSVD (defaults to 10x10 identity if None).
    - plot (bool): Whether to enable plotting in process_lesion_tsvd.
    - max_rank (int): Maximum truncation rank for tSVD approximations.

    Returns:
    - df (pandas.DataFrame): A dataframe containing the overall difference ('delta') and 
      truncated approximation errors (one column per rank) for each lesion.
    """
    results = []
    
    for i in range(num_lesions):
        file_path = f"MockLesions/MockLesions/Mock_lesion_{i}.nii.gz"
        try:
            # Process the lesion and obtain the overall difference (delta) and errors per rank.
            image_array, result, delta, errors = process_lesion_tsvd(
                path=file_path, M=M, plot=plot, max_rank=max_rank
            )
            
            # Create a dictionary to store error metrics for this lesion.
            lesion_dict = {"lesion_index": i, "delta": delta}
            # Assuming errors is a list with one error per rank (from rank 1 to max_rank)
            for rank in range(1, max_rank + 1):
                lesion_dict[f"error_rank_{rank}"] = errors[rank - 1]
            results.append(lesion_dict)
        except Exception as e:
            print(f"Error processing lesion {i}: {e}")
    
    # Convert the list of dictionaries to a pandas dataframe
    df = pd.DataFrame(results)
    return df
M_dft = dft(10)


# Example usage:
df_errors = collect_errors(num_lesions=1000, M= M_dft, plot=False, max_rank=5)


# Set seaborn style for prettier plots
sns.set(style="darkgrid")

# Plot the overall delta error per lesion
plt.figure(figsize=(10, 5))
plt.plot(df_errors["lesion_index"], df_errors["delta"], color="red", label="Delta error")
plt.title("Delta Error across Lesions")
plt.xlabel("Lesion Index")
plt.ylabel("Delta Error")
plt.legend()
plt.show()

# Plot errors for each rank
plt.figure(figsize=(10, 6))
for rank in range(1, 6):  # Adjust range to your max rank
    plt.plot(df_errors["lesion_index"], df_errors[f"error_rank_{rank}"], label=f"Rank {rank}")

plt.title("Truncated SVD Errors by Rank across Lesions")
plt.xlabel("Lesion Index")
plt.ylabel("Error Value")
plt.legend()
plt.show()

# Optional: Boxplot to summarize error distribution by rank
plt.figure(figsize=(10, 6))
rank_columns = [f"error_rank_{rank}" for rank in range(1, 6)]
sns.boxplot(data=df_errors[rank_columns])
plt.title("Error Distribution by Rank")
plt.xlabel("Rank")
plt.ylabel("Error Value")
plt.show()

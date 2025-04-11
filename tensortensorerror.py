import numpy as np
import scipy
import scipy.linalg
from scipy.linalg import dft

import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import seaborn as sns

# NOTE: A LOT OF THIS IS DUPLICATE CODE, SINCE WE DID NOT WANT MERGE ERRORS. WILL SANITIZE LATER!!
# Calculate A ×_3 M
def three_prod(A, M):
    res = np.zeros((A.shape[0], A.shape[1], M.shape[1]), dtype=complex)
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

def plot_2_3D_matrices(matrix1, matrix2, title_1="Original image", title_2="tSVD"):
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

def process_lesion_tsvd(path, M=None, plot=False, max_rank=5, trunc_lesions_1=False):
    """
    Process a lesion image using tSVD and compute truncated approximations.

    Parameters:
    - path (str): Path to the NIfTI file of the lesion.
    - M (ndarray): The transformation matrix to use in tSVD. Defaults to a 10x10 identity matrix if None.
    - plot (bool): Whether to plot the original image and the approximations.
    - max_rank (int): Maximum truncation rank for the SVD approximation.
    - trunc_lesions_1 (bool): Whether to truncate the lesion image dimensions.

    Returns:
    - image_array (ndarray): The original lesion image array.
    - result (ndarray): The tSVD reconstructed image from the full decomposition.
    """
    # Set default transformation matrix if not provided
    if M is None:
        M = np.identity(10)

    # Import lesion
    nii_image = nib.load(path)
    image_array = nii_image.get_fdata()  # Convert to numpy array
    if trunc_lesions_1:
        image_array = image_array[1:9, 1:9, 1:9]

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

def collect_errors(num_lesions=1000, M=None, plot=False, max_rank=5, trunc_lesions=False):
    """
    Process multiple lesion files and collect error metrics in a pandas dataframe.

    Parameters:
    - num_lesions (int): Number of lesion files to process.
    - M (ndarray): The transformation matrix for tSVD (defaults to 10x10 identity if None).
    - plot (bool): Whether to enable plotting in process_lesion_tsvd.
    - max_rank (int): Maximum truncation rank for tSVD approximations.
    - trunc_lesions (bool): Whether to truncate the lesion image dimensions.

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
                path=file_path, M=M, plot=plot, max_rank=max_rank, trunc_lesions_1=trunc_lesions
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

# Create a 10x10 DFT matrix
n = 10
omega = np.exp(-2j * np.pi / n)
DFT_matrix = np.array([[omega ** (i * j) for j in range(n)] for i in range(n)])
# Optionally, normalize the DFT matrix if desired:
# DFT_matrix = DFT_matrix / np.sqrt(n)

wavelet_matrix = np.array([
    [1,  1,  1,  1,  1,  1,  1,  1],
    [1,  1,  1,  1, -1, -1, -1, -1],
    [1,  1, -1, -1,  0,  0,  0,  0],
    [0,  0,  0,  0,  1,  1, -1, -1],
    [1, -1,  0,  0,  0,  0,  0,  0],
    [0,  0,  1, -1,  0,  0,  0,  0],
    [0,  0,  0,  0,  1, -1,  0,  0],
    [0,  0,  0,  0,  0,  0,  1, -1]
])
matrices = [("Identity", None), ("DFT", DFT_matrix), ("Haar", wavelet_matrix)]

sns.set(style="darkgrid")

# Initialize list to collect summary statistics
summary_data = []

for matrix_name, matrix_value in matrices:
    # Compute error DataFrame using collect_errors
    if matrix_name == "Haar":
        df_errors = collect_errors(num_lesions=1000, M=matrix_value, plot=False, max_rank=5, trunc_lesions=True)
    else:
        df_errors = collect_errors(num_lesions=1000, M=matrix_value, plot=False, max_rank=5, trunc_lesions=False)
    
    # Compute the mean overall difference and the mean error for each rank
    mean_delta = df_errors['delta'].mean()
    mean_errors = {f"error_rank_{rank}": df_errors[f"error_rank_{rank}"].mean() 
                   for rank in range(1, 6)}
    
    # Combine into one summary dictionary for this matrix
    summary_dict = {"Matrix": matrix_name, "delta": mean_delta}
    summary_dict.update(mean_errors)
    summary_data.append(summary_dict)
    
    # Create a boxplot to visualize the error distribution by rank for this matrix
    plt.figure(figsize=(10, 6))
    rank_columns = [f"error_rank_{rank}" for rank in range(1, 6)]
    ax = sns.boxplot(data=df_errors[rank_columns])
    ax.set_xticklabels([f"Rank {rank} error" for rank in range(1, 6)])
    plt.title(f"Error Distribution by Rank (Matrix: {matrix_name})")
    plt.xlabel("Rank")
    plt.ylabel("Error Value")
    plt.savefig(f"boxplot_{matrix_name}.png")
    plt.close()

# Create a summary DataFrame with rows corresponding to matrices and columns for each error metric.
summary_df = pd.DataFrame(summary_data).set_index("Matrix")
print(summary_df)
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 14:03:09 2025

@author: Sjoerd
"""

import os

# Global constants
# You can also use a raw path. On UNIX, make sure it ends with a /
BASE_PATH = os.path.expanduser('~\\Documents\\Uni\\Industry\\MockLesions')
DEBUG_INFO = True
LESION_MAT_SHAPE = (10,10,10)
SHOW_WARNINGS = True

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import nibabel as nib # Necessary for importing lesions

# When debugging is activated, time durations are printed
if DEBUG_INFO:
    import time
    
if not SHOW_WARNINGS:
    import warnings
    warnings.filterwarnings('ignore')

def print_report(verbose, r, timestamp=None):
    """Print verbosity report. Also returns current timestamp."""
    
    if not verbose: return None
    
    t = time.perf_counter()
    
    if timestamp==None:
        print(r)
    else:
        print(f"{r} [In {t - timestamp:0.4f} seconds]")
    
    return t

#-----------------------------------------------------------------------------#
# Functions for handling lesions                                              #
#-----------------------------------------------------------------------------#
def import_lesions(file_path, filename_base = "Mock_lesion_", 
                   filename_extension = ".nii", as_vector=True, 
                   lesion_count = -1, verbose = DEBUG_INFO):
    """
    Import a set of previously generated lesions from numbered files.
    
    Parameters
    ----------
    file_path : str
        The path from which the files are to be loaded.
    filename_base : str, default 'Mock_lesion_'
        The filename base. The complete filename would be
        `filename_base` + <index> + `filename_extension`.
    filename_extension: str, default '.nii'
        The extension for the filenames.
    as_vector: bool, default True
        If true, the lesions are returned in vectorized format, rather than
        as intensity matrices.
    lesion_count : int, default -1
        The amount of lesions that should be imported. If negative, all lesions
        from the path are imported.
    verbose: bool, default `DEBUG_INFO`
        If true, verbosity reports are printed.

    Returns
    -------
    nparray of float
        A list of imported lesions. The shape of the list depends on whether
        or not the lesions were vectorized.
    tuple of int
        The (original) shape of the imported lesions intensity matrices.
    """
    
    # Make sure path and extension are proper
    if (not file_path.endswith("\\")) and (not file_path.endswith("/")): 
        file_path=file_path+"\\"
    if not filename_extension.startswith("."): 
        filename_extension = "." + filename_extension
    
    # Define function for getting complete path & extracting a lesion
    def get_path(i): return file_path + filename_base + str(i) + filename_extension
    def extract_lesion(p): return np.array(nib.load(p).get_fdata())
    
    
    # If the lesion_count is not set, set it
    if lesion_count < 0:
        b = True
        i = -1
        while b:
            i += 1
            b = os.path.isfile(get_path(i))
        lesion_count = i
    
    # Print debug info
    tic = print_report(verbose, "Importing " + str(lesion_count) + " lesions...")
        
    # Find lesion shape, necessary for initializing the result variable
    lesion_shape=np.shape(extract_lesion(get_path(0)))
    
    # If the lesions are imported as vectors, the result is a list of vectors
    # rather than of 10x10x10 matrices.
    if as_vector:
        result = np.zeros((lesion_count, np.prod(lesion_shape)))
        
        for i in range(0, lesion_count):
            result[i] = inmat_to_vector(extract_lesion(get_path(i)))
    else:
        result = np.zeros((lesion_count,) + lesion_shape)
        
        for i in range(0, lesion_count):
            result[i] = extract_lesion(get_path(i))
        
    # Print debug info
    print_report(verbose, "Lesions imported!", tic)
        
    # Return result variable
    return result, lesion_shape

def inmat_to_vector(inmat):
    """Return the vectorized version of a given intensity matrix/lesion."""
    return np.ravel(inmat)

def vector_to_inmat(vector, shape):
    """Return the intensity matrix of a vectorized lesion, based on a shape."""
    return np.reshape(vector, shape)

#-----------------------------------------------------------------------------#
# Functions for analysis                                                      #
#-----------------------------------------------------------------------------#

def calc_SVD_basis(a, d = -1, full_basis = False, return_svals = False, 
                    verbose = DEBUG_INFO):
    """
    Calculate the basis (and respective singular values) of a set of vectors 
    using SVD.
    
    Parameters
    ----------
    a : (M,N) array of float
        A list of N-dimensional real vectors.
    d : int, optional
        The desired dimension of the (truncated) basis. Defaults to 0, in
        which case the entire basis is returned.
    full_basis : bool, optional
        If True, the basis is returned including vectors with a singular value
        of 0. Defaults to False.
    return_svals : bool, optional
        If True, an array of singular values is returned as well. Defaults to
        False.
    verbose : bool, optional
        If True, verbosity reports are given. Defaults to DEBUG_INFO.

    Returns
    -------
    (min(M,N),N) array of float
        The list of vectors spanning the input vectors.
    array of float, optional
        The array containing the singular values of the input array.
    
    Raises
    ------
    LinAlgError
        If SVD computation does not converge.
    
    Notes
    -----
        The SVD-basis is based on the first matrix of (left-)singular vectors
        in the SVD-equation. Given matrix M, SVD finds U and Vt such that
                                    M=USVt. 
        If we let u_i be the columns of u, s_i the singular values in S, and 
        v_i the rows of Vt, the singular value description of this system 
        becomes
                                Mv_i = s_iu_i.
        From this, we may conclude that the columns of U are linear combinations
        of the columns of M, so the reverse is true as well. Hence, the returned
        basis-matrix is the matrix U.
        Note that we can actually find the reverse spanning as well, as
                                 m_i = USw_i
        where w_i is the i-th column of Vt.
    """
    
    # Print debug info
    tic = print_report(verbose, "Calculating SVD basis of " 
                                + str(np.shape(a)[1]) + " vectors...")
    
    # We cannot truncate to a higher dimension than the amount of given
    # vectors, so then we must include vectors with a singular value of 0.
    if d > len(a):
        full_basis = True
    
    # Apply SVD and get basis
    U, S = np.linalg.svd(np.transpose(a),full_matrices=full_basis)[0:2]
    
    # Truncate results to desired dimension
    if d > 0:
        U = U[:,:d]
        S = S[:d]
        
    # Print debug info
    print_report(verbose, "SVD basis calculated!", tic)
    
    # Return results (U transposed, to make an array)
    if return_svals:
        return np.transpose(U), S
    else:
        return np.transpose(U)

def calc_SV_energy(a):
    """ Calculate the cumulative energy for given singular values. """
    return np.cumsum(a)/np.sum(a)

def calc_SV_cutoff(a, t):
    """
    Calculate the cutoff indices for singular values based on energy retention.
    
    Parameters
    ----------
    a : array-like
        The array containing the cumulative energy values of singular values.
    t: array-like
        The array of desired energy retention threshold values, in range [0,1].
    
    Returns
    -------
    array of int
        The array of cutoff indices corresponding to the given thresholds.
    """
    
    # Initialize result array
    result = np.zeros(len(t), np.uint)
    
    # Find threshold indices
    for i,threshold in enumerate(t):
        result[i] = int(np.floor(len(a[a <= threshold]) - 1))
    
    # Return results
    return result

def calc_SVD_basis_local_trivial(a, d, verbose = DEBUG_INFO):
    """
    Calculate bases of the set of input vectors using LSVD. The bases are
    of the size of a given desired dimension d, by truncating the LSVD result,
    and they apply to the trivial clusterings of the input vectors where each
    cluster is regarded as some input vector and its (d - 1) closest neighbours,
    i.e. the result will extacly be the d-dimensional plane spanned by these
    vectors.
    
    Parameters
    ----------
    a : (M,N) array
        A sequence of vectors.
    d: int
        The desired dimension of the result bases.
    verbose : bool, optional
        If True, prints verbosity reports. Defaults to DEBUG_INFO.

    Returns
    -------
    (M,d,N) array
        An array of bases of the list of input vectors.
    (M, N) array
        An array containing support vectors for each basis.
    
    Raises
    ------
    LinAlgError
        If SVD computation does not converge.
    
    Notes
    -----
        Same as local_svd, but this time it creates a patch for each vector
    """

    # Print debug info
    tic = print_report(verbose, "Calculating trivial local SVD bases...")
    
    # Initialize intermediate and result variables
    support_vectors = np.copy(a)
    bases = np.zeros((len(a),) + np.shape(a[:d + 1]))
    
    translated = np.zeros(np.shape(a))
    sorted_indices = np.zeros(len(a), dtype=int)

    # Calculate results
    for i,v in enumerate(a):
        # Get closest vectors and find and use support vector
        sorted_indices = np.argsort(np.linalg.norm(a - v, axis=1))
        current_patch = sorted_indices[:d + 1]
        translated = a[current_patch] - v
        
        # Calculate basis
        bases[i] = calc_SVD_basis(translated, verbose=False)
        
    # Print verbosity report
    print_report(verbose, "Trivial local SVD applied!", tic)
    
    # Return results
    return bases, support_vectors
    
def calc_SVD_basis_local(a, d, max_err = 0.1, max_err_abs = False,
                         max_err_quantile = 1.0, init_flat_count = 1, 
                         verbose = DEBUG_INFO):
    """
    Calculate bases of the set of input vectors using Local SVD (LSVD). The
    bases are chosen such that the compression error (the objective function)
    is bounded, and minimized for the amount of planes used. This combines LSVD 
    with q-flat clustering.
    
    For more information, see the Notes.
    
    Parameters
    ----------
    a : (M,N) array
        A sequence of vectors.
    d: int
        The desired dimension of the result bases.
    max_err: float in range [0, 1], optional
        The maximal value of the compression error. Defaults to 0.1.
    max_err_quantile: float in range [0, 1], optional
        The quantile at which the error distribution may not exceed `max_err`
        for the algorithm to finish. Reducing this may prevent overfitting. 
        Defaults to 1.0.
    init_flat_count : int, optional
        The amount of flats that is initialized at the start of the algorithm.
        Defaults to 1. If a higher value is chosen, it is possible for a 
        (locally) non-optimal amount of planes to be found, if the final result
        has empty clusters. In that case, empty clusters are removed before
        returning the resulting bases of the optimal planes.
    max_err_abs : bool, optional
        If True, the maximal error is interpreted as absolute rather than
        relative. Defaults to False.
    verbose : bool, optional
        If True, prints verbosity reports. Defaults to DEBUG_INFO.

    Returns
    -------
    (P,d,N) array
        An array of bases of the list of input vectors.
    (P, N) array
        An array containing support vectors for each basis.
    
    Raises
    ------
    LinAlgError
        If SVD computation does not converge.
    
    Notes
    -----
        The principle behind this function relies on the assumption that 
        (locally) the input vectors can be described as d-dimensional. By 
        clustering the set of vectors into sets resembling planes, we can 
        approximate their d-dimensional forms using SVD-truncation of 
        the bases of these planes.
        
        To find clusters, we apply a q-flat clustering (qFC) algorithm. If
        the found clusters do not satisfy the error bound, another plane is
        added and the algorithm is repeated. When a new plane is added, it
        is based on a vector which distance to the closest plane exceeds the
        error bound, and its (d-1) closest neighbours.
        
        When the entire set has been clustered with fitting d-dimensional 
        bases, one may consider the (d+1)-dimensional compression of the
        initial vectors, by letting the first value of the vector indicate the
        cluster/basis index, and the average vector in the cluster be the
        central vector of the cluster, i.e. the support vector of the plane. 
        The compression of a new vector could then rely on the basis 
        of compression from which cluster yields the lowest error.
    """
    
    # Make init_flat_count bound
    init_flat_count = int(np.max((init_flat_count, 1)))
    init_flat_count = int(np.min((init_flat_count, np.floor(len(a)/d))))
    
    def _assign_clusters(a, W, g):
        """Assign vectors to plane clusters using the q-flat algorithm."""
        k = len(W) # Plane count
        m = len(a) # Vector count
        
        errors_new = np.zeros((k, m))
        #gg = np.vstack([g[0]] * m)
        errors_new[0] = np.linalg.norm(a @ W[0] - g[0], axis=1) # Squaring only slows down and does not impact results, and neither does transposing
        for i in range(1, k):
            #gg = np.vstack([g[i]] * m)
            errors_new[i] = np.linalg.norm(a @ W[i] - g[i], axis=1)
        
        cluster_assign = np.argmin(errors_new, axis=0)
        return cluster_assign, np.min(errors_new, axis=0)
    
    def _update_plane(A, p):
        """Update plane for cluster using q-flat algorithm"""
        ml = len(A)        
        
        B = np.transpose(A) @ (np.identity(ml) - np.ones((ml,ml))/ml) @ A
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        lowest_indices = np.argsort(eigenvalues)
        W = eigenvectors[:,lowest_indices[:p]]
        
        g = np.ones(ml) @ A @ W/ml
        
        return W, g
    
    def _get_plane_new(a, s, d):
        """Get a new plane based around a base vector."""
        
        # Get the initial cluster (local d-dimensional plane)
        sorted_indices = np.argsort(np.linalg.norm(a - s, axis=1))
        cluster = (a[sorted_indices])[:d]
        
        return _update_plane(cluster, len(s) - d)
    
    # Print verbosity report
    tic = print_report(verbose, "Finding optimal q-flat for LSVD...")
    
    # Pre-calculate lengths
    lengths_inv = 1/np.linalg.norm(a, axis=1)
    if max_err_abs: lengths_inv = np.ones(len(a))

    # Set apropriate variables, in accordance to q-flat
    k = init_flat_count
    n = len(a[0])
    m = len(a)
    q = d
    p = n - q

    # Initialize q-flat alg planes
    tic2 = print_report(verbose, "Initializing " + str(k) + " initial planes...")
    W = np.zeros((k, n, p))
    g = np.zeros((k, p))
    
    # For the first plane, every vector is suitable
    base_vec = a[np.random.choice(m, 1)[0]]
    W[0], g[0] = _get_plane_new(a, base_vec, d)
    cluster_assign, cluster_errors = _assign_clusters(a, [W[0]], [g[0]])

    # Base additional planes around vectors exceeding the maximal error
    # Also initializes the cluster assignments
    for i in range(1, k):
        suitable = np.where(cluster_errors * lengths_inv > max_err)[0]
        base_vec_index = suitable[np.random.choice(len(suitable), 1)[0]]
        base_vec = a[base_vec_index]
        W[i], g[i] = _get_plane_new(a, base_vec, d)
        #print("Dist: " + str(_assign_clusters(a, [W[i]], [g[i]])[1][base_vec_index]))
        cluster_assign, cluster_errors = _assign_clusters(a, W[:i+1], g[:i+1])
        #print(base_vec_index)
        #print(str(cluster_assign[base_vec_index]) + ": " + str(cluster_errors[base_vec_index]))
    
    # Without this step the first loop always malfunctions
    if k > 1:
        for i in range(k):
            W[i], g[i] = _update_plane(a[cluster_assign == i], p)
    cluster_assign, cluster_errors = _assign_clusters(a, W, g)
    
    print_report(verbose, "Initial planes initialized! About to start main loop", tic2)

    # Main loop
    obj_result = np.quantile(cluster_errors * lengths_inv, max_err_quantile)
    min_err_change = max_err/100 # For q-flat clustering loop
    while obj_result > max_err:
        # Add plane (based around poorly-performing vector)
        W = np.vstack((W, np.zeros(W[0].shape)[None,:,:]))
        g = np.vstack((g, np.zeros(g[0].shape)))
        k = len(W)

        suitable = np.where(cluster_errors * lengths_inv > max_err)[0]
        base_vec_index = suitable[np.random.choice(len(suitable), 1)[0]]
        base_vec = a[base_vec_index]
        W[k-1], g[k-1] = _get_plane_new(a, base_vec, d)
        #print("Dist: " + str(_assign_clusters(a, [W[k-1]], [g[k-1]])[1][base_vec_index]))
        
        # Assign clusters
        cluster_assign, cluster_errors = _assign_clusters(a, W, g)
        #print(str(base_vec_index) + " index out of " + str(len(suitable)))
        #print(str(cluster_assign[base_vec_index]) + ": " + str(cluster_errors[base_vec_index]))
        
        # Loop for q-flat clustering
        done = False
        tic2 = print_report(verbose, "Added plane! (total: " + str(k) + "). Applying qFC...")
        print_report(verbose, "Based around " + str(base_vec_index) + ", overshooters left = " + str(len(suitable)))
        while not done:
            cluster_assign_old = cluster_assign.copy()
            max_err_old = np.max(cluster_errors)
            
            # Update clusters
            for i in range(k):
                cluster = a[cluster_assign == i]
                if len(cluster) > 0:
                    W[i], g[i] = _update_plane(cluster, p)
            
            # Assign clusters
            cluster_assign, cluster_errors = _assign_clusters(a, W, g)
            
            # Check if done (no changes or minimal obj-func change)
            done = ((np.all(cluster_assign_old == cluster_assign))
                    or (max_err_old - np.max(cluster_errors) < min_err_change))
        
        # Find new objective result
        obj_result = np.quantile(cluster_errors * lengths_inv, max_err_quantile)
        print_report(verbose, "qFC applied! Error overshoot is " + str(obj_result - max_err), tic2)
        print_report(verbose, "Actual upper bound overshoot is " + str(np.max(cluster_errors * lengths_inv) - max_err))
    
    # Print verbosity report
    print_report(verbose, "qFC applied!", tic)
    tic = print_report(verbose, "Applying LSVD...")
    
    # Make SVD-bases of the (active) clusters
    unique_clusters = np.unique(cluster_assign)
    k = len(unique_clusters)
    
    if len(W) > k:
        print_report(verbose, "Found and cut " + str(len(W) - k) + " redundant clusters!")
    
    supps = np.zeros((k, np.shape(a)[1]))
    bases = np.zeros((k,) + np.shape(a[:d]))
    
    for i in range(k):
        cluster = a[cluster_assign == unique_clusters[i]]
        supps[i] = np.average(cluster, axis=0)
        translated = cluster - supps[i]
        bases[i] = calc_SVD_basis(translated, d, full_basis=True, verbose=False)
    
    print_report(verbose, "LSVD applied!", tic)    
    return bases, supps
    
    

def calc_similarity(a, verbose=DEBUG_INFO):
    """
    Calculates the similarity between vectors. When comparing multiple vectors,
    the maximal siliarity is chosen, and the simiarities are only calculated
    backwards, i.e. the similarity of vector j is chosen as the maximal
    similarity between vector j and vectors 0:j-1.
    
    Parameters
    ----------
    a : array-like
        Array-like argument containing the vectors which need be compared.
    verbose : bool, optional
        If True, verbosity reports are printed. Defaults to DEBUG_INFO.
        
    Returns
    -------
    float, or array of float
        Either the result of the similarity function if only two vectors
        are given, or the maximal backwards result if more vectors are given.
    """
    
    # Define the similarity function itself
    def similarity_function(u,v):
        return np.sqrt(np.sum(np.abs(u * v))
                            /(np.linalg.norm(u) * np.linalg.norm(v)))
    
    # If only two vectors are given, simply return the similarity
    if len(a) == 2:
        return similarity_function(a[0], a[1])
    
    
    # Print debug info
    tic = print_report(verbose, "Calculating similarity of " + str(len(a)) + " vectors...")
    
    # Initialize result array
    result = np.zeros(len(a))
    
    # Go over each vector, except the first one
    for i,u in enumerate(a[1:,:]):
        # Go over the vectors already passed, i.e. look "backwards"
        for j,v in enumerate(a[0:i,:]):
            # Adjust the result variable if necessary
            result[i] = max(result[i], similarity_function(u,v))
    
    # Print debug info
    print_report(verbose, "Vector similarity calculated!", tic)
    
    # Return results
    return result



# This is basically also local SVD
def get_k_plane_clustering(a, k, dim=0, local_initialization=False, verbose=DEBUG_INFO):
    if dim <= 0: dim = len(a[0]) - 1
    
    tic = print_report(verbose, "Calculating plane clustering...")
    
    # Initialize result variables (randomly)
    supps = np.zeros((k, len(a[0])))
    bases = np.zeros((k, dim, len(a[0])))
    
    if local_initialization: # Initialize based on nearest neighbours
        supps = a[np.random.choice(len(a),k)]
        
        for i,s in enumerate(supps):
            sorted_indices = np.argsort(np.linalg.norm(a - s, axis=1))
            cluster = a[sorted_indices[:dim]]
            supps[i] = np.average(cluster, axis=0)
            bases[i] = calc_SVD_basis(cluster - supps[i], verbose=False)[0]
    else: # Completely random clusters
        for i in range(len(bases)):
            cluster = a[np.random.choice(len(a),dim)]
            supps[i] = np.average(cluster, axis=0)
            bases[i] = calc_SVD_basis(cluster - supps[i], verbose=False)[0]
    
    # Initialize intermediate variables
    remat = get_decompression_matrix(bases) @ get_compression_matrix(bases)
    plane_errors = np.zeros((k, len(a)))
    plane_assign = np.zeros(len(a))
    
    # Initial cluster assignment
    for p in range(k):
        translated = np.transpose(a - supps[p])
        plane_errors[p] = np.linalg.norm(translated - remat[p] @ translated, axis=0)
        
    plane_assign = np.argmin(plane_errors, axis=0)
    
    done = False
    while not done:
        tic2 = print_report(verbose, "Renewing plane assignment...")
        plane_assign_old = np.copy(plane_assign) # Later we check for changes
        unused_planes = 0
        
        # Cluster update (find best planes for each cluster)
        for p in range(k):
            cluster = a[plane_assign == p]

            if len(cluster) > 0:
                supps[p] = np.average(cluster)
                U = calc_SVD_basis(cluster - supps[p], verbose=False)[0]
                if len(U) > dim: U = U[:dim]
                bases[p][:len(U),:] = U
            else:
                unused_planes += 1
                pass #? Not really supposed to happen I guess
                cluster = a[np.random.choice(len(a),dim)]
                supps[i] = np.average(cluster, axis=0)
                bases[i] = calc_SVD_basis(cluster - supps[i], verbose=False)[0]
        
        # (Re-calculate the reconstruction matrix)
        remat = get_decompression_matrix(bases) @ get_compression_matrix(bases)
        
        # Cluster assignment
        for p in range(k):
            translated = np.transpose(a - supps[p])
            plane_errors[p] = np.linalg.norm(translated - remat[p] @ translated, axis=0)
        
        plane_assign = np.argmin(plane_errors, axis=0)
        print_report(verbose, "Plane assignment renewed! (" + str(unused_planes) + " unused planes...)", tic2)
        
        # Check if done
        if np.all(plane_assign_old == plane_assign):
            done = True
    
    print_report(verbose, "Plane clustering calculated!", tic)
    
    return bases, supps
        
                
    

#-----------------------------------------------------------------------------#
# Functions for compression and decompression                                 #
#-----------------------------------------------------------------------------#

def get_compression_matrix(a, d=0):
    """
    Calculate the compression matrix for a given set of vectors
    
    Parameters
    ----------
    a : ndarray
        The array containing the vectors which form a basis (i.e. the
        transpose of the matrix of such a basis), or a set of such arrays.
    d : int, optional
        The dimension of the desired compression. Defaults to 0, in which
        case the dimension of the compression will be the size of the basis.
    
    Returns
    -------
    array
        A 2D compression matrix, or a set of such matrices.
    
    Notes
    -----
        Given the linear equation m_i = Bv_i, the compression matrix gives the
        matrix which can be used to calculate v_i from m_i using linear least
        squares using the columns of B. Note that the input array `a` is (or
        contains) the transpose of a basis.
    """
    
    def _comp_mat(basis):
        return np.linalg.inv(basis @ np.transpose(basis)) @ basis
    
    # It's easiest if the input vector is a 2D-matrix
    if len(np.shape(a)) == 2:
        if d <= 0: d = len(a)
        return _comp_mat(a[:d])
    
    if d<=0: d = np.shape(a)[1]
    result = np.zeros((len(a),d,np.shape(a)[2]))
    
    for i,b in enumerate(a):
        result[i] = _comp_mat(b[:d])
    
    return result

def compress_lesion(a, c):
    """
    Describe a lesion in terms of eigenlesions. For more information, see the
    Notes of get_compression_matrix.
    
    Parameters
    ----------
    lesion : ndarray
        A vectorized lesion, or a set of such vectorized lesions.
    c : array
        The compression matrix for the desired compression.
        
    Returns
    -------
    ndarray
        Either one vector, or a set of multiple vectors. These are the
        coefficients which, when multiplied by the basis defining the
        compression matrix, approximate `a`.
    """

    return np.transpose(c @ np.transpose(a))

def compress_lesion_local(a, c, d, s = None, return_error=False, verbose=DEBUG_INFO):
    """
    Describe a vectorized lesion, or a set of vectorized lesions in terms of 
    local eigenlesions from local SVD bases. Also pick the best local basis 
    for this job.
    
    Parameters
    ----------
    a : array-like
        Either a single vectorized lesion, or a set of vectorized lesions.
    c : array-like
        The compression matrices.
    d : array-like
        The decompression matrices.
    s : array-like or None, optional
        A set of support vectors, which are used to translate the input
        vector(s) to the right plane in order to do compression for the given
        bases. Defaults to None, in which case it is filled with zeroes.
    return_error : bool, optional
        Lorem ipsum
    verbose : bool, optional
        Lorem ipsum
    
    Returns
    -------
    array
        A compressed vectorized lesion, or a set of such. The dimension is
        based on the size of the bases in `b` plus one, as the first value
        is used to indicate which basis and support vector are used.
    array (optional)
        errors
    """
    
    def _comp_err(a, c, d):
        return np.linalg.norm(np.transpose(d @ c @ np.transpose(a)) - a, axis=1)
        #return np.linalg.norm(compress_decompress(a, c, d, verbose=False) - a, axis=1)
    
    # Pre-calculate the compression and decompression matrices
    comp_mat = c
    decomp_mat = d
    comp_dim = np.shape(c)[1]
    
    if s is None:
        s = np.zeros((len(c)))
    
    # The process is easier for only 1 lesion
    if len(np.shape(a)) < 2:
        min_err = 1
        result = np.zeros(comp_dim + 1)
        
        for i,support in enumerate(s):
            translated = a - support
            err = _comp_err(translated, comp_mat[i], decomp_mat[i])
            
            if err < min_err:
                min_err = err
                result[1:] = compress_lesion(translated, comp_mat)
                result[0] = i
        
        return result
    
    # This seems optimized, although the sequence operations are pretty bad
    # Initialize intermediate and result variables
    translated = a - s[0]
    min_err = _comp_err(translated, comp_mat[0], decomp_mat[0])
    result = np.zeros((len(a), comp_dim + 1))
    result[:,1:] = compress_lesion(translated, comp_mat[0])  
    result[:,0] = 0  

    for i,support in enumerate(s[1:]):
        translated = a - support
        err = _comp_err(translated, comp_mat[i + 1], decomp_mat[i + 1])
        
        result[err<min_err,1:] = compress_lesion(translated, comp_mat[i + 1])[err<min_err,:]
        result[err<min_err,0] = i + 1
        min_err[err<min_err] = err[err<min_err]
    
    if not return_error:
        return result
    else:
        return result, min_err

def get_decompression_matrix(a, d=-1):
    """
    Calculate the decompression matrix for a given basis.
    
    Parameters
    ----------
    a : ndarray
        The array containing the vectors which form a basis (i.e. the
        transpose of the matrix of such a basis), or a set of such arrays.
    d : int, optional
        The dimension from which the compression takes place. Defaults to
        -1, which indicates the entire basis is used.
    
    Returns
    -------
    array
        A 2D compression matrix, or a set of such matrices.
    
    Notes
    -----
        See 'get_compression_matrix' for a better explanation
    """
    if len(np.shape(a)) < 3:
        if d > 0: a = a[:d,:]
        return np.transpose(a)
    
    if d > 0: a = a[:,:d,:]
    return np.transpose(a, axes=(0,2,1))

def decompress_lesion(a,d):
    """
    Reverse operation of compress_lesion.
    
    Parameters
    ----------
    a : array
        A compressed vectorized lesion, or a set of such compressed lesions.
    d : array
        A decompression matrix.
    
    Returns
    -------
    array
        An array of decompressed vectorized lesions.
    """
    return np.transpose(d @ np.transpose(a))

def decompress_lesion_local(a, d, s):
    """
    Reverse operation of compress_lesion_local.
    
    Parameters
    ----------
        a : array-like
            (Locally!) compressed vectorized lesion, or set of such compressed lesion.
        d : array-like
            Sequence of decompression matrices.
        s : array-like
            Array of support vectors for the basis on which compressions are based.
    
    Returns
    -------
        array-like
            The decompressed lesion, or an array of decompressed lesion.
    """
    
    # The case for one lesion is simplest
    if len(np.shape(a)) < 2:
        return s[a[0]] + decompress_lesion(a[1:], d[a[0]])
    
    # Initialize result array
    result = np.zeros((len(a), np.shape(d)[1]))
    
    # Calculate results
    #a[:,0] = np.floor(a[:,0])
    for i,lesion in enumerate(a):
        j = int(lesion[0])
        result[i] = s[j] + decompress_lesion(lesion[1:], d[j])
    
    # Return results
    return result

def compress_decompress(a, c, d, dim=-1, verbose=DEBUG_INFO):
    """
    Go through the compression-decompression pipeline. It returns the
    reconstructed lesion(s), as well as the relative error(s) and compression
    coefficients.
    
    Parameters
    ----------
    a : array-like
        A vectorized lesion, or a set of vectorized lesions.
    c : array-like
        The compression matrix.
    d : array-like
        The decompression matrix.
    dim : int, optional
        The desired dimension to which the input vectors are compressed based
        on truncation of the given basis. Defaults to -1, in which case no
        truncation takes place.
    verbose : bool, optional
        If True, verbosity reports are printed. Defaults to False.
    
    Returns
    -------
    array-like
        Reconstructed lesion(s), after the pipeline
    array-like
        Relative errors between the original lesions and the reconstructed
        ones.
    array-like
        List of compression coefficients of the vectors, i.e. the
        compressed lesions.
    """
    
    # Print debug info
    verbose = verbose and len(np.shape(a)) >= 2
    tic = print_report(verbose, 
                       "Executing compression pipeline for " 
                       + str(len(a)) + " lesions...")
    
    # Get the compression and decompression matrices
    if dim<=0: dim=len(c)
    comp_mat = c[:dim,:]
    decomp_mat = d[:,:dim]
    
    # Calculate results
    result = np.transpose(decomp_mat @ comp_mat @ np.transpose(a))
    
    # Debug info
    print_report(verbose, "Compression pipeline finished!", tic)
    
    # Return results
    return result

def compress_decompress_local(a, c, d, s=None, verbose=DEBUG_INFO):
    """
    Go through the local compression-decompression pipeline. It returns the
    reconstructed lesion(s), as well as the relative error(s) and compression
    coefficients.
    
    Parameters
    ----------
    a : array-like
        A vectorized lesion, or a set of vectorized lesions.
    c : sequence of array-like
        A list of compression matrices.
    d : sequence of array-like
        A list of decompression matrices.
    s : array-like or None, optional
        The support vectors for the bases. Defaults to None, which sets
        it to zeros.
    verbose : bool, optional
        If True, verbosity reports are printed. Defaults to False.
    
    Returns
    -------
    array-like
        Reconstructed lesion(s), after the pipeline
    array-like
        Relative errors between the original lesions and the reconstructed
        ones.
    array-like
        List of compression coefficients of the vectors, i.e. the
        compressed lesions.
    """
    
    # Get the compression and decompression matrices
    comp_mat = c
    decomp_mat = d
    
    if s is None:
        s = np.zeros(len(comp_mat))
    
    # If there is only one lesion, calculate and return the results
    if len(np.shape(a)) == 1:
        return decompress_lesion_local(
            compress_lesion_local(a, comp_mat, decomp_mat, s, verbose=False), 
            decomp_mat, s
            )
    
    # Print debug info
    tic = print_report(verbose, "Executing LOCAL compression pipeline for " + str(len(a)) + " lesions...")
    
    # Calculate results
    result = decompress_lesion_local(
        compress_lesion_local(a, comp_mat, decomp_mat, s, verbose=False), 
        decomp_mat, s)
    
    # Debug info
    print_report(verbose, "Compression pipeline finished!", tic)
    
    # Return results
    return result
 
#-----------------------------------------------------------------------------#
# Functions for bootstrapping                                                 #
#-----------------------------------------------------------------------------#   

def calc_coefficient_distribution(a, local = False):
    """
    Calculate the coefficient distribution variables.
    
    Parameters
    ----------
    a : array-like
        Set of compression coefficients.
    local : bool, optional
        If True, the compression coefficients are treated as locally
        compressed. Then, the first value of each coefficient vector (which is
        its patch index) is ignored, and the returned averages and covariance
        matrices are calculated per patch index.
    
    Returns
    -------
    array
        The average values of the coefficients. If `local` is True, this is
        a list of such arrays of averages (one for each patch).
    array
        The covariance matrix of the coefficients. If `local` is True, this
        is a list of such matrices (one for each patch).
    tuple of float, optional
        If `local` is True, an additional tuple is returned, containing
        the average and variance of the patch index parameter.
    """
    
    # If not local, simply return the result values
    if not local:
        return np.average(a, 0), np.cov(np.transpose(a))
    
    # Initialize the result values (one distribution for each patch)
    d = len(a[0]) - 1 # The dimension, without indexing value
    indices = np.array(np.unique(a[:,0]), dtype=int) # The unique patch indices
    avg = np.zeros((len(indices), d))
    cov = np.zeros((len(indices), d, d))
    
    # Calculate the results
    for i in indices:
        masked = a[(a[:,0] == i)][:,1:]
        
        if len(masked.shape) >= 1:
            avg[i,:] = np.average(masked, axis=0)
            cov[i,:,:] = np.cov(np.transpose(masked))  
        #elif len(masked.shape) == 1:
            #avg[i,:] = np.copy(masked)
    
    add = (np.average(a[:,0]), np.sqrt(np.var(a[:,0])))
    
    # Return results
    return avg, cov, add

def generate_lesion(b, a, c, amount = 1):
    """
    Generate new lesions based on coefficient distribution and a basis found
    using general SVD
    
    Parameters
    ----------
    b : array-like
        A basis
    a : array-like
        Averages of coefficients for the given basis.
    c : array-like
        The covariance matrix for the coefficients of given basis.
    amount : int, optional
        The amount of lesions that must be generated. Defaults to 1.
    
    Returns
    -------
    array
        A (vectorized) lesion, or a set of lesions.
    """
    
    # Initialize result array
    result = np.zeros((amount, b.shape[-1]))
    
    for i in range(amount):
        coefficients = np.random.multivariate_normal(a, c)
        result[i] = decompress_lesion(coefficients, get_decompression_matrix(b))
    
    #if amount == 1:
    #    np.flatten(result)
    
    return result    
    

def generate_lesion_local(b, a, c, d, s, amount = 1):
    """
    Generate new lesions based on coefficient distribution and a bases found
    using local SVD
    
    Parameters
    ----------
    b : array-like
        The bases.
    a : array-like
        Averages of coefficients for the given bases.
    c : array-like
        The covariance matrices for the given bases.
    d : tuple of float
        The distribution (average, variance squared) of the patch index vector.
    s : array-like
        The array of support vectors for each plane.
    amount : int, optional
        The amount of lesions that must be generated. Defaults to 1.
    
    Returns
    -------
    array
        A (vectorized) lesion, or a set of lesions.
    """
    
    # Initialize result array
    result = np.zeros((amount, np.shape(b)[-1]))
    
    # For local SVD, the apropriate patch must be chosen as well
    for i in range(amount):
        patch = int(np.clip(np.random.normal(d[0], d[1]), 0, len(a) - 1))
        
        coefficients = np.random.multivariate_normal(a[patch], c[patch])
        decomp = get_decompression_matrix(b[patch])
        result[i] = decompress_lesion(coefficients, decomp) + s[patch]
    
    #if amount == 1:
    #    np.transpose(result)
    
    return result
    

#-----------------------------------------------------------------------------#
# Functions for visualization                                                 #
#-----------------------------------------------------------------------------#

def plot_singular_values(S, cutoffs = None, labels = None, ax=None,
                         verbose = DEBUG_INFO):
    """
    Plot the singular values into a figure / two graphs.
    
    Parameters
    ----------
    S : array-like
        The singular values to be plotted. May be a sequence of singular values.
    cutoffs : array-like or None, optional
        At which indices the cutoffpoints are drawn. If None, no cutoff lines
        are drawn. Defaults to None.
    labels : array-like or None, optional
        The labels of the cutoff indices to draw. If None, no labels and legend
        are given. Defaults to None.
    ax : array [0 .. 1] of plt axis, or None, optional
        The axes onto which the pltos are drawn, or None, in which case a
        new figure is created and returned. Defaults to None.
    verbose : bool, optional
        If True, verbosity reports are printed. Defaults to DEBUG_INFO.
        
    Return
    ------
    figure, optional
        The figure that is drawn. Is not returned if `ax` is not None.
    axes
        The axes that the plots are drawn to. Either one or two, with the
        second one being the optional energy plot.
    """
    
    # Print debug info
    tic = print_report(verbose, "Plotting singular values...")
    
    # Initialize figure and axes
    f = None
    if ax is None:
        f, ax = plt.subplots(1, 2, figsize=(20, 5))
    
    # Get right data
    S_energy = calc_SV_energy(S)
    
    # Set right ax properties
    ax[0].set_title("Singular values")
    if cutoffs is not None:
        ax[0].set_title("Singular values and threshold cutoffs")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Singular values (log)")
    ax[0].set_xlabel("Eigenlesions")
    ax[0].set_xlim([0, len(S)])
    
    ax[1].set_title("Cumulative energy of singular values")
    if cutoffs is not None:
        ax[1].set_title("Cumulative energy of singular values and threshold cutoffs")
    ax[1].set_ylabel("Cumulative energy")
    ax[1].set_xlabel("Eigenlesions")
    ax[1].set_xlim([0, len(S)])
    ax[1].set_ylim([0, 1.05])
    
    # Plot singular values
    ax[0].plot(S, marker="o", linestyle="-", label="Singular values")
    ax[1].plot(S_energy, marker="o", linestyle="-", label="Cumulative energy of singular values")
    
    # Add cutoff lines
    if cutoffs is not None:
        label = ""    
        for i,cutoff_index in enumerate(cutoffs):
            if labels is not None:
                label = labels[i]
            
            S_cutoff = S[cutoff_index]
            cutline, = ax[0].plot([0, cutoff_index], [S_cutoff,S_cutoff], linestyle="--", 
                       label=label)
            color = cutline.get_color() # Get proper color
            ax[0].plot([cutoff_index, cutoff_index], [0,S_cutoff], linestyle="--", color=color)
            ax[0].plot([cutoff_index, cutoff_index], [S_cutoff,S_cutoff], marker="o", color=color)
            
            S_cutoff = S_energy[cutoff_index]
            ax[1].plot([0, cutoff_index], [S_cutoff,S_cutoff], linestyle="--", color=color,
                       label=label)
            ax[1].plot([cutoff_index, cutoff_index], [0,S_cutoff], linestyle="--", color=color)
            ax[1].plot([cutoff_index, cutoff_index], [S_cutoff,S_cutoff], marker="o", color=color)
    

    # Add legend if necessary
    if labels is not None:
        ax[0].legend()
        ax[1].legend()
    
    # Print debug info
    print_report(verbose, "Plotting singular values finished!", tic)
    
    # Return figure
    if f is not None:
        return f,ax
    return ax

def plot_similarities(S, c):
    """
    POTENTIALLY DEPRECATED?
    """ 

    # Sort S and the cutoff index
    #S.sort()
    S_cumsum = calc_SV_energy(S)
    cutoff_index = calc_SV_cutoff(S_cumsum, [1 - c])[0]
    cutoff_value = S[cutoff_index]
    
    # Initialize plots
    f, ax = plt.subplots(1, 2, figsize=(20, 5))
    
    # Plot stuff
    ax[0].bar(np.arange(len(S)), S, label="Similarities of lesions")
    ax[0].plot([0, len(S) - 1], [cutoff_value, cutoff_value], "r--", label="Cutoff for " + "{:10.2f}".format(c * 100) + "%")
    ax[0].set_title("Maximal lesion similarities")
    ax[0].set_xlabel("Lesions")
    ax[0].set_ylabel("Similarity")
    
    S_cutoff = S_cumsum[cutoff_index]
    ax[1].plot(np.arange(len(S_cumsum)), S_cumsum, label="Relative cumulative sum of similarities")
    ax[1].plot([0, cutoff_index], [S_cutoff,S_cutoff], "r--",
               label="Cutoff for " + "{:10.1f}".format(c * 100) + "%")
    ax[1].plot([cutoff_index, cutoff_index], [0,S_cutoff], "r--")
    ax[1].plot([cutoff_index, cutoff_index], [S_cutoff,S_cutoff], "ro")
    ax[1].set_title("Relative cumulative sum of maximal lesion similarities")
    ax[1].set_xlabel("Lesions")
    ax[1].set_ylabel("Rel.cum. similarity")

    ax[0].legend(loc="lower left")
    ax[1].legend(loc="lower left")
    plt.suptitle("Maximal similarities, compared to cutoff")
    plt.show()

def plot_errors(a, labels=None, err_type_abs=False, max_err = 1, ax=None, 
                verbose=DEBUG_INFO):
    """
    Plot the relative errors, relative error frequencies, and error probability
    density functions of the compression pipeline.
    
    Parameters
    ----------
    a : array-like
        An array of relative errors for various reconstructions, or a list of
        such arrays.
    labels : array of str or None, optional
        A list of string-identifiers for the legends of the plots. Defaults to
        None, in which case no legend is plotted.
    err_type_abs : bool, optional
        If True, the errors are denoted as absolute rather than relative on
        the axes. Defaults to False.
    max_err : float, optional
        The maximal error for plotting the x- and y-scales. Defaults to 1.
    ax : array [0 .. 2] of plt axis, or None, optional
        The axes onto which the pltos are drawn, or None, in which case a
        new figure is created and returned. Defaults to None.
    verbose : bool, optional
        If True, print verbosity reports. Defaults to False.
    
    Returns
    -------
    figure, optional
        The matplotlib-figure. Is not returned if ax is given.
    array of axis
        The matplotlib-axes.
    """
    
    # Print debug info
    tic = print_report(verbose, "Plotting errors...")
    
    # Make sure figure is initialized
    f = None
    if ax is None:
        f, ax = plt.subplots(1, 3, figsize=(30, 5))
    
    # Set axis values
    errtype_name = "relative"
    if err_type_abs:
        errtype_name = "absolute"
    
    ax[0].set_xlabel("Lesions")
    ax[0].set_ylabel(errtype_name.title() + " error")
    ax[0].set_title(errtype_name.title() + " error per lesion")
    
    ax[1].set_xlabel(errtype_name.title() + " error")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Cumulative frequency of " + errtype_name + " errors")
    
    # This one is only scaled for relative errors
    ax[2].set_xlabel(errtype_name.title() + " error (x10)")
    if max_err >= 2:
        ax[2].set_xlabel(errtype_name.title() + " error")
    ax[2].set_ylabel("Probability")
    ax[2].set_title("Probability density of "+ errtype_name +" errors, with normal fits")
    
    # Plot the errors of each lesion
    err_shape = np.shape(a)[1]
    colors = []
    
    bar_base = np.zeros(err_shape)
    for i,err in enumerate(a):
        lbl=""
        if labels is not None:
            lbl = labels[i]
        
        container = ax[0].bar(np.arange(len(err)), err, label=lbl, bottom=bar_base)
        bar_base += err
        colors.append(container[-1].get_facecolor())
        
    # Transpose a for easier use in plot functions
    a = np.transpose(a)
        
    # Plot the cumulative frequencies
    bin_count = int(np.floor(err_shape/10))
    ax[1].hist(a, bin_count, range=(0,max_err), label=labels, stacked=True, 
               histtype='bar', color=colors)
    
    # Plot the PDFs
    fac = 10
    if max_err >= 2:
        fac = 1
        
    a = np.transpose(a)
    for i,err in enumerate(a):
        count, bins = np.histogram(err * fac, bin_count, range=(0,max_err * fac), 
                                   density=True)[:2]
        ax[2].plot(bins, np.insert(count,0,0), label=labels[i], color=colors[i])
    # Commented out unfilled hist, because this looked worse
    #ax[2].hist(a*fac, bin_count, range=(0,max_err*fac), 
    #                                label=labels, density=True, fill=False, 
    #                                histtype='step', color=colors)
    

    # Normal fit
    for i,err in enumerate(a):
        mu=np.average(err * fac)
        sigma=np.sqrt(np.var(err * fac))
        
        X = np.linspace(0, max_err * fac, 100)
        ax[2].plot(X, stats.norm.pdf(X,mu,sigma), color=colors[i], 
                   linestyle="--")

    # Finalize plot
    if labels is not None:
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
    
    #ax[0].set_xlim(0, np.shape(errors)[1])
    ax[0].set_ylim(0, len(a) * max_err)
    ax[1].set_xlim(0, max_err)
    ax[2].set_xlim(0, max_err * fac)
    ax[2].set_ylim(0, 1)
    
    # Print debug info
    print_report(verbose, "Plotting errors finished!", tic)
    
    # Return figure
    if f is not None:
        return f, ax
    return ax

def plot_lesion(lesion, ax=None, margin=0.1, new_shape=None):
    """
    Visualize a lesion and using a scatter plot, mapping intensity to colors 
    with a colorbar.
    # Taken from ... #
    
    Parameters: 
        - lesion: The lesion that is to be compared.
        - ax: axis onto which the lesion is plotted.
        - margin: margin to prevent drawing 0's.
        - new_shape: if not None, reshapes the lesion to this shape
    """
    
    if new_shape is not None:
        lesion=vector_to_inmat(lesion, new_shape)
    
    if ax is None: 
        fig = plt.figure(figsize=(30, 7))
        ax = fig.add_subplot(111, projection='3d')
    
    # Get the coordinates of non-zero values
    coords = np.argwhere(lesion > margin)
    intensities = lesion[lesion > margin]
    
    # Create a scatter plot with intensity values mapped to color
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=intensities, cmap='rainbow', alpha=0.8, vmin=0, vmax=1)

    # Fix colorbar range to [0, 1] for consistent coloring
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Intensity')
    
    # Adjust the scale and view
    ax.set_xlim(0, lesion.shape[0])
    ax.set_ylim(0, lesion.shape[1])
    ax.set_zlim(0, lesion.shape[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
def plot_lesion_comparison(lesionA, lesionB, ax=None, margin=0.1, 
                           new_shape=None):
    """
    Visualize both an original lesion and its reconstructed counterpart using a
    scatter plot, mapping intensity to colors with a colorbar.
    
    Parameters: 
        - lesion: The lesion that is to be compared.
        - basis: The basis used for the compression pipeline.
    """
    
    if new_shape is not None:
        lesionA=vector_to_inmat(lesionA, new_shape)
        lesionB=vector_to_inmat(lesionB, new_shape)
    
    # Set up the 3D plots
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(30, 7))
        ax_org = fig.add_subplot(131, projection='3d')
        ax_rec = fig.add_subplot(132, projection='3d')
        ax_dif = fig.add_subplot(133, projection='3d')
    else:
        ax_org = ax[0]
        ax_rec = ax[1]
        ax_dif = ax[2]

    # Plot the lesions
    plot_lesion(lesionA, ax_org)
    plot_lesion(lesionB, ax_rec)
    
    # Half the margin for the difference lesion! To make it fair
    plot_lesion(np.abs(lesionA - lesionB), ax_dif, margin=margin/2)

    # Set the apropriate titles
    ax_org.set_title("Original lesion")
    ax_rec.set_title("Reconstructed lesion")
    ax_dif.set_title("Difference")
    #plt.suptitle("A reconstruction comparison for the compression pipeline (to " + str(len(basis)) + " dimensions), relative error = " + str(err))

    if fig is None:
        return ax
    return fig, ax

#-----------------------------------------------------------------------------#
# Actual execution                                                            #
#-----------------------------------------------------------------------------#

# Parameters lesions
shuffle_lesions = False # Impacts choice of training vs validation data
training_lesion_perc = 80 # Setting lower than 50% may give problems, as data is def. globally <500-dim.

# Parameters similarity cut
simret = 0.95 # Percentual similarity retention

# Parameters general compression
desired_dims = [100, 60, 20]

# Parameters local SVD
max_err = 0.0001
max_err_quantile = 0.5
max_err_abs = True

# Parameters global SVD
thresholds = []
plot_sv_graph = True

# Parameters for error plotting
use_abs_error = True




# Import lesions
Lesions = import_lesions(BASE_PATH, filename_extension='.nii.gz')[0]
avg_lesion = np.average(Lesions, axis=0)
Lesions = Lesions - avg_lesion # Center around 0
if shuffle_lesions:
    np.random.shuffle(Lesions)

# Execute similarity calculations
similarities = calc_similarity(Lesions)

plot_similarities(similarities, 1 - simret)
lesions_all = np.copy(Lesions)
simcut = np.quantile(similarities, simret)
lesions_cut = np.copy(Lesions[np.where(similarities <= simcut)])
print("Simcut results in " + str(len(Lesions[np.where(similarities <= simcut)])) + " retained lesions!")

title_adder = ", and with a similarity cut of " + str(100 - simret * 100) + "% of the lesions"
for Lesions in [lesions_cut, lesions_all]:
    # Pre-calculate groups and lengths
    training_lesion_count = int(np.floor(len(Lesions) * training_lesion_perc/100))
    training_lesions = Lesions[:training_lesion_count ]
    validation_lesions = Lesions[training_lesion_count:]
    
    lengths = np.linalg.norm(Lesions, axis=1)
    
    max_err_for_plotting = 1.0
    if use_abs_error:
        max_err_for_plotting = np.max(lengths)/2 # Div by 2 is imperical
    
    if use_abs_error:
        lengths = np.ones(len(Lesions))
    lengths_training = lengths[:training_lesion_count]
    lengths_validation = lengths[training_lesion_count:]
    
    # Get labels
    labels=[]
    for d in desired_dims:
        labels.append("Dimension=" + str(d))
        
        
    # qFC-clustering into LSVD
    errors_training = []
    errors_validation = []
    K = []
    for d in desired_dims:
        # Calculate basis
        res = calc_SVD_basis_local(training_lesions, d-1, max_err=max_err,
                                   max_err_abs=max_err_abs, 
                                   max_err_quantile=max_err_quantile, 
                                   init_flat_count = 5)
        
        bases = res[0]
        supps = res[1]
        K.append(len(bases))
        
        # Calculate error  
        rec = compress_decompress_local(training_lesions, 
                                    get_compression_matrix(bases),
                                    get_decompression_matrix(bases),
                                    supps, verbose=False)
                                    
        err = np.linalg.norm(training_lesions-rec, axis=1)/lengths_training
        errors_training.append(err)
        
        
        rec = compress_decompress_local(validation_lesions, 
                                    get_compression_matrix(bases),
                                    get_decompression_matrix(bases),
                                    supps, verbose=False)
                                    
        err = np.linalg.norm(validation_lesions-rec, axis=1)/lengths_validation
        errors_validation.append(err)
    
    # Plot errors
    labels_old = labels.copy()
    
    for i,l in enumerate(labels):
        labels[i] = l + ", k=" + str(K[i])
    
    plot_errors(np.array(errors_training), labels, err_type_abs=use_abs_error, 
                max_err=max_err_for_plotting)
    plt.suptitle("Compression errors for training data of qFC local SVD" + title_adder)
    plt.show()
    
    plot_errors(np.array(errors_validation), labels, err_type_abs=use_abs_error, 
                max_err=max_err_for_plotting)
    plt.suptitle("Compression errors for validation data of qFC local SVD" + title_adder)
    plt.show()
    
    labels = labels_old
    
    
    
    # Execute trivial local SVD
    errors_training = []
    errors_validation = []
    for d in desired_dims:
        # Calculate basis
        res = calc_SVD_basis_local_trivial(training_lesions, d - 1)
        
        bases = res[0]
        supps = res[1]
        
        # Calculate error
        rec = compress_decompress_local(validation_lesions, 
                                    get_compression_matrix(bases),
                                    get_decompression_matrix(bases),
                                    supps, verbose=False)
                                    
        err = np.linalg.norm(validation_lesions-rec, axis=1)/lengths_validation
        errors_validation.append(err)
    
    # Plot errors
    plot_errors(np.array(errors_validation), labels, err_type_abs=use_abs_error, 
                max_err=max_err_for_plotting)
    plt.suptitle("Compression errors for validation data of trivial local SVD" + title_adder)
    plt.show()
    
    
    
    # Execute global SVD
    B,S = calc_SVD_basis(training_lesions, return_svals=True)
    S_energy = calc_SV_energy(S)
    
    # Fix legend labels
    labels=[]
    thresholds=[] # quick fix
    for t in thresholds:
        labels.append("Energy cutoff at " + str(t * 100) + "%")
    for d in desired_dims:
        thresholds.append(S_energy[d]) # This will add cutoffs easily
        labels.append("Dimension=" + str(d))
        
    # Calculate cutoffs
    S_cutoffs = calc_SV_cutoff(S_energy, thresholds)
    
    # Calculate reconstruction errors
    errors_training = []
    errors_validation = []
    for i,c in enumerate(S_cutoffs):
        rec = compress_decompress(training_lesions, get_compression_matrix(B, c), get_decompression_matrix(B, c), verbose=False)
        err = np.linalg.norm(training_lesions-rec, axis=1)/lengths_training
        errors_training.append(err)
        
        rec = compress_decompress(validation_lesions, get_compression_matrix(B, c), get_decompression_matrix(B, c), verbose=False)
        err = np.linalg.norm(validation_lesions-rec, axis=1)/lengths_validation
        errors_validation.append(err)
    
    # Plot singular values & errors
    if plot_sv_graph:
        plot_singular_values(S, S_cutoffs, labels=labels)
    
    # Plot errors
    plot_errors(np.array(errors_training), labels, err_type_abs=use_abs_error, 
                max_err=max_err_for_plotting)
    plt.suptitle("Compression errors (training) for general SVD" + title_adder)
    plt.show()
    
    plot_errors(np.array(errors_validation), labels, err_type_abs=use_abs_error, 
                max_err=max_err_for_plotting)
    plt.suptitle("Compression errors (validation) for general SVD" + title_adder)
    plt.show()
    
    title_adder = ""
    
    #exit()
    
 
    
# Calculate coefficient distributions (based on the SVD)
cof_avg = np.zeros((len(S_cutoffs), len(Lesions)))
cof_cov = np.zeros((len(S_cutoffs), len(Lesions), len(Lesions)))
for i,c in enumerate(S_cutoffs):
    Coefficients = compress_lesion(lesions_all, get_compression_matrix(B, c))
    mu,sig = calc_coefficient_distribution(Coefficients)
    cof_avg[i,:c] = mu
    cof_cov[i,:c,:c] = sig
 
# Get and plot new lesions
for i,c in enumerate(S_cutoffs):
    new_lesion = generate_lesion(B[:c], cof_avg[i,:c], cof_cov[i,:c,:c]) + avg_lesion
    
    fig = plt.figure(figsize=(30, 7))
    ax = fig.add_subplot(111, projection='3d')
    plot_lesion(new_lesion, ax, new_shape=(10,10,10))
    plt.title("New random lesion, generated from dimension " + str(c))
    plt.show()

# Plot reconstructions
#reconstructions = np.random.randint(len(Lesions), size=2)
#for i,c in enumerate(S_cutoffs):
#    for r in reconstructions:
#        plot_pipeline_comparison(Lesions[r], np.transpose(B[:c,:]), r)
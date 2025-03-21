from networkx import k_nearest_neighbors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_swiss_roll

import nibabel as nib
path = "../MockLesions/Mock_lesion_0.nii"
nii_image = nib.load(path)
image_array = nii_image.get_fdata()

from Pipeline import twoNN

def get_mock_lesion_n(n):
    path = "../MockLesions/Mock_lesion_"+str(n)+".nii"
    nii_image = nib.load(path)
    image_array = np.array(nii_image.get_fdata())
    return image_array

def transform_lesion_to_1d_array(array3d):
    image_vector = np.reshape(array3d,(1000,))
    return image_vector


def n_mock_lesions(n):
    all_mock_lesions = []
    for i in range(n): 
        lesion_i = get_mock_lesion_n(i)
        lesion_i_1d = transform_lesion_to_1d_array(lesion_i)
        all_mock_lesions.append(lesion_i_1d) 
    
    result = np.vstack(all_mock_lesions)
    return result 


def transform_3d_array_to_2d_array(array3d):
    array2d = []
    for i in range(len(array3d)):
        for j in range(len(array3d[0])):
            for k in range(len(array3d[0][0])):
                if array3d[i][j][k] != 0:
                    array2d.append([i,j,k,array3d[i][j][k]])
    return array2d

def testTwoNN(array):
    d_est = twoNN.twoNearestNeighbors(affinity="nearest_neighbors")
    d_est_fit = d_est.fit(array)
    dim = d_est_fit.dim_
    x = d_est_fit.x_
    y = d_est_fit.y_

    plt.plot(x, y, 'ro')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

    print(dim)

def getTwoNN(array):
    d_est = twoNN.twoNearestNeighbors(affinity="nearest_neighbors")
    d_est_fit = d_est.fit(array)
    dim = d_est_fit.dim_
    return dim


def getTwoNN_dim(array):
    d_est = twoNN.twoNearestNeighbors(affinity="nearest_neighbors")
    d_est_fit = d_est.fit(array)
    dim = d_est_fit.dim_
    return dim

def testTwoNN_Mock_Lesion(n):
    lesion_array = get_mock_lesion_n(n)
    lesion_array2d = transform_3d_array_to_2d_array(lesion_array)
    testTwoNN(lesion_array2d)
    
def getTwoNN_Mock_Lesion(n):
    lesion_array = get_mock_lesion_n(n)
    lesion_array2d = transform_3d_array_to_2d_array(lesion_array)
    return getTwoNN(lesion_array2d)



def TwoNN_dim_all_n():
    mock_lesions = [transform_lesion_to_1d_array(get_mock_lesion_n(n)) for n in range(1000)]
    dims = []
    for i in range(3,999):
        dim_i = getTwoNN_dim(mock_lesions[:i])
        dims.append(dim_i)
    return dims

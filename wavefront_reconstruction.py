import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from fast_zernike import j_to_mn, zernike_derivative_cartesian

def normalize_coordinates(coord_array):
    """
        Given an nx2 numpy array of of (x,y) coordinates in pixels, apply 
        a coordinate transformation so that the mean of the coordinates is 
        at (0,0) and all pixels are within the unit circle. This function
        returns the center of the coordinates and the amount by which the
        coordinates were scaled, so that the original coordinates may be 
        retrieved from the returns as
        coord_array = coords_transformed*coord_scale + coord_mean
    """
    coord_mean = np.mean(coord_array,axis=0)
    coords_centered = coord_array - coord_mean
    coord_scale = np.max(np.linalg.norm(coords_centered,axis=1))
    coords_transformed = coords_centered / coord_scale 
    return coords_transformed, coord_mean, coord_scale

def find_spot_centers(img):
    """
        Returns a numpy array of indices (coordinates in the pixel-based 
        coordinate system of an image obtained from the Shack-Hartmann 
        sensor) denoting the center of each Shack-Hartmann spot.
    """
    intensity_threshold = 50
    ret, img_bw = cv.threshold(img, intensity_threshold, 255, 0)
    contours, hierarchy = cv.findContours(img_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    spot_centers = np.zeros((len(contours),2))
    for i, contour in enumerate(contours):
        spot_centers[i,:] = np.mean(contour,axis=0)[0].astype(int)
    # openCV uses coordinate system [horizontal_coord, vertical_coord] 
    # (standard spatial x,y coordinates) in place of the array index coordinates 
    # [vertical_coord,horizontal coord] so we swap the coordinates to ensure consistency
    spot_centers[:,[0, 1]] = spot_centers[:, [1, 0]]
    return spot_centers

def min_distance_between_spots(spot_centers):
    """
        For a given Shack-Hartmann pattern, this function returns the
        separation between spots in the Shack-Hartmann pattern. This is
        to be used in deciding the appropriate size of detection bounding
        boxes for spot deviations in a Shack-Hartmann pattern.
    """
    return min(np.linalg.norm(np.diff(spot_centers,axis=0),axis=1).astype(int))

def compute_SH_diffs(spot_centers, img_aberrated):
    """
    INPUTS:
        spot_centers  : centers of spots in the unabberated Shack-Hartmann pattern
        img_aberrated : image of Shack-Hartmann pattern for aberrated image
    RETURNS:
        diffs         : an array of differences between the spots in `spot_centers` 
                        and the centroids of the aberrated spots
    """
    diffs = np.empty(spot_centers.shape)
    
    intensity_threshold = 50
    ret, img_bw_aberrated = cv.threshold(img_aberrated, intensity_threshold, 255, 0)

    box_width = (min_distance_between_spots(spot_centers) // 2).astype(int)
    for index,spot in enumerate(spot_centers):
        box_x_min = int(spot[0] - box_width)
        box_x_max = int(spot[0] + box_width)
        box_y_min = int(spot[1] - box_width)
        box_y_max = int(spot[1] + box_width)
        box_aberrated = img_bw_aberrated[box_x_min:box_x_max, box_y_min:box_y_max].astype('uint8')
        contours, hierarchy = cv.findContours(box_aberrated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            print("Warning: multiple spots detected in same bounding box of Shack-Hartmann pattern. Automatically choosing the first spot.")
            #print("Num spots detected: ", len(contours))
        elif len(contours) == 0:
            print("Warning: no spot detected in bounding box of Shack-Hartmann pattern. Assigning None values to difference")
            diffs[index] = np.array([None,None])
            continue
        contour = contours[0]
        spot_aberrated = (np.mean(contour,axis=0) + np.array([box_x_min,box_y_min])).astype(int)
        diffs[index] = spot - spot_aberrated
        #print("Spot: ", spot, "   aberr: ", spot_aberrated)
    return diffs.astype(int)

def B_matrix(spot_centers, num_zernike_modes):
    """
        If s is the vectorized vector of spot centers and z is a vector of
        Zernike coefficients, this function returns B such that s = B*z.
    """
    num_spots = spot_centers.shape[0]
    B = np.empty((2*num_spots, num_zernike_modes))
    print(B)
    for (row_index,zern_index),_ in np.ndenumerate(B):
        spot_index = row_index // 2
        m,n = j_to_mn(zern_index) 
        spot = spot_centers[spot_index]
        x,y = spot
        if row_index % 2 == 0:
            derivative_variable = "x"
        elif row_index % 2 == 1:
            derivative_variable = "y"
        B[row_index,zern_index] = zernike_derivative_cartesian(m, n, x, y, derivative_variable)
    return B

def estimate_wavefront_zernike(spot_centers, img_aberrated, num_zernike_modes):
    """ 
        Returns vector of zernike coefficients. First coefficient (position zero) 
        is the piston mode. The ordering used is 
    """
    num_spots = spot_centers.shape[0]
    spots_normalized, spots_mean, spots_scale = normalize_coordinates(spot_centers)
    diffs = compute_SH_diffs(spot_centers, img_aberrated)
    diffs_normalized = diffs * spots_scale
    diffs_normalized_vectorized = diffs_normalized.reshape((2*num_spots,1))
    B = B_matrix(spots_normalized, num_zernike_modes)
    return np.matmul(np.linalg.pinv(B), diffs_normalized_vectorized)
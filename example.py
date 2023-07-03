#%%
import matplotlib.pyplot as plt
from wavefront_reconstruction import *
from plotting_functions import *

# load SH sensor image
img = np.load("example_sh_image.npy")
img_init = np.load("sh_img_init.npy")
img_aberrated = np.load("sh_img_aberrated.npy")

num_zernike_modes = 36
# Zernike modes will range from 0 to 20 in standard ordering


spot_centers = find_spot_centers(img_init.copy())
num_spots = spot_centers.shape[0]
# The spot_centers array has dimensions nx2, where n is the number of detected
# spots in the initial image. To calculate the B matrix, we require normalized
# cartesian coordinates (i.e., coordinates with mean zere, and all coordinates)
# in the unit circle. To achieve this, we normalize the coordinates.

spot_centers_normalized, spots_mean, spots_scale = normalize_coordinates(spot_centers)

# Here, spots_mean and spots_scale can be used to retrieve the original coordinates
# from the normalized coordinates, so (approximately) we have
#   spot_centers = spot_centers_normalized * spots_scale + spots_mean
# We then use this for the B matrix:

B = B_matrix(spot_centers_normalized,num_zernike_modes)
#B = B_matrix(spot_centers,num_zernike_modes)

# Now, if you have an array of zernike coefficients `z`, then B*z will return the 
# normalized SH deviations (i.e., the actual SH deviations divided by spots_scale)

# The estimate_wavefront_zernike function already normalizes the coordinates 
# inside it, so there is no need to feed it normalized coordinates. Just feed
# it the spot centers returned by the find_spot_centers function, and the
# aberrated image, and the desired number of Zernike modes. The returned array
# `z` is just a vector of Zernike coefficients.

z = estimate_wavefront_zernike_normalized(spot_centers, img_aberrated, num_zernike_modes)

# We compute the SH deviations:
diffs = compute_SH_diffs(spot_centers, img_aberrated)
normalized_diffs = diffs / spots_scale

# With this, we should have
#   normalized_diffs = B*z
# Note that the output of B*z is vectorized (i.e., the two columns for the x 
# and y columns are combined into one column), so we must reshape the output
# first.

errors = normalized_diffs.reshape(2*num_spots,1) - np.matmul(B,z)
#errors = diffs.reshape(2*num_spots,1) - np.matmul(B,z)
# %%

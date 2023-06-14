import matplotlib.pyplot as plt
from wavefront_reconstruction import *
from plotting_functions import *

# load SH sensor image
img = np.load("example_sh_image.npy")
img_init = np.load("sh_img_init.npy")
img_aberrated = np.load("sh_img_aberrated.npy")

spot_centers = find_spot_centers(img_init.copy())
diffs = compute_SH_diffs(spot_centers, img_aberrated.copy())
num_zernike_modes = 21 # zernike modes will range from 0 to 20 in standard ordering
zernike_coeffs = estimate_wavefront_zernike(spot_centers, img_aberrated, num_zernike_modes)

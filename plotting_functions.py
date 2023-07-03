import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from wavefront_reconstruction import *
from zernike import RZern
from fast_zernike import j_to_mn

def plot_bounding_boxes(img):
    spot_centers = find_spot_centers(img)
    spot_centers[:,[0, 1]] = spot_centers[:, [1, 0]]
    box_width = min_distance_between_spots(spot_centers) // 2
    for spot in spot_centers:
        cv.circle(img, tuple(spot.astype(int)), radius=2, color=(0, 0, 255), thickness=1)
        box_corner1 = tuple((spot - np.array([box_width,box_width])).astype(int))
        box_corner2 = tuple((spot + np.array([box_width,box_width])).astype(int))
        cv.rectangle(img, box_corner1, box_corner2, color=(255,0,0))
    plt.imshow(img)

def plot_bounding_boxes_on_aberrated_image(spot_centers,img_aberrated):
    spot_centers[:,[0, 1]] = spot_centers[:, [1, 0]]
    box_width = min_distance_between_spots(spot_centers) // 2
    for spot in spot_centers:
        cv.circle(img_aberrated, tuple(spot.astype(int)), radius=2, color=(0, 0, 255), thickness=1)
        box_corner1 = tuple((spot - np.array([box_width,box_width])).astype(int))
        box_corner2 = tuple((spot + np.array([box_width,box_width])).astype(int))
        cv.rectangle(img_aberrated, box_corner1, box_corner2, color=(255,0,0))
    plt.imshow(img_aberrated)

def plot_wavefront_from_zernike_coeffs(z):
    """
    Args:
        z (numpy vector): zernike coefficients. Ideally, supply z with 
        a triangular number of coefficients (e.g., 1, 3, 6, 10, 15, 21, 28, ...),
        otherwise some coefficients will be ignored. 
    """
    # Setup for plotting reconstructed wavefront
    j = len(z)
    m,n = j_to_mn(j)
    num_coeffs = (n**2 + n)//2
    
    cart = RZern(n-1)
    L, K = 200, 200
    ddx = np.linspace(-1.0, 1.0, K)
    ddy = np.linspace(-1.0, 1.0, L)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)

    Phi = cart.eval_grid(z[0:num_coeffs], matrix=True)
    plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
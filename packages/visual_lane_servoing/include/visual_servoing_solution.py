from typing import Tuple
from matplotlib import pyplot as plt


import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for reactive control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    steer_matrix_left = np.ones(img.shape, dtype=np.uint8)
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    steer_matrix_right = -np.ones(img.shape, dtype=np.uint8)
    # ---
    return steer_matrix_right

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)

    sigma = 4.5

    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    threshold = 45

    mask_mag = (Gmag > threshold)

    image_masekd = mask_mag*Gmag

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    white_lower_hsv = np.array([0, 0, 153])         # CHANGE ME
    white_upper_hsv = np.array([179, 60, 255])   # CHANGE ME
    yellow_lower_hsv = np.array([15, 100, 100])        # CHANGE ME
    yellow_upper_hsv = np.array([30, 255, 255])  # CHANGE ME

    mask_white = cv2.inRange(image_hsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(image_hsv, yellow_lower_hsv, yellow_upper_hsv)

    width = img.shape[1]
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    mask_left_edge =  mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge =  mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white


    return mask_left_edge, mask_right_edge

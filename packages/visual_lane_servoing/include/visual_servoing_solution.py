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
    steer_matrix_left = - np.ones(shape, dtype=np.uint8) # I define a matrix of ones. and assign the negative sign to them
    top_part = int(shape[0] * 0.34) # the top 34% of the matrix is zero to remove distractions (anything above an approximate fixed horizon)
    steer_matrix_left[:top_part, :] = 0

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
    steer_matrix_right = np.ones(shape, dtype=np.uint8) * 0.82 # I added weights to this matrix of ones. I optimized it to balance the left and right matrices.
    top_part = int(shape[0] * 0.34) # the top 34% of the matrix is zero to remove distractions (anything above an approximate fixed horizon)
    steer_matrix_right[:top_part, :] = 0

    # ---
    return steer_matrix_right

    #
    #  Note: To find these parameters I added a temporary subsriber function
    #  into the code of "src/visual_lane_servoing_node.py" to receive the
    # parameters from the user through a ros topic that I defined. This way, I was able
    #  to easily modify parameters like the coefficient of the ones matrix 
    #  in "get_steer_matrix_right_lane_markings()" or the coefficient of steer 
    #  in "src/visual_lane_servoing_node.py". by doing this, I didn't have to rebuild
    #  the code in the robot again for every combination of parameter. I just needed to send 
    # a message thought a publisher to this subscriber of the robot to modify the parameters and 
    # find the optimal combination to do the loop. I did this part in collaboration with Amir 
    # Hossein Zandi in our class. I then removed the ros publisher and receiver after i found the
    #  good parameters.
    #

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

    sigma = 4.4

    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    threshold = 48

    mask_mag = (Gmag > threshold)

    image_masekd = mask_mag*Gmag

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    white_lower_hsv = np.array([0, 0, 205])         # CHANGE ME
    white_upper_hsv = np.array([228, 69, 255])   # CHANGE ME
    yellow_lower_hsv = np.array([15, 30, 100])        # CHANGE ME
    yellow_upper_hsv = np.array([35, 254, 255])  # CHANGE ME

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

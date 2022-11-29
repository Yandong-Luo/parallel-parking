import numpy as np
import cv2
from studentVision import lanenet_detector as ld
from skimage import morphology

img = cv2.imread("test_2.jpg")

def gradient_thresh(img, thresh_min=25, thresh_max=100):
    """
    Apply sobel edge detection on input image in x, y direction
    """
    #1. Convert the image to gray scale
    #2. Gaussian blur the image
    #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
    #4. Use cv2.addWeighted() to combine the results
    #5. Convert each pixel to unint8, then apply threshold to get binary image

    ## TODO

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_GB = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_sobelX = cv2.Sobel(img_GB, cv2.CV_32F, 1, 0, 3)
    img_sobelY = cv2.Sobel(img_GB, cv2.CV_32F, 0, 1, 3)
    img_add = cv2.addWeighted(img_sobelX, 0.5, img_sobelY, 0.5, 0)
    img_uint8 = cv2.convertScaleAbs(img_add)
    grey_output = cv2.threshold(img_uint8, thresh_min, thresh_max, cv2.THRESH_BINARY)[1]
    binary_output = np.zeros_like(grey_output)
    binary_output[grey_output!=0] = 1

    ####

    return binary_output


def color_thresh(img, thresh=(100, 255)):
    """
    Convert RGB to HSL and threshold to binary image using S channel
    """
    #1. Convert the image from RGB to HSL
    #2. Apply threshold on S channel to get binary image
    #Hint: threshold on H to remove green grass

    ## TODO

    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    Hchannel = img_HLS[:, :, 0]
    Lchannel = img_HLS[:, :, 1]
    Schannel = img_HLS[:, : ,2]
    maskH = cv2.inRange(Hchannel, 20, 85)
    maskL = cv2.inRange(Lchannel, 120, 255)
    # maskS = cv2.inRange(Schannel, thresh[0], thresh[1])
    grey_output = cv2.bitwise_and(maskL, maskL, mask=maskH)
    # grey_output = maskH
    binary_output = np.zeros_like(grey_output)
    binary_output[grey_output!=0] = 1

    #### 

    return binary_output

h = img.shape[0]
w = img.shape[1] 

input = np.float32([
                [0.42 * w, 0.55 * h],
                [0.1 * w, 0.96 * h],
                [0.9 * w, 0.96 * h],
                [0.58 * w, 0.55 * h]])

output =  np.float32([
                    [0, 0],
                    [0, h - 1],
                    [w - 1, h - 1],
                    [w - 1, 0]])

M = cv2.getPerspectiveTransform(input, output)
Minv = np.linalg.inv(M)

img = cv2.warpPerspective(img.astype('uint8'), M, (w, h))

cv2.imwrite("test_images/1.png", img)

SobelOutput = gradient_thresh(img, 5, 100)
ColorOutput = color_thresh(img)

####

binaryImage = np.zeros_like(SobelOutput)
# binaryImage[(ColorOutput==1)&(SobelOutput==1)] = 1
# binaryImage[SobelOutput==1] = 1
binaryImage[ColorOutput==1] = 1
# Remove noise from binary image
binaryImage = morphology.remove_small_objects(binaryImage,min_size=50,connectivity=2)

cv2.imwrite("test_images/3.png", (np.dstack((binaryImage, binaryImage, binaryImage))*255).astype('uint8'))

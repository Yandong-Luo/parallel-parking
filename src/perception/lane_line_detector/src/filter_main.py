# This file has NOTHING to do with the rest of the MP
# Please answer the filter question in your report with
# results from running this script
# You still have to do "source devel/setup.bash".
# Eric Liang

import rospkg
import os
import cv2

# Please check the following website for reference
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
def filter_gaussian(input_img):
    img = input_img.copy()
    # TO-DO
    # YOUR CODE HERE
    # img = cv2.GaussianBlur(?)

    # YOUR CODE ENDS HERE
    return img

def filter_median(input_img):
    img = input_img.copy()
    # TO-DO
    # YOUR CODE HERE
    # img = cv2.medianBlur(?)

    # YOUR CODE ENDS HERE
    return img

if __name__ == '__main__':
    # Get path
    rospack = rospkg.RosPack()
    mp1_path = rospack.get_path('mp1')
    sp_img_path = os.path.join(mp1_path, 'images', 'salt_and_pepper.jpg')
    ga_img_path = os.path.join(mp1_path, 'images', 'gaussian.jpg')
    result_sp_img_ga_filter_path = os.path.join(mp1_path, 'images', 'results',
    'salt_and_pepper_gaussian_filter.jpg')
    result_sp_img_me_filter_path = os.path.join(mp1_path, 'images', 'results',
    'salt_and_pepper_median_filter.jpg')
    result_ga_img_ga_filter_path = os.path.join(mp1_path, 'images', 'results',
    'gaussian_noise_gaussian_filter.jpg')
    result_ga_img_me_filter_path = os.path.join(mp1_path, 'images', 'results',
    'gaussian_noise_median_filter.jpg')
    # Read images
    sp_img = cv2.imread(sp_img_path)
    ga_img = cv2.imread(ga_img_path)
    # Run filters
    sp_img_gaussian_filter = filter_gaussian(sp_img)
    sp_img_median_filter = filter_median(sp_img)
    ga_img_gaussian_filter = filter_gaussian(ga_img)
    ga_img_median_filter = filter_median(ga_img)
    # Show images
    cv2.imshow("Salt and Pepper before Filtering", sp_img)
    cv2.imshow("Salt and Pepper after Gaussian Filter", sp_img_gaussian_filter)
    cv2.imshow("Salt and Pepper after Median Filter", sp_img_median_filter)
    cv2.imshow("Gaussian Noise before Filtering", ga_img)
    cv2.imshow("Gaussian Noise after Gaussian Filter", ga_img_gaussian_filter)
    cv2.imshow("Gaussian Noise after Median Filter", ga_img_median_filter)
    # Write images to images/results folder
    cv2.imwrite(result_sp_img_ga_filter_path, sp_img_gaussian_filter)
    cv2.imwrite(result_sp_img_me_filter_path, sp_img_median_filter)
    cv2.imwrite(result_ga_img_ga_filter_path, ga_img_gaussian_filter)
    cv2.imwrite(result_ga_img_me_filter_path, ga_img_median_filter)
    # Pause to show images
    print("Press any key on image windows to quit")
    cv2.waitKey(0)

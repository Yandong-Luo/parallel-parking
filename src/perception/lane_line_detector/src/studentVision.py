#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2022/11/28 09:51:06
@Author      :Yandong Luo
@version      :1.0
'''

import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
from lane_line_detector.msg import line_info, lane_info
# import pyzed.sl as sl


class lanenet_detector():
    def __init__(self):
        self.bridge = CvBridge()
        self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)
        # /zed2/zed_node/point_cloud/cloud_registered
        # self.sub_image = rospy.Subscriber('/zed2/zed_node/point_cloud/cloud_registered', PointCloud2, self.cloud_callback, queue_size=1)
        self.sub_depth_image = rospy.Subscriber('/zed2/zed_node/depth/depth_registered', Image, self.depth_callback, queue_size=1)
        # rosrun image_view extract_images image:=/zed2/zed_node/rgb/image_rect_color
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)

        # 发布线的信息，包括曲率，左右车道线
        self.pub_line = rospy.Publisher("/lane_info",lane_info,queue_size=1)

        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True

        self.m_lane_info = lane_info()

        # self.zed = sl.Camera()
        # self.point_cloud = sl.Mat()

    def img_callback(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)
        self.depth = np.array(depth_image, dtype=np.float32)


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
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

    def color_thresh(self, img, thresh=(100, 255)):
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
        # grey_output = maskL
        binary_output = np.zeros_like(grey_output)
        binary_output[grey_output!=0] = 1

        #### 

        return binary_output

    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        # 1. Apply sobel filter and color filter on input image
        # 2. Combine the outputs
        # Here you can use as many methods as you want.

        SobelOutput = self.gradient_thresh(img, 5, 100)
        ColorOutput = self.color_thresh(img)

        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput==1)&(SobelOutput==1)] = 1
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage,min_size=50,connectivity=2)

        # cv2.imwrite("test_images/25.png", (np.dstack((binaryImage, binaryImage, binaryImage))*255).astype('uint8'))

        return binaryImage

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        # img = cv2.imread("test.jpg")

        ## TODO
        h = img.shape[0]
        w = img.shape[1]
        # print(h,w)

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

        warped_img = cv2.warpPerspective(img.astype('uint8'), M, (w, h))

        # cv2.imwrite("test_images/1.png", (np.dstack((warped_img, warped_img, warped_img))*255))
        # cv2.imwrite("test_images/1.png", img)

        ####

        return warped_img, M, Minv

    def detection(self, img):
        # img = cv2.imread("test.jpg")

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        # print('------')

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds'] 
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    if np.all(ret['left_fit'] != ret['right_fit']):
                        self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None and not (np.all(ret['left_fit'] == ret['right_fit']) and np.abs(ret['left_fit'][1] < 1)):
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False
            
            left_line_info = line_info()
            left_line_info.type = "left"
            left_line_info.fit = left_fit
            # left_line_info.curvature = ret['left_curverad']
            self.m_lane_info.lines.append(left_line_info)

            right_line_info = line_info()
            right_line_info.type = "right"
            right_line_info.fit = right_fit
            # right_line_info.curvature = ret['right_curverad']
            self.m_lane_info.lines.append(right_line_info)

            self.pub_line.publish(self.m_lane_info)

            # print('la:', left_fit[0], 'lb:', left_fit[1], 'lc:', left_fit[2])
            # print('ra:', right_fit[0], 'rb:', right_fit[1], 'rc:', right_fit[2])
            # print('lb:', left_fit[1])
            # print('rb:', right_fit[1])
            # print('la:', ret['left_fit'][0], 'lb:', ret['left_fit'][1], 'lc:', ret['left_fit'][2])
            # print('ra:', ret['right_fit'][0], 'rb:', ret['right_fit'][1], 'rc:', ret['right_fit'][2])
            state = 'Continue'
            if np.abs(left_fit[1]) < 1 or np.abs(right_fit[1]) < 1:
                state = 'Go Ahead'
            elif left_fit[1] >= 1 and right_fit[1] >= 1:
                state = 'Turn Left'
            elif left_fit[1] <= -1 and right_fit[1] <= -1:
                state = 'Turn Right'
            # print(state)
            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            # print(img.shape)
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(state, img, self.depth, left_fit, right_fit, Minv)
            # else:
            #     print("Unable to detect lanes")
                # print('Continue')

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2022/11/24 18:00:48
@Author      :Yandong Luo
@version      :1.0
'''

import rospy
import cv2
import numpy as np
import message_filters
import math
import tf
# from darknet_ros_msgs.msg import BoundingBoxes
from hog_haar_person_detection.msg import Pedestrians,BoundingBox
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from rgb_depth.msg import detector_info, object_info

class RGB_Depth:
    def __init__(self) -> None:
        
        cam_info_topic = rospy.get_param('cam_info_topic','/zed2/zed_node/rgb_raw/camera_info')
        rgb_img_topic = rospy.get_param('rgb_img_topic','/zed2/zed_node/rgb/image_rect_color')
        depth_img_topic = rospy.get_param('depth_info_topic','/zed2/zed_node/depth/depth_registered')
        # darknet_result_topic = rospy.get_param('darknet_result_topic','/darknet_ros/bounding_boxes')
        detect_result_topic = rospy.get_param('detect_result_topic','/person_detection/pedestrians')

        # 相机矩阵的信息
        self.camera_info_sub = message_filters.Subscriber(cam_info_topic, CameraInfo)
        # yolo的检测结果,放弃yolo
        # self.result_sub = message_filters.Subscriber(darknet_result_topic,BoundingBoxes)

        self.result_sub = message_filters.Subscriber(detect_result_topic,Pedestrians)

        # RGB图像,暂时用了kitti的数据集
        self.rgb_img_sub = message_filters.Subscriber(rgb_img_topic,Image)
        # 深度图像
        self.depth_img_sub = message_filters.Subscriber(depth_img_topic,Image)

        # 发布
        self.detector_info_pub = rospy.Publisher('/detector_info',detector_info,queue_size=1)

        # 合并RGB和深度的信息的图像发布
        self.detected_pub = rospy.Publisher('/rgb_depth/detected_image', Image, queue_size=1)
        # 检测且带距离的RGB图像
        self.detected_rgb_pub = rospy.Publisher('/rgb_depth/detected_rgb_image',Image,queue_size=1)
        # 检测后且带有距离的Depth深度图像发布
        # self.detected_depth_pub = rospy.Publisher('/rgb_depth/detected_depth_image',Image,queue_size=1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_img_sub, self.depth_img_sub, self.result_sub, self.camera_info_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.detector_callback)

    def detector_callback(self, rgb_data,depth_data, result, camera_info):
        try:
            # camera_info_K = np.array(camera_info.K)
            
            # Intrinsic camera matrix for the raw (distorted) images.
            #     [fx  0 cx]
            # K = [ 0 fy cy]
            #     [ 0  0  1]
            m_fx = camera_info.K[0]
            m_fy = camera_info.K[4]
            m_cx = camera_info.K[2]
            m_cy = camera_info.K[5]
            inv_fx = 1. / m_fx
            inv_fy = 1. / m_fy

            cv_rgb = CvBridge().imgmsg_to_cv2(rgb_data, "bgr8")
            depth_image = CvBridge().imgmsg_to_cv2(depth_data, "32FC1")
            depth_array = np.array(depth_image, dtype=np.float32)
            cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
            depth_8 = (depth_array * 255).round().astype(np.uint8)
            cv_depth = np.zeros_like(cv_rgb)
            cv_depth[:,:,0] = depth_8
            cv_depth[:,:,1] = depth_8
            cv_depth[:,:,2] = depth_8

            # RGB图像尺寸
            rgb_height, rgb_width, rgb_channels = cv_rgb.shape

            # 在图像中心画轴
            cv2.line(cv_rgb,(int(rgb_width/2),int(rgb_height/2)),(int(rgb_width/2)+150,int(rgb_height/2)),(0,255,0),2)
            cv2.line(cv_rgb,(int(rgb_width/2),int(rgb_height/2)),(int(rgb_width/2),int(rgb_height/2)+150),(0,255,0),2)
            cv2.putText(cv_rgb, "x axis", (int(rgb_width/2)+180,int(rgb_height/2)), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.7, (200,255,0), 1, cv2.LINE_AA)
            cv2.putText(cv_rgb, "y axis", (int(rgb_width/2),int(rgb_height/2)+180), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.7, (125,255,0), 1, cv2.LINE_AA)

            # 遍历yolo的检测结果
            for object in result.pedestrians:
                print("有人")
                # 只保留stop sign 和人的检测
                # obj_width = object.width
                # obj_height = object.height
                # # 检测结果的方框对应的像素点
                # x_min = int(object.center.x - obj_width/2)
                # x_max = int(object.center.x + obj_width/2)
                # y_min = int(object.center.y - obj_height/2)
                # y_max = int(object.center.y + obj_height/2)

                x = int(object.center.x)
                y = int(object.center.y)
                w = object.width
                h = object.height

                cv2.circle(cv_rgb, (x,y), 1, (0, 0, 255), 4)

                # cv2.rectangle(cv_rgb,(x_min,y_min),(x_max,y_max),(255,0,0),2)
                # cv2.rectangle(cv_depth,(x_min,y_min),(x_max,y_max),(255,0,0),2)

                cv2.rectangle(cv_rgb,(x,y),(x+w,y+h),(255,0,0),2)
                # cv2.rectangle(cv_depth,(x,y),(x+w,y+h),(255,0,0),2)

                # 选取方框内局部的中心区域来作为深度的计算空间
                # roi_depth = depth_image[y_min+30:y_max-30, x_min+30:x_max-30]
                roi_depth = depth_image[y+30:y+h-30, x+30:x+w-30]

                # 用于记录有多少个点的深度信息是不为0的
                n = 0
                # 记录计算区域内的深度总和
                sum = 0
                for i in range(0,roi_depth.shape[0]):
                    for j in range(0,roi_depth.shape[1]):
                        depth_value = roi_depth.item(i, j)
                        if depth_value > 0.:
                            n = n + 1
                            sum = sum + depth_value
                if n == 0:
                    return
                # 求均值
                mean_z = sum / n

                # 通过相机内参矩阵计算像素坐标系变换到相机坐标系
                point_z = mean_z  # 变换到以m为单位-zed本身就是以m为单位
                # (x_min + x_max)/2计算检测的中心位置
                # point_x = (((x_min + x_max)/2) - m_cx) * point_z * inv_fx 
                # point_y = (((y_min + y_max)/2) - m_cy) * point_z * inv_fy
                point_x = ((x + w/2) - m_cx) * point_z * inv_fx
                point_y = ((y + h/2) - m_cy) * point_z * inv_fy

                # PointStamp
                targetPoint = PointStamped()
                targetPoint.point.x = point_x   

                # 用于显示在图像中的位置信息字符串
                x_str = "X: " + str(format(point_x, '.3f'))
                y_str = "Y: " + str(format(point_y, '.3f'))
                z_str = "Z: " + str(format(point_z, '.3f'))

                # 类别信息
                class_str = "class: pedestrians"

                cv2.putText(cv_rgb, class_str, (x+w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                0.7, (0,0,255), 1, cv2.LINE_AA) 

                cv2.putText(cv_rgb, x_str, (x+w, y+20), cv2.FONT_HERSHEY_SIMPLEX,  
                0.7, (0,0,255), 1, cv2.LINE_AA) 
                cv2.putText(cv_rgb, y_str, (x+w, y+40), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.7, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(cv_rgb, z_str, (x+w, y+60), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.7, (0,0,255), 1, cv2.LINE_AA)
                # 计算距离
                dist = math.sqrt(point_x * point_x + point_y * point_y + point_z * point_z)
                # 距离字符串用于显示
                dist_str = "dist:" + str(format(dist, '.2f')) + "m"

                cv2.putText(cv_rgb, dist_str, (x+w, y+80), cv2.FONT_HERSHEY_SIMPLEX,  
                0.7, (0,255,0), 1, cv2.LINE_AA)

        except CvBridgeError as e:
            print(e)
        
        # 合并RGB和depth图像，一起发布，在另一个Python文件中进行可视化
        # rgb_depth = np.concatenate((cv_rgb, cv_depth), axis=1)

        try:
            # convert opencv format back to ros format and publish result
            # rgb_depth_image = CvBridge().cv2_to_imgmsg(rgb_depth, "bgr8")
            # 将有标记的RGB图像发布出来
            rgb_image = CvBridge().cv2_to_imgmsg(cv_rgb, "bgr8")
            # 将带有标记的深度图像发布出来
            # depth_img = CvBridge().cv2_to_imgmsg(cv_depth, "bgr8")

            # self.detected_pub.publish(rgb_depth_image)
            self.detected_rgb_pub.publish(rgb_image)
            # self.detected_depth_pub.publish(depth_img)
        except CvBridgeError as e:
            print(e)        
    
    # def detector_callback(self, rgb_data,depth_data, result, camera_info):
    #     try:
    #         # camera_info_K = np.array(camera_info.K)
            
    #         # Intrinsic camera matrix for the raw (distorted) images.
    #         #     [fx  0 cx]
    #         # K = [ 0 fy cy]
    #         #     [ 0  0  1]
    #         m_fx = camera_info.K[0]
    #         m_fy = camera_info.K[4]
    #         m_cx = camera_info.K[2]
    #         m_cy = camera_info.K[5]
    #         inv_fx = 1. / m_fx
    #         inv_fy = 1. / m_fy

    #         cv_rgb = CvBridge().imgmsg_to_cv2(rgb_data, "bgr8")
    #         depth_image = CvBridge().imgmsg_to_cv2(depth_data, "32FC1")
    #         depth_array = np.array(depth_image, dtype=np.float32)
    #         cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
    #         depth_8 = (depth_array * 255).round().astype(np.uint8)
    #         cv_depth = np.zeros_like(cv_rgb)
    #         cv_depth[:,:,0] = depth_8
    #         cv_depth[:,:,1] = depth_8
    #         cv_depth[:,:,2] = depth_8

    #         # RGB图像尺寸
    #         rgb_height, rgb_width, rgb_channels = cv_rgb.shape

    #         # 在图像中心画轴
    #         cv2.line(cv_rgb,(int(rgb_width/2),int(rgb_height/2)),(int(rgb_width/2)+150,int(rgb_height/2)),(0,255,0),2)
    #         cv2.line(cv_rgb,(int(rgb_width/2),int(rgb_height/2)),(int(rgb_width/2),int(rgb_height/2)+150),(0,255,0),2)
    #         cv2.putText(cv_rgb, "x axis", (int(rgb_width/2)+180,int(rgb_height/2)), cv2.FONT_HERSHEY_SIMPLEX,  
    #                0.7, (200,255,0), 1, cv2.LINE_AA)
    #         cv2.putText(cv_rgb, "y axis", (int(rgb_width/2),int(rgb_height/2)+180), cv2.FONT_HERSHEY_SIMPLEX,  
    #                0.7, (125,255,0), 1, cv2.LINE_AA)

    #         # 用于发布检测到物体的位置、距离、类型
    #         objections = detector_info()
    #         objections.objects = []

    #         # 遍历yolo的检测结果
    #         for object in result.bounding_boxes:
    #             # 只保留stop sign 和人的检测
    #             # print(object.Class)
    #             if object.Class != "stop sign" and object.Class != "person":
    #                 continue
    #             else:
    #                 # 检测结果的方框对应的像素点
    #                 x_min = object.xmin
    #                 x_max = object.xmax
    #                 y_min = object.ymin
    #                 y_max = object.ymax

    #                 cv2.rectangle(cv_rgb,(x_min,y_min),(x_max,y_max),(255,0,0),2)
    #                 cv2.rectangle(cv_depth,(x_min,y_min),(x_max,y_max),(255,0,0),2)

    #                 # 选取方框内局部的中心区域来作为深度的计算空间
    #                 roi_depth = depth_image[y_min+30:y_max-30, x_min+30:x_max-30]

    #                 # 用于记录有多少个点的深度信息是不为0的
    #                 n = 0
    #                 # 记录计算区域内的深度总和
    #                 sum = 0
    #                 for i in range(0,roi_depth.shape[0]):
    #                     for j in range(0,roi_depth.shape[1]):
    #                         depth_value = roi_depth.item(i, j)
    #                         if depth_value > 0.:
    #                             n = n + 1
    #                             sum = sum + depth_value
    #                 if n == 0:
    #                     return
    #                 # 求均值
    #                 mean_z = sum / n

    #                 # 通过相机内参矩阵计算像素坐标系变换到相机坐标系
    #                 point_z = mean_z  # 变换到以m为单位-zed本身就是以m为单位
    #                 # (x_min + x_max)/2计算检测的中心位置
    #                 point_x = (((x_min + x_max)/2) - m_cx) * point_z * inv_fx 
    #                 point_y = (((y_min + y_max)/2) - m_cy) * point_z * inv_fy

    #                 # 用于显示在图像中的位置信息字符串
    #                 x_str = "X: " + str(format(point_x, '.3f'))
    #                 y_str = "Y: " + str(format(point_y, '.3f'))
    #                 z_str = "Z: " + str(format(point_z, '.3f'))

    #                 # 类别信息
    #                 class_str = "class:"+object.Class

    #                 cv2.putText(cv_rgb, class_str, (x_max, y_min), cv2.FONT_HERSHEY_SIMPLEX,  
    #                 0.7, (0,0,255), 1, cv2.LINE_AA) 

    #                 cv2.putText(cv_rgb, x_str, (x_max, y_min+20), cv2.FONT_HERSHEY_SIMPLEX,  
    #                 0.7, (0,0,255), 1, cv2.LINE_AA) 
    #                 cv2.putText(cv_rgb, y_str, (x_max, y_min+40), cv2.FONT_HERSHEY_SIMPLEX,  
    #                         0.7, (0,0,255), 1, cv2.LINE_AA)
    #                 cv2.putText(cv_rgb, z_str, (x_max, y_min+60), cv2.FONT_HERSHEY_SIMPLEX,  
    #                         0.7, (0,0,255), 1, cv2.LINE_AA)
    #                 # 计算距离
    #                 dist = math.sqrt(point_x * point_x + point_y * point_y + point_z * point_z)
    #                 # 距离字符串用于显示
    #                 dist_str = "dist:" + str(format(dist, '.2f')) + "m"

    #                 cv2.putText(cv_rgb, dist_str, (x_max, y_min+80), cv2.FONT_HERSHEY_SIMPLEX,  
    #                 0.7, (0,255,0), 1, cv2.LINE_AA)

    #                 objection_info = self.transform_objection(point_x,point_y,point_z,dist,object.Class)
    #                 objections.objects.append(objection_info)
    #                 objections.header.stamp = rospy.Time(0)
    #                 # 发布
    #                 self.detector_info_pub.publish(objections)

    #     except CvBridgeError as e:
    #         print(e)
        
    #     # 合并RGB和depth图像，一起发布，在另一个Python文件中进行可视化
    #     rgb_depth = np.concatenate((cv_rgb, cv_depth), axis=1)

    #     try:
    #         # convert opencv format back to ros format and publish result
    #         rgb_depth_image = CvBridge().cv2_to_imgmsg(rgb_depth, "bgr8")
    #         # 将有标记的RGB图像发布出来
    #         rgb_image = CvBridge().cv2_to_imgmsg(cv_rgb, "bgr8")
    #         # 将带有标记的深度图像发布出来
    #         # depth_img = CvBridge().cv2_to_imgmsg(cv_depth, "bgr8")

    #         self.detected_pub.publish(rgb_depth_image)
    #         self.detected_rgb_pub.publish(rgb_image)
    #         # self.detected_depth_pub.publish(depth_img)
    #     except CvBridgeError as e:
    #         print(e)

    # 返回的是自定义的消息内容，包含位置和距离、类型的信息
    def transform_objection(self,point_x, point_y, point_z, dist, type):
        object_point = PointStamped()
        # 图像和camera的坐标系之间的关系
        object_point.point.x = point_z
        object_point.point.y = -point_x
        object_point.point.z = -point_y
        object_point.header.frame_id = "front_single_camera_link"          # 仿真里是这个link但实际车上要修改

        # 转换到basefootprint坐标系中
        object_point = self.transform_to_target_frame(init_pose=object_point,target_frame='base_footprint',inital_frame='front_single_camera_link')    # base_footprint可能不是gem的坐标系，要检查

        objection_info = object_info()
        objection_info.objection_point = object_point
        objection_info.type = type
        objection_info.distance.data = dist

        return objection_info

    # transform the pose from initial frame to target frame
    def transform_to_target_frame(self,init_pose,target_frame, inital_frame):
        self.listener = tf.TransformListener()
        self.listener.waitForTransform(inital_frame,target_frame,rospy.Time(),rospy.Duration(4.0))

        got_tf_transform = False
        while got_tf_transform == False:
            try:
                now = rospy.Time.now()

                (trans,rot) = self.listener.lookupTransform(inital_frame, target_frame, now)
            
                got_tf_transform = True
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                got_tf_transform = False
        # have got the transform matrix, and then just calculate the position of odom at the frame of map
        if got_tf_transform == True:
            rob_in_target_frame = self.listener.transformPoint(target_frame,init_pose)
            return rob_in_target_frame
        else:
            return

if __name__ == "__main__":
    rospy.init_node('rgb_depth_calculator',anonymous=True)
    m_rgb_depth = RGB_Depth()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
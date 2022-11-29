#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date           :2022/11/26 09:57:26
@Author         :Yandong Luo
@version        :1.0
@Description    : 用于pointPillars的parallel parking
'''

import rospy
import numpy as np
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped,PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from gazebo_msgs.srv import GetModelState,GetModelStateResponse
from gazebo_msgs.msg import ModelState
import tf
import tf2_ros
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from rgb_depth.msg import detector_info

class map_Builder():
    def __init__(self) -> None:
        # rospy.Subscriber("/detection/lidar_detector/objects", DetectedObjectArray, self.callback3)
        rospy.Subscriber("/detector_info",detector_info, self.callback3)
        rospy.Subscriber("/replanning",Bool,self.replan_callback)
        self.pub_map = rospy.Publisher('/map', OccupancyGrid, queue_size=1)
        self.pub_goal = rospy.Publisher('/goal_pose',PoseStamped,queue_size=1)
        self.pub_start = rospy.Publisher('/start_pose',PoseWithCovarianceStamped,queue_size=1)
        self.replan = True
    
    def replan_callback(self, replan_msg):
        self.replan = replan_msg.data
    
    # 方案三，创建一个map，map的坐标系采用的是以车右边方向为x，以前进为y，parking位置位于右边，resolution设为1米
    def callback3(self, objects_data):
        if self.replan == False:
            return
        
        self.replan = False

        height = 40
        width = 40
        factor = 1      # 4个grid=1m

        # initial map data
        map_data = OccupancyGrid()
        map_data.header.frame_id = "map"
        # 设定map的尺寸，尺寸适合能够优化计算，目前初始估计为20x20的栅格，每栅格1m
        map_data.info.height = height
        map_data.info.width = width
        map_data.info.resolution = 1     # The map resolution [m/cell] 一个cell里边长为多少米，默认为一米
        map_data.info.origin.position.x = -width/2/factor
        map_data.info.origin.position.y = -height/2/factor
        map_data.info.origin.position.z = 0
        map_data.info.origin.orientation.x = 0
        map_data.info.origin.orientation.y = 0
        map_data.info.origin.orientation.z = 0
        map_data.info.origin.orientation.w = 1

        # 初始化map内容全为0，用一个二维数组来表达
        map = np.zeros((width, height))

        for object in objects_data.objects:
            if object.type != "stop sign":
                continue
            object_x = object.point.x
            object_y = object.point.y

            # 在robot frame下所检测到的目标位置变换到map坐标系中
            rob_object_pose = PointStamped()
            rob_object_pose.header.frame_id = "/base_footprint"
            rob_object_pose.header.stamp =rospy.Time(0)
            rob_object_pose.point.x = object_x
            rob_object_pose.point.y = object_y
            rob_object_pose.point.z = 0

            # 将车辆坐标系所检测到的物体的坐标转换到map frame下
            map_object_pos = self.transform_to_map_frame(init_pose=rob_object_pose,target_frame='map',inital_frame='base_footprint')
            map_object_pos_x = map_object_pos.point.x
            map_object_pos_y = map_object_pos.point.y

            col = round(width/2) + factor*round(map_object_pos_x)
            row = round(height/2) + factor*round(map_object_pos_y)

            if col>height or row>width:
                print("object的尺寸超出了栅格化地图的边界了")
                map = np.zeros((width, height))
                # return
            else:
                map[row][col] = 100

        convert_data = np.reshape(map.astype(int), (1, height*width))
        map_data.data = convert_data[0]
        self.pub_map.publish(map_data)

        self.pubGoalPose(map_object_pos_x, map_object_pos_y)
        self.pubStartPose()
    
    def pubGoalPose(self,map_object_pos_x,map_object_pos_y):
        goal_x = map_object_pos_x
        goal_y = map_object_pos_y-0.5     # stop sign 前0.5米

        goal_data = PoseStamped()
        goal_data.pose.position.x = goal_x
        goal_data.pose.position.y = goal_y
        goal_data.pose.position.z = 0
        goal_data.pose.orientation.x = 0
        goal_data.pose.orientation.y = 0
        goal_data.pose.orientation.z = 0.706825181105366
        goal_data.pose.orientation.w = 0.7073882691671998

        self.pub_goal.publish(goal_data)
    
    # Publish the start pose
    def pubStartPose(self):
        # get current position and orientation in the world frame
        cur_x, cur_y, cur_yaw = self.get_gem_pose()

        odom_rob_pose = PointStamped()
        odom_rob_pose.header.frame_id = "/odom"
        odom_rob_pose.header.stamp =rospy.Time(0)
        odom_rob_pose.point.x = cur_x
        odom_rob_pose.point.y = cur_y
        odom_rob_pose.point.z = 0

        # 将odom坐标系的车辆坐标转换到map frame下
        map_rob_pos = self.transform_to_map_frame(init_pose=odom_rob_pose,target_frame='map',inital_frame='odom')

        init_data = PoseWithCovarianceStamped()
        init_data.pose.pose.position.x = map_rob_pos.point.x
        init_data.pose.pose.position.y = map_rob_pos.point.y
        init_data.pose.pose.position.z = 0

        init_data.pose.pose.orientation.x = 0
        init_data.pose.pose.orientation.y = 0
        init_data.pose.pose.orientation.z = 0.706825181105366
        init_data.pose.pose.orientation.w = 0.7073882691671998

        self.pub_start.publish(init_data)
    
    # transform the point from initial frame to target frame
    def transform_to_map_frame(self,init_pose,target_frame, inital_frame):
        self.listener = tf.TransformListener()
        self.listener.waitForTransform(inital_frame,target_frame,rospy.Time(),rospy.Duration(4.0))

        got_tf_transform = False
        while got_tf_transform == False:
            try:
            #   rospy.loginfo('Waiting for the robot transform')
                now = rospy.Time.now()

                #   self.listener.waitForTransform("/odom","/map",now,rospy.Duration(4.0))  # 过时了这方法

                (trans,rot) = self.listener.lookupTransform(inital_frame, target_frame, now)
            
                got_tf_transform = True
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                got_tf_transform = False
        # have got the transform matrix, and then just calculate the position of odom at the frame of map
        if got_tf_transform == True:
            rob_in_map_frame = self.listener.transformPoint(target_frame,init_pose)
            return rob_in_map_frame
        else:
            return



if __name__ == "__main__":
    # init args
    rospy.init_node('mapbuilder_node', anonymous=True)
    map_builder = map_Builder()
    rospy.spin()
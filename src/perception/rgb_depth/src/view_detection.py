#!/usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class viewer:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/rgb_depth/detected_image", Image, self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)    

    cv2.imshow("distance calculator", cv_image)
    cv2.waitKey(30)
   

def main(args):
  v = viewer()
  rospy.init_node('viewer_detector', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

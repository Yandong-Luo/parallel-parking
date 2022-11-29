import rospy
import math
import argparse

from gazebo_msgs.msg import  ModelState
from controller import bicycleModel
import time
from waypoint_list import WayPoints

def run_model():
    rospy.init_node("model_dynamics")
    model = bicycleModel(velocity = 3.0, deceleration = 0.0)
    
    pos_list = [[100,-98.5],[200,-98.5],[210,-98.5],[220,-97],[230,-95],[240,-92],[245,-90],[250,-87],[255,-85],
                [260,-81.5],[265,-78],[270,-74],[275,-69],[280,-63],[285,-57],[288,-52],[290,-49],[292,-45.5],
                [295,-39],[298,-30], [300,-22],[303,-5]]
    pos_idx = 0

    targetState = ModelState()
    targetState.pose.position.x = pos_list[pos_idx][0]
    targetState.pose.position.y = pos_list[pos_idx][1]

    def shutdown(): 
        """Stop the car when this ROS node shuts down"""
        model.stop()
        rospy.loginfo("Stop the car")

    rospy.on_shutdown(shutdown)

    rate = rospy.Rate(100)  # 100 Hz

    while not rospy.is_shutdown():
        rate.sleep()  # Wait a while before trying to get a new state

        # Get the current position and orientation of the vehicle
        currState =  model.getModelState()

        if not currState.success:
            continue

        distToTargetX = abs(targetState.pose.position.x - currState.pose.position.x)
        distToTargetY = abs(targetState.pose.position.y - currState.pose.position.y)

        if (distToTargetX < 1 and distToTargetY < 1):
            # if safe and at the target - move to the next target
            prev_pos_idx = pos_idx
            pos_idx = pos_idx+1
            if pos_idx == len(pos_list):
                print("explored all waypoints")
                model.stop()
                exit(0)
            targetState = ModelState()
            targetState.pose.position.x = pos_list[pos_idx][0]
            targetState.pose.position.y = pos_list[pos_idx][1]
            print("reached",pos_list[prev_pos_idx][0],pos_list[prev_pos_idx][1],"next",pos_list[pos_idx][0],pos_list[pos_idx][1])
        else:
            # safe, but not yet at traget - chance control to move to target
            model.setModelState(currState, targetState, "run")

if __name__ == "__main__":
    try:
        run_model()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutting down")
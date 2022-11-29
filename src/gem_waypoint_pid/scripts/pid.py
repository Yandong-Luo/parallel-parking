#!/usr/bin/python

import time
import numpy as np
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class PID:
    def __init__(
        self,
        Kp=0.0,
        Ki=0.0,
        Kd=0.0,
        set_point=0.0,
        sample_time=0.01,
        out_limits=(None, None),
    ):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

        self.set_point = set_point

        self.sample_time = sample_time

        self.out_limits = out_limits

        self.last_err = 0.0

        self.last_time = time.time()

        self.output = 0.0

        self.k_x = 0.1
        self.k_y = 0.1
        self.k_v = 0.5
        self.k_theta = 1.0

    def update(self, currentPose,referencePose):
        """Compute PID control value based on feedback_val.
        """

        # TODO: implement PID control
        orientation_list = [
            currentPose.pose.orientation.x,
            currentPose.pose.orientation.y,
            currentPose.pose.orientation.z,
            currentPose.pose.orientation.w,
        ]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        # current state
        current_x = currentPose.pose.position.x
        current_y = currentPose.pose.position.y
        current_theta = yaw

        current_vx = currentPose.twist.linear.x
        current_vy = currentPose.twist.linear.y
        current_v = math.sqrt(pow(current_vx, 2) + pow(current_vy, 2))

        # reference state
        reference_x = referencePose[0]
        reference_y = referencePose[1]
        reference_orientation = referencePose[2]
        reference_v = referencePose[3]

        # The along-track error: (x ∗ (t) − x B (t)) cos θ B (t) + (y ∗ (t) − y B (t)) sin θ B (t)
        delta_s = (reference_x - current_x)*np.cos(current_theta) + (reference_y - current_y)*np.sin(current_theta)

        # The cross-track error: −(x ∗ (t) − x B (t)) sin θ B (t) + (y ∗ (t) − y B (t)) cos θ B (t).
        delta_n = -(reference_x - current_x)*np.sin(current_theta) +  (reference_y - current_y)*np.cos(current_theta)

        # The heading error
        delta_theta = reference_orientation - current_theta

        # The speed error
        delta_v = reference_v - current_v

        # using matrix to describe the error part
        error_u = np.array([[delta_s], [delta_n], [delta_theta], [delta_v]])

        # K matrix
        # k_x = 0.1
        # k_y = 0.5
        # k_v = 0.5
        # k_theta = 0.8
        # K = np.array([[k_x, 0, 0, k_v],[0,k_y,k_theta,0]])

        K = np.array([[self.k_x, 0, 0, self.k_v],[0,self.k_y,self.k_theta,0]])
        

        # U matrix
        U = np.array([[current_v],[currentPose.twist.angular.z]])
        
        result = np.dot(K,error_u)

        return result

    def __call__(self, currentPose, referencePose):
        return self.update(currentPose,referencePose)

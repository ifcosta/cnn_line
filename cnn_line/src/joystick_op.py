#!/usr/bin/env python3

import rospy, argparse
import numpy as np

from time import time_ns
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger

class JoystickControllerNode:
    def __init__(self):
        rospy.init_node('joystick_controller')

        self.rate = rospy.Rate(args.publish_frequency)

        # Subscribers
        rospy.Subscriber('/joy', Joy, self.joy_callback)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher(args.output_topic, Twist, queue_size=10)

        self.start_button_pressed = False
        self.select_button_pressed = False
        self.a_button_pressed = False

        self.twist_msg = Twist()

        self.joy_received = False

        self.spd_gain = 0.5
        self.check = time_ns()
        self.axis0 = 0
        self.axis1 = 0

        self.control_type = ''

    def joy_callback(self, data):
        # Get joystick values and apply a curve
        AX0 = data.axes[1]**3
        AX1 = data.axes[0]**3

        self.axis0 = AX0  # Left joystick X-axis
        self.axis1 = AX1  # Left joystick Y-axis

        # Check for button presses
        self.start_button  = data.buttons[7] == 1  # Start button
        self.select_button = data.buttons[6] == 1  # Select button
        self.a_button = data.buttons[0] == 1       # A button
        self.b_button = data.buttons[1] == 1       # B shoulder button
        self.L_button = data.buttons[4] == 1       # L shoulder button
        self.R_button = data.buttons[5] == 1       # R shoulder button

        self.joy_received = True

    def perform_actions(self):
        if self.start_button and self.select_button:
            # Toggle autonomous mode when Start and Select buttons are pressed together
            mode = rospy.get_param('/autonomous_mode')
            mode = not mode
            rospy.set_param('/autonomous_mode', mode)
            rospy.loginfo('Autonomous mode status:' + str(mode))
        
        if self.a_button:
            # Perform the home steering process when A button is pressed
                rospy.wait_for_service('/base_driver/home_steering')
                try:
                    home_steering = rospy.ServiceProxy('/base_driver/home_steering', Trigger)
                    home_steering.call()
                    msg = 'Home Steering Done Complete'
                    rospy.loginfo(msg)
                except rospy.ServiceException as e:
                    msg = 'Home Steering Failed'
                    rospy.logerr(msg + '   ' + str(e))
        
        # Check if the L button is pressed
        if self.L_button:
            # Decrease the speed gain by 0.1
            self.spd_gain -= 0.1
            
            # Ensure the speed gain does not go below 0.1
            if self.spd_gain < 0.1:
                self.spd_gain = 0.1
            
            # Update the ROS parameter for linear velocity with the new speed gain
            rospy.set_param('/linear_velocity', self.spd_gain)

        # Check if the R button is pressed
        if self.R_button:
            # Increase the speed gain by 0.1
            self.spd_gain += 0.1
            
            # Ensure the speed gain does not exceed 1.5
            if self.spd_gain > 1.5:
                self.spd_gain = 1.5
            
            # Update the ROS parameter for linear velocity with the new speed gain
            rospy.set_param('/linear_velocity', self.spd_gain)

        # Check if both the A and B buttons are pressed simultaneously
        if self.a_button and self.b_button:
            # Get the current control type from the ROS parameter
            self.control_type = rospy.get_param('/velocity_control')

            # Toggle between control types ('cascade' and 'row_control')
            if self.control_type == 'cascade':
                self.control_type = 'row_control'
            elif self.control_type == 'row_control':
                self.control_type = 'cascade'

            # Update the ROS parameter for velocity control type with the new control type
            rospy.set_param('/velocity_control', self.control_type)

            # Log the updated control type
            rospy.loginfo('Control type set: ' + self.control_type)
    
    def run(self):
        while not rospy.is_shutdown():
            # Check if a joystick message has been received
            if self.joy_received:
                self.joy_received = False

                # Calculate linear and angular velocities based on joystick axis values and speed gain
                self.twist_msg.linear.x = self.axis0 * self.spd_gain
                self.twist_msg.linear.y = 0.0
                self.twist_msg.linear.z = 0.0
                self.twist_msg.angular.x = 0.0
                self.twist_msg.angular.y = 0.0
                self.twist_msg.angular.z = self.axis1 * self.spd_gain / 2

                # Publish the calculated twist message to control the robot's motion
                self.cmd_vel_pub.publish(self.twist_msg)

                # Check if a certain time has passed to perform button actions
                # This was done so the buttons aren't registered twice
                # TODO:should be improved
                if ((time_ns() - self.check) / 1000) > 200000:
                    self.perform_actions()   # Perform button actions
                    self.check = time_ns()   # Update the time of the last action

            # Check if there is a non-zero joystick axis value
            # This was done so the speed still is published if the values doesn't updade
            # the node will not update the velocity if it is zero, to not get in the way of other nodes
            # TODO:this shold be impoved
            if self.axis0 != 0 or self.axis1 != 0:
                # Publish the twist message to control the robot's motion
                self.cmd_vel_pub.publish(self.twist_msg)

            # Pause for a short duration to control the loop rate
            self.rate.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish_frequency', type=float,          nargs='?',default=50)
    parser.add_argument('--output_topic',      type=str,            nargs='?',default='/nav_vel')
    parser.add_argument('__name', type=str,  nargs='?',default='')
    parser.add_argument('__log', type=str,  nargs='?',default='')
    args = parser.parse_args()

    try:
        joystick_controller_node = JoystickControllerNode()
        joystick_controller_node.run()
    except rospy.ROSInterruptException:
        pass
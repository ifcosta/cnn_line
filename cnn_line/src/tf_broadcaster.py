#!/usr/bin/env python  
import rospy

import numpy as np
import tf2_ros, tf_conversions
import geometry_msgs.msg
from tf2_geometry_msgs import PoseStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates

model_ = "soybot"
model_pose_ = PoseStamped()
model_pose_.pose.orientation.w = 1
model_pose_.header.frame_id = "world"
model_idx = 0

def stateCallback(msg):
    global model_pose_,model_idx, t, t2
    model_pose_.header.stamp = rospy.Time.now()
    
    if msg.name[model_idx] == model_:
        model_pose_.pose = msg.pose[model_idx]
    else:
        for idx, name in enumerate(msg.name):
            if name == model_:
                model_pose_.pose = msg.pose[idx]
                model_idx = idx
    #print(model_pose_)

    x_pose = model_pose_.pose.position.x
    y_pose = model_pose_.pose.position.y
    z_pose = model_pose_.pose.position.z
    x_rot = model_pose_.pose.orientation.x
    y_rot = model_pose_.pose.orientation.y
    z_rot = model_pose_.pose.orientation.z
    w_rot = model_pose_.pose.orientation.w

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world" #world #base_link_control
    t.child_frame_id = "base_link" #base_link #base_link
    t.transform.translation.x = x_pose
    t.transform.translation.y = y_pose
    t.transform.translation.z = z_pose
    t.transform.rotation.x = x_rot
    t.transform.rotation.y = y_rot
    t.transform.rotation.z = z_rot
    t.transform.rotation.w = w_rot
    #(roll, pich, yaw) = tf_conversions.transformations.euler_from_quaternion([x_rot,y_rot,z_rot,w_rot])
    

    t2.header.stamp = rospy.Time.now()
    t2.header.frame_id = "world"
    t2.child_frame_id = "odom"
    t2.transform.translation.x = -0.5
    t2.transform.translation.y = 1.0
    t2.transform.translation.z = 0
    #(x, y, z, w) = tf_conversions.transformations.quaternion_from_euler(-roll, -pich, -yaw)
    t2.transform.rotation.x = 0
    t2.transform.rotation.y = 0
    t2.transform.rotation.z = 0
    t2.transform.rotation.w = 1


def main():
    while not rospy.is_shutdown():
    
        pose_publisher.publish(model_pose_)
        br.sendTransform(t)
        br.sendTransform(t2)
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('tf2_model_broadcaster')
    rospy.Subscriber('/gazebo/model_states', ModelStates, stateCallback)

    pose_publisher = rospy.Publisher(model_ + "_pose", PoseStamped, queue_size=10)
   

    br = tf2_ros.TransformBroadcaster()

    t  = geometry_msgs.msg.TransformStamped()
    t2 = geometry_msgs.msg.TransformStamped()
    rate = rospy.Rate(50)

    try:
        main()
    except rospy.ROSInterruptException:
        pass    
    #rospy.spin()

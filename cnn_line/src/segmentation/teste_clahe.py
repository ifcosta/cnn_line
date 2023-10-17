from clahe import *

import rospy

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge

import cv2

import numpy as np

def PointCLoudCallback(msg):
	
	global depth_cloud
	global flagPointCLoudCallback
	depth_cloud = msg	
	flagPointCLoudCallback = 1 

def ImageCallbackRgb(msg):
	
	global cv_image_RGB
	global flagImageCallbackRgb 
	raw_image = msg
	
	bridge = CvBridge()
	image = bridge.imgmsg_to_cv2(raw_image, "bgr8")
	cv_image_RGB = image
	flagImageCallbackRgb = 1
	
def ImageCallbackIr(msg):
	
	global cv_image_ir
	global flagImageCallbackIr

	raw_image = msg
	
	bridge = CvBridge()
	image = bridge.imgmsg_to_cv2(raw_image, "passthrough")
	cv_image_ir = image
	flagImageCallbackIr = 1
	
def ImageCallbackDepthRGB(msg):
	
	global cv_image_depthRGB
	global flagImageCallbackDepthRGB

	raw_image = msg
	
	bridge = CvBridge()
	image = bridge.imgmsg_to_cv2(raw_image, "passthrough")
	cv_image_depthRGB = image	
	flagImageCallbackDepthRGB = 1
	
def ImageCallbackDepthIR(msg):
	
	global cv_image_depthIR
	global flagImageImageCallbackDepthIR

	raw_image = msg
	
	bridge = CvBridge()
	image = bridge.imgmsg_to_cv2(raw_image, "passthrough")
	cv_image_depthIR = image
	flagImageImageCallbackDepthIR = 1
	

def segmentar():
    image = cv_image_depthRGB
    mask = clahe_otsu(image)
	

if __name__ == '__main__':
    global flagPointCLoudCallback, flagImageCallbackRgb, flagImageCallbackIr, flagImageCallbackDepthRGB, flagImageImageCallbackDepthIR
    global depth_cloud, cv_image_RGB
    flagPointCLoudCallback = 0
    flagImageCallbackRgb = 0
    flagImageCallbackIr = 0
    flagImageCallbackDepthRGB = 0
    flagImageImageCallbackDepthIR = 0
    rospy.init_node('rs_embrapa', anonymous= True)
    #rospy.Subscriber('/camera/depth/color/points', PointCloud2, PointCLoudCallback)
    rospy.Subscriber('/camera/depth_registered/points', PointCloud2, PointCLoudCallback)
    rospy.Subscriber('/camera/color/image_raw', Image, ImageCallbackRgb)
    rospy.Subscriber('/camera/infra1/image_rect_raw', Image, ImageCallbackIr)	
    rospy.Subscriber('camera/aligned_depth_to_color/image_raw', Image, ImageCallbackDepthRGB)
    rospy.Subscriber('/camera/aligned_depth_to_infra1/image_raw', Image, ImageCallbackDepthIR)	
    while(not flagImageCallbackDepthRGB):
        pass
    while(True):
        segmentar()
    cv2.destroyAllWindows()
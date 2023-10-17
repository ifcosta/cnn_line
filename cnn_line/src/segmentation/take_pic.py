#!/usr/bin/env python

import rospy
import pickle as pkl
import time
import argparse

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge


#parser = argparse.ArgumentParser()
#parser.add_argument("-f", '--filename', help="Name of file to be debayered")
#args = parser.parse_args()

filename = 'soytest'

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
	
def ImageNavTopCallback(msg):
    global flagImageTop
    global cv_image_top
    

    raw_image = msg

    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(raw_image, "passthrough")
    cv_image_top = image
    flagImageTop = 1
	
def main():
    global flagPointCLoudCallback, flagImageCallbackRgb, flagImageCallbackIr, flagImageCallbackDepthRGB, flagImageImageCallbackDepthIR, flagImageTop
    global depth_cloud, cv_image_RGB
    flagPointCLoudCallback = 0
    flagImageCallbackRgb = 0
    flagImageCallbackIr = 0
    flagImageCallbackDepthRGB = 0
    flagImageImageCallbackDepthIR = 0
    flagImageTop = 0
    
    rospy.init_node('rs_embrapa', anonymous= True)
    #rospy.Subscriber('/camera/depth/color/points', PointCloud2, PointCLoudCallback)
    rospy.Subscriber('/camera/depth_registered/points', PointCloud2, PointCLoudCallback)
    rospy.Subscriber('/camera/color/image_raw', Image, ImageCallbackRgb)
    rospy.Subscriber('/camera/infra1/image_rect_raw', Image, ImageCallbackIr)	
    rospy.Subscriber('camera/aligned_depth_to_color/image_raw', Image, ImageCallbackDepthRGB)
    rospy.Subscriber('/camera/aligned_depth_to_infra1/image_raw', Image, ImageCallbackDepthIR)	
    rospy.Subscriber('/nav_front/image_raw',Image,ImageNavTopCallback)
	

	
    while flagImageTop ==0 or flagPointCLoudCallback == 0 or flagImageCallbackRgb == 0 or flagImageCallbackIr == 0 or flagImageCallbackDepthRGB == 0 or flagImageImageCallbackDepthIR == 0:
        #print("ESPERANDO MENSAGENS")
        #print("{}, {}, {}, {}, {}".format(flagPointCLoudCallback, flagImageCallbackRgb,flagImageCallbackIr, flagImageCallbackDepthRGB, flagImageImageCallbackDepthIR))
        #time.sleep(1)
        pass


def save():

    with open(filename + ".pkl", 'wb') as f:
        #pkl.dump({'depth_cloud': depth_cloud, 'cv_image_RGB': cv_image_RGB}, f,pkl.HIGHEST_PROTOCOL)
		pkl.dump({'nav_top': cv_image_top, 'cv_image_RGB': cv_image_RGB, 'cv_image_ir': cv_image_ir,'cv_image_depthRGB': cv_image_depthRGB,'cv_image_depthIR': cv_image_depthIR}, f,pkl.HIGHEST_PROTOCOL)
    with open(filename + "_pc2.pkl", 'wb') as f:
        pkl.dump({'depth_cloud': depth_cloud}, f,pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
    save()


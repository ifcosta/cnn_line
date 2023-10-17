#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2', '3'}

import rospy, cv2, tf2_ros, math, tf_conversions,time
import numpy as np
from time import time_ns
from pathlib import Path

from cv_bridge import CvBridge
from image_geometry.cameramodels import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo, Joy

from geometry_msgs.msg import Twist, PoseStamped
from tf2_geometry_msgs import PointStamped

from sklearn.linear_model import LinearRegression
from suppress_TF_REPEATED_DATA import *


IMAGE_SIZE = 256
NUM_CLASSES = 3

bridge = CvBridge()
top_image = None
bot_image = None
camera = None
robot_pose = PoseStamped()
img_stamp = 0
last_save = time.time()-5
cmd = Twist()
current_state = [0]*13
button_state = [0]*13

control_axl = None

img_top_stack = []
img_bot_stack = []

rows = np.asarray([1.0])


#ROS
def callback_joy(msg):
    global cmd, current_state, control_axl
    current_state = msg.buttons
    control_axl = msg.axes
    cmd.linear.x = msg.axes[1]
    cmd.angular.z = msg.axes[0]

def callback(msg):
    global top_image, img_stamp
    top_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    img_stamp = msg.header.stamp

def callback2(msg):
    global bot_image
    bot_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

def pose_callback(msg):
    global robot_pose
    robot_pose = msg

def get_caminfo():
    global camera
    cam_info = None
    cam_info = rospy.wait_for_message('/nav_front/camera_info',CameraInfo)
    if cam_info is not None:
        camera = PinholeCameraModel()
        camera.fromCameraInfo(cam_info)

def point_conversion(point,cur_frame,dest_frame):
    cur_point = PointStamped()
    dest_point = PointStamped()
    cur_point.header.stamp = rospy.Time(0) #img_stamp 
    cur_point.header.frame_id = cur_frame
    cur_point.point.x = point[0]
    cur_point.point.y = point[1]
    cur_point.point.z = point[2]

    #tfBuffer.lookup_transform(cur_frame,dest_frame,img_stamp,rospy.Duration(1))
    try:
        dest_point = tfBuffer.transform(cur_point, dest_frame,rospy.Duration(0))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
        #print (err)
        pass

    return (dest_point.point.x,dest_point.point.y,dest_point.point.z)

#Image Processing
def exgr_mask(image, th = 0.10):
    if(isinstance(th, list)):
        th = th[-1]
    im_to_binarize = image.astype('float')
    
    R_ = im_to_binarize[:,:,2]/np.max(im_to_binarize[:,:,2])
    G_ = im_to_binarize[:,:,1]/np.max(im_to_binarize[:,:,1])
    B_ = im_to_binarize[:,:,0]/np.max(im_to_binarize[:,:,0])
    
    r = R_/(R_+G_+B_+0.000001)
    g = G_/(R_+G_+B_+0.000001)
    b = B_/(R_+G_+B_+0.000001)
    
    excess_red = 0.6*r - g
    excess_green = 2*g - r - b*2
    
    #a = 2*im_to_binarize[:,:,1] - im_to_binarize[:,:,0] - im_to_binarize[:,:,2]
    #excess_red = 1.4*im_to_binarize[:,:,2] - im_to_binarize[:,:,1]
    eG_eR = excess_green - excess_red
    
    thresh2 = np.where(eG_eR > th, 1.0, 0.0)
    mask = cv2.normalize(thresh2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return mask

def to_class(mask):
    r,g,b = cv2.split(mask)
    r = np.clip(r,0,1)
    g = np.clip(g,0,1)
    b = np.clip(b,0,1)
    
    class0 = b*(1-r)
    class1 = r*(1-b)
    class2 = g
    class3 = r*b
    
    t = np.stack(( class0 , class1 , class2 , class3 ), axis=2)
    
    return t.argmax(axis=2).astype(np.uint8)

def line_from_tf2(start_point,end_point,camera_frame='camera_link',img=None):
    (x_i,y_i) = start_point
    (x_f,y_f) = end_point

    (x,y,z) = point_conversion((x_i,y_i,0.02),'world',camera_frame)
    u1,v1 = camera.project3dToPixel((x,y,z))

    (x,y,z) = point_conversion((x_f,y_f,0.02),'world',camera_frame)
    u2,v2 = camera.project3dToPixel((x,y,z))

    if not (math.isnan(u1) or math.isnan(u2)):
        x_values = np.array([v1,v2])
        if camera_frame == 'camera_link2':
            diff = (u2-u1)
            y_values = np.array([u1+diff*0.08,u2-diff*0.08])
        else:
            y_values = np.array([u1,u2])

        slope, y_intercept = np.polyfit(x_values, y_values, 1)

        x0 = int(y_intercept)
        x1 = int(slope * 480 + y_intercept)
        xh = int(slope * 315 + y_intercept) #hardcoded size

        if img is not None:
            cv2.circle(img,(int(u1  +diff*0.08   ),int(v1)),5,(255,255,0),3) 
            cv2.circle(img,(int(u2  -diff*0.08   ),int(v2)),5,(255,0,255),3)
            cv2.circle(img,(int(xh),int(315)),5,(0,255,0),3)

            cv2.line(img,(int(x0),int(0)),(int(x1),int(480)),(0,255,255),2)
            cv2.imshow(camera_frame, cv2.cvtColor(img,cv2.COLOR_RGB2BGR))

        return (x0,x1,xh)

    else:

        return (-100,-100,-100)

def save_data(img,mask,cat,lbl,path,im_id):
    cv2.imwrite(str(path)+str(im_id)+'_img.png' ,img ) #,imagem base
    cv2.imwrite(str(path)+str(im_id)+'_mask.png',mask) #,mascara RGB
    cv2.imwrite(str(path)+str(im_id)+'_cat.png' ,cat ) #,mascara categorizada

    filename = Path(str(path)+str(im_id)+'_lbl.txt')
    #filename.touch()  # will create file, if it exists will do nothing
    out = open(filename, 'w+')
    print(lbl,file = out) #,lbl: x_inicio da linha e x_final da linha
    out.close()

    im_id = time.time()
    return im_id

def show_display():
    global last_save, img_top_stack, img_bot_stack, button_state, current_state, rows, count
    
    if len(img_top_stack) < 2: 
        img_top_stack = [top_image.copy() for _ in range(7)]
        img_bot_stack = [bot_image.copy() for _ in range(7)]

    img_top_stack.append(top_image); img_bot_stack.append(bot_image)
    img_top_stack.pop(0)           ; img_bot_stack.pop(0)
    img1 = img_top_stack[0]        ; img2 = img_bot_stack[0]

    cv2.imshow('Real_time', cv2.cvtColor(img_top_stack[5],cv2.COLOR_RGB2BGR))

    shape = (img1.shape[0], img1.shape[1])
    mask1 = exgr_mask(img1, th = 0.10)
    mask2 = exgr_mask(img2, th = 0.10)

    white = np.full((shape[0], shape[1]),255, np.uint8)
    black = np.full((shape[0], shape[1]),0, np.uint8)
    line1 = black.copy()
    line2 = black.copy()

    for i in range(len(current_state)):
        if current_state[i] != button_state[i]:
            # Button state has changed
            if current_state[4] == 1:
                rows[0] += 0.5
                print(f"Row {np.abs(rows[0]*2 - 2)} Selected")
            if current_state[5] == 1:
                rows[0] -= 0.5
                print(f"Row {np.abs(rows[0]*2 - 2)} Selected")

            # Update the button state
            button_state[i] = current_state[i]
    

    x_pose = robot_pose.pose.position.x
    (r,p,yaw) = tf_conversions.transformations.euler_from_quaternion([
        robot_pose.pose.orientation.x,robot_pose.pose.orientation.y,robot_pose.pose.orientation.z,robot_pose.pose.orientation.w])
    l_s = x_pose - 2
    l_f = x_pose + 2
    x0_bot = 0
    x1_bot = 0
    x0_top = 0
    x1_top = 0
    x_hor = 0

    for idx in range(len(rows)):
        x0_top,x1_top,_ = line_from_tf2((x_pose - 2,rows[idx]),(x_pose + 2,rows[idx]),camera_frame='camera_link')
        pts = np.array([(x0_top-65 - int(np.abs(np.sin(yaw))*20), 0),
                        (x0_top+65 + int(np.abs(np.sin(yaw))*20), 0),
                        (x1_top+90 + int(np.abs(np.sin(yaw))*35), 480),
                        (x1_top-90 - int(np.abs(np.sin(yaw))*35), 480)])
        cv2.fillPoly(line1, [pts], (255))

        if np.abs(yaw) < 0.6:
            x0_bot,x1_bot,x_hor = line_from_tf2((x_pose+0.7,rows[idx]),(x_pose+5,rows[idx]+np.sin(yaw)/2),camera_frame='camera_link2')#,img = img2)
            pts = np.array([(x_hor, 315+1), (x_hor-1, 315), (x1_bot+230, 480), (x1_bot-230, 480)])
            cv2.fillPoly(line2, [pts], (255))

        elif np.abs(yaw) > 2.5:
            x0_bot,x1_bot,x_hor = line_from_tf2((x_pose-0.7,rows[idx]),(x_pose-5,rows[idx]+np.sin(yaw)/2),camera_frame='camera_link2')#,img = img2) #
            pts = np.array([(x_hor, 315+1), (x_hor-1, 315), (x1_bot+230, 480), (x1_bot-230, 480)])
            cv2.fillPoly(line2, [pts], (255))
        else:
            x0_bot = x1_bot = -1000

    #Imagem de cima -----------------------------------------------------------
    soil1 = np.where(mask1 == 0, white, 0)
    soil1 = np.where(line1 == 0, soil1, 0)
    line1 = np.where(mask1 == 0, line1, 0)
    mask_comb1 = cv2.merge((soil1,mask1,line1)) #BGR
    mask_clss1 = to_class(mask_comb1)
    show1 = cv2.addWeighted(img1,0.5,mask_comb1,0.5,0)

    cv2.line(show1 ,(int(x0_top),int(0)),(int(x1_top),int(480)),(255,255,255),2)
    lbl1 = str(int(x0_top))+','+str(int(x1_top))
    img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)

    #cv2.imshow('top_img', img1)
    #cv2.imshow('top_mask', mask_comb1)
    #cv2.imshow('top_class', mask_clss1*80)
    cv2.imshow('top', show1)

    #Imagem de baixo ----------------------------------------------------------
    soil2 = np.where(mask2 == 0, white, 0)
    soil2[320:,:] = np.where(line2[320:,:] == 0, soil2[320:,:], 0)
    line2[0:320,:] = white[0:320,:]
    line2 = np.where(mask2 == 0, line2, 0)
    mask_comb2 = cv2.merge((soil2,mask2,line2)) #BGR
    mask_clss2 = to_class(mask_comb2)
    show2 = cv2.addWeighted(img2,0.5,mask_comb2,0.5,0)

    cv2.line(show2 ,(int(x0_bot),int(0)),(int(x1_bot),int(480)),(255,255,255),2)
    lbl2 = str(int(x0_bot))+','+str(int(x1_bot))
    img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)

    #cv2.imshow('bot_img', img2)
    #cv2.imshow('bot_mask', mask_comb2)
    #cv2.imshow('bot_class', mask_clss2*80)
    cv2.imshow('bot', show2)

    time_now = time.time()
    if time_now - last_save > 2:
        print('image:',str(time.time())[4:10],'top:', lbl1, 'bot:', lbl2)
        #last_save = save_data(img1,mask_comb1,mask_clss1,lbl1,'soybot_sim/',im_id = str(time.time())[4:10]+'_top')
        #last_save = save_data(img2,mask_comb2,mask_clss2,lbl2,'sim_tunnel/',im_id = str(time.time())[4:10]+'_bot')
        last_save = time.time()
        pass

    return
    

#Main Controller
def basic_control():
    global cmd
    w=0
    vel = Twist()

    while not rospy.is_shutdown():
        vel.angular.z = cmd.angular.z*0.4
        vel.linear.x = cmd.linear.x*0.6

        if top_image is not None:
            start = time_ns()

            show_display()
            cv2.waitKey(1)

            stop = time_ns()

        pub.publish(vel)
        rate.sleep()

        time_used = ((stop-start)/1000000)

        #print(f'{time_used:.4}')

if __name__ == '__main__':

    pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/nav_front/image_raw', Image , callback)
    rospy.Subscriber('/soybot/left_camera1/image_raw/', Image , callback2)
    rospy.Subscriber('/soybot_pose', PoseStamped , pose_callback)
    rospy.Subscriber('joy', Joy, callback_joy) 
    rospy.init_node('auto_labeler', anonymous=True)
    supress = suppress_TF_REPEATED_DATA()
    get_caminfo()

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rate = rospy.Rate(11) # 10hz

    try:
        basic_control()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass
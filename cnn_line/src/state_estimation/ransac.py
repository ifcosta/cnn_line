import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import time
import cv2
from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

import numpy as np

from math import atan2, sqrt, pi, hypot
import matplotlib.pyplot as plt

from sklearn import linear_model
from copy import deepcopy
import os

import time


p_left = np.zeros([3,2])

p_right = np.zeros([3,2])

p_middle = np.zeros([3,2])




def seg_image(image,show=False):
    im = deepcopy(image)
    im_to_binarize = im.astype('float')
    
    R_ = im_to_binarize[:,:,2]/np.max(im_to_binarize[:,:,2])
    G_ = im_to_binarize[:,:,1]/np.max(im_to_binarize[:,:,1])
    B_ = im_to_binarize[:,:,0]/np.max(im_to_binarize[:,:,0])
    
    r = R_/(R_+G_+B_+0.000001)
    g = G_/(R_+G_+B_+0.000001)
    b = B_/(R_+G_+B_+0.000001)
    
    excess_red = 1.4*r - g
    excess_green = 2*g - r - b
    
    #a = 2*im_to_binarize[:,:,1] - im_to_binarize[:,:,0] - im_to_binarize[:,:,2]
    #excess_red = 1.4*im_to_binarize[:,:,2] - im_to_binarize[:,:,1]
    eG_eR = excess_green - excess_red
    
    thresh2 = np.where(eG_eR > 0.4, 255.0, 0.0)

    if show:
        '''
        fig = plt.figure(figsize=(16,9))

        fig.suptitle('Segmentation using Excess Green - Excess Red')
        plt.subplot(321)
        plt.imshow(im)
        plt.subplot(322)
        plt.imshow(excess_green)
        plt.subplot(323)
        plt.imshow(excess_red)
        plt.subplot(324)
        plt.imshow(eG_eR)
        plt.subplot(325)
        #plt.hist(thresh2)
        plt.imshow(im)
        plt.subplot(326)
        plt.imshow(thresh2)
        plt.close()
        #plt.title('Threshold = 0')
        '''
    return thresh2

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

   
def morePoints_ransac(thresh,show=False, control = False, original_image = np.array([])):  

    global p_left, p_right, p_middle

    init_time = time.time()
    #init_time = time.time()
    out_image = np.zeros((thresh.shape[0],thresh.shape[1],3))
    
    #Just for plotting purpose. If it's given the original image, then use it to create lines and show it.
    if len(original_image.shape) > 0:
        out_image = original_image
    
    #Otherwise, use grayscale image to display.
    else:
        for i in range(out_image.shape[2]):
            out_image[:,:,i] = thresh
    

    #Just morphological operations. We do it as follow: Closing -> Dilate, using appropriate kernel size.
    kernel = np.ones((12,4),np.uint8)
    th1 = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations = 2)
    #th1 = cv2.morphologyEx(th1,cv2.MORPH_OPEN,kernel, iterations = 2)
    thresh = cv2.dilate(th1,kernel)

    
        
    points = []
    px_left = []
    py_left = []
    px_right = []
    py_right = []


    #Scanning every pixel column.
    for i in range(thresh.shape[0]):
        #OLD: Using only even rows. We can change for whatever value, specially for speed improvement
        #All rows
        if (i%1 == 0):
            #print('i = ', i)
            
            #This part is ommited, but we can rollback. It uses moving average to 'blur' the lines, and use a 
            #threshold to detect a better region, using different values of threshold insteand of the max value.
            #Based on [1] 
            
            #blah = moving_average(thresh1[i,:],10)
            #np.where(blah>10,1,0)
            #m = max(blah)
            
            #This also works, due to image size.
            m = max(thresh[i,:])
            
            #print('--------- \n Row {} \n----------'.format(i))
            
            #Check if there's values segmented.
            if m > 0:
                #Get all index where the max value is reached.
                index = [i1 for i1,j in enumerate(thresh[i,:]) if j == m]

                #print(index)
                #centroid = []
                
                #Now, for every index
                for k in range(len(index)):
                    
                    #Initialize index, if it's first time.
                    if k == 0:
                        initial_index = index[k]
                        final_index = index[k]

                    #Check if it's the last iteration.
                    elif k == len(index) - 1:
        
                        #If so, check if the interval's length is greater than a threshold value, maintaining only
                        #bigger images and removing 'noises'.
                        if (final_index - initial_index > 10):
                            #centroid.append((final_index + initial_index)/2)
                            #print('Final index: ',final_index)
                            #print('Initial index: ', initial_index)
                            #print('i :',i)
                            #print('--------')
                            points.append((int((final_index+initial_index)/2),(int(i))))
                            if (final_index + initial_index)/2 > thresh.shape[1]/2:
                                px_right.append((final_index + initial_index)/2)
                                py_right.append(i)
                            else:
                                px_left.append((final_index+initial_index)/2)
                                py_left.append(i)

                    #if it's not either first or last iteration, do:
                    else:
                        #Check if the distance between these indexes is lower than a manual value. If so, we 
                        #consider as pixels from the same region. This if check avoid the creation of more points
                        #in regions which can be very close, but with small distance, due to segmentation problems
                        #or even leaves distances.
                        if (index[k] - index[k-1])< 20:
                            final_index = index[k]
                            #print(final_index)
                            
                        
                        else:
                            #Else, if the interval's length is greater than a threshold value, we keep the center 
                            #value of it as our point.
                            if (final_index - initial_index > 10):
                                #centroid.append((final_index + initial_index)/2)
                                #print('Final index: ',final_index)
                                #print('Initial index: ', initial_index)
                                #print('--------')
                                points.append((int((final_index+initial_index)/2),(int(i))))
                                if (final_index + initial_index)/2 > thresh.shape[1]/2:
                                    px_right.append((final_index + initial_index)/2)
                                    py_right.append(i)
                                else:
                                    px_left.append((final_index+initial_index)/2)
                                    py_left.append(i)


                            #Reset initial and final index to this new index.
                            initial_index = index[k]
                            final_index = index[k]
                #print(centroid)
            else:
                #print('No values found')

                continue
        else:
            continue

    
    #Adjust these points for RANSAC interpolation
    cx_left = np.array([[x] for x in px_left])
    cy_left = np.array([[y] for y in py_left])

    cx_right = np.array([[x] for x in px_right])
    cy_right = np.array([[y] for y in py_right])

    #Using RANSAC to create the line

    try:
        ransac = linear_model.RANSACRegressor(min_samples=int(len(cx_left)/1.5+1),residual_threshold=10.0)
        ransac.fit(cy_left, cx_left)
        
        #Get the extreme points on image.
        point1_ransac_left = ransac.predict([[0]])
        point2_ransac_left = ransac.predict([[thresh.shape[0]]])

        if (p_left[0,:] == np.array([0,0])).all():
            p_left[0,:] = [point1_ransac_left, point2_ransac_left]
            
        elif (p_left[1,:] == np.array([0,0])).all():
            p_left[1,:] = p_left[0,:]
            p_left[0,:] = [point1_ransac_left, point2_ransac_left]
            
        else:
            p_left[2,:] = p_left[1,:]
            p_left[1,:] = p_left[0,:]
            p_left[0,:] = [point1_ransac_left, point2_ransac_left]
            point1_ransac_left = np.mean(p_left[:,0])
            point2_ransac_left = np.mean(p_left[:,1])

    except:
        point1_ransac_left = p_left[np.nonzero(p_left[:,0])].mean()
        point2_ransac_left = p_left[np.nonzero(p_left[:,1])].mean()

    try:
        ransac = linear_model.RANSACRegressor(min_samples=int(len(cx_right)/1.5+1),residual_threshold=10.0)
        ransac.fit(cy_right, cx_right)

        point1_ransac_right = ransac.predict([[0]])
        point2_ransac_right = ransac.predict([[thresh.shape[0]]])

        if (p_right[0,:] == np.array([0,0])).all():
            p_right[0,:] = [point1_ransac_right, point2_ransac_right]
            
        elif (p_right[1,:] == np.array([0,0])).all():
            p_right[1,:] = p_right[0,:]
            p_right[0,:] = [point1_ransac_right, point2_ransac_right]
            
        else:
            p_right[2,:] = p_right[1,:]
            p_right[1,:] = p_right[0,:]
            p_right[0,:] = [point1_ransac_right, point2_ransac_right]

            point1_ransac_right = np.mean(p_right[:,0])
            point2_ransac_right = np.mean(p_right[:,1])

    except:
        point1_ransac_right = p_right[np.nonzero(p_right[:,0])].mean()
        point2_ransac_right = p_right[np.nonzero(p_right[:,1])].mean()



    p1_diff = int((point1_ransac_left + point1_ransac_right)/2)
    p2_diff = int((point2_ransac_left + point2_ransac_right)/2)
    

    #Just error metrics. Lateral offset and heading error. May be used further for control.
    lateral_offset = (p1_diff + p2_diff)/2 - thresh.shape[1]/2
    
    angle = atan2(thresh.shape[0],(p2_diff-p1_diff))
    heading_error = pi/2 - angle
    
    theta = round(atan2(lateral_offset,thresh.shape[0]/2),2)

    #Visualization of center line. May comment if you dont want to plot and only run it.
    output = cv2.line(out_image,(int(p1_diff * original_image.shape[1] / thresh.shape[1]),0),(int(p2_diff * original_image.shape[1] / thresh.shape[1]),original_image.shape[0]),(0,0,0),5)

        
    #Visualization of side lines. May comment if you dont want to plot and only run it.
    output = cv2.line(out_image,(int(point1_ransac_left * original_image.shape[1] / thresh.shape[1]),0),(int(point2_ransac_left * original_image.shape[1] / thresh.shape[1]),original_image.shape[0]),128,5)
    output = cv2.line(out_image,(int(point1_ransac_right * original_image.shape[1] / thresh.shape[1]),0),(int(point2_ransac_right * original_image.shape[1] / thresh.shape[1]),original_image.shape[0]),128,5)


    #print('Elapsed time: {} sec.'.format(time.time() - init_time))

    if control:
        
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # bottomLeftCornerOfText = (10,500)
        # fontScale = 100
        # fontColor = (255,255,255)
        # lineType = 2
        # cv2.putText(output,str(theta),bottomLeftCornerOfText,font,fontScale,fontColor,lineType)


        #cv2.imshow('vector', vector)
        #o1 = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        #cv2.imshow('frame',o1)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    quit

        
        return theta, (p1_diff - 0.5*thresh.shape[1])/thresh.shape[1], output
    
    elif show:
        cv2.imshow('frame',output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

        return output
    else:
        return None

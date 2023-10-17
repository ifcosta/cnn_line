
import cv2
import numpy as np

def get_centroid(mask1, cv_output):
    # Image process method for feature extration

    
    mask_out = ~mask1

    # Fiding mask contours
    contour = cv2.findContours(mask_out, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[1]
    
    # Contours parameters
    area_1 = 0
    area_2 = 0
    moment_1 = []
    moment_2 = []
    Cont_1 = []
    Cont_2 = []
    centroid_1 = [0,0]

    # Fiding great area in the mask
    for c in contour:
        M = cv2.moments(c)
        if (M["m00"] > area_2):
            if (M["m00"] > area_1):
                area_2 = area_1
                moment_2 = moment_1
                moment_1 = M
                
                area_1 = M["m00"]
                Cont_2 = Cont_1
                Cont_1 = [c]
            else:
                area_2 = M["m00"]
                moment_2 = M
                Cont_2 = [c]
    #print area_1
    if area_1 > 1000: # 1000:
        centroid_1[0] = int(moment_1["m10"]/moment_1["m00"])
        centroid_1[1] = int(moment_1["m01"]/moment_1["m00"])
        cv2.circle(cv_output, (centroid_1[0], centroid_1[1]), 7, (255,0,0),-1)
        cv2.drawContours(cv_output, Cont_1 ,-1,(0,255,0),2)
    else:
        centroid_1[0] = 0
        centroid_1[1] = 0
    
    # Find higher point in the contour
    K = Cont_1[0][0]
    s = max(K, key=lambda item: (-item[0], item[1]))
    cv2.circle(cv_output, (s[0], s[1]), 7, (255,0,0),-1)
    cv2.line(cv_output, (320,0),(320,480), (255,0,0),2)

    #print(Cont_1)



    mask_show = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
    mask_show2 = cv2.fillPoly(np.zeros_like(mask_show), pts = Cont_1, color=(255,255,255))
    
    return cv_output, mask_show, centroid_1, mask_show2




#
#     _,contours,hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#        # find largest area contour
#        max_area = -1
#        for i in range(len(contours)):
#            area = cv2.contourArea(contours[i])
#            if area>max_area:
#                cnt = contours[i]
#                max_area = area

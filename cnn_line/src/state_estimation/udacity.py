
import cv2
import numpy as np

#from flexbe_core import EventState, Logger

from math import atan2

#UDACITY
def adjustPerspective(img,toBird = True):
    # TALVEZ PRECISA MELHORAR
    img_size = (img.shape[1],img.shape[0])
    src = np.float32([[0.14*img.shape[1],img.shape[0]],[0.86*img.shape[1],img.shape[0]],[0.784*img.shape[1],0],[0.216*img.shape[1],0]])
    dst = np.float32([[0.14*img.shape[1],img.shape[0]],[0.86*img.shape[1],img.shape[0]],[0.86*img.shape[1],0],[0.14*img.shape[1],0]])

    if(toBird == True):
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

ym_per_pix = 1
xm_per_pix = 1

left_fit = np.array([])
right_fit = np.array([])

windows_centers = np.array([[]])

def findLines(image, mask, nwindows=9, margin=120, minpix=80, order = 2):
    """
    Find the polynomial representation of the lines in the `image` using:
    - `nwindows` as the number of windows.
    - `margin` as the windows margin.
    - `minpix` as minimum number of pixes found to recenter the window.
    - `ym_per_pix` meters per pixel on Y.
    - `xm_per_pix` meters per pixels on X.
    
    Returns (left_fit, right_fit, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)
    """    
    # Make a binary and transform image
    binary_warped = adjustPerspective(mask)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    leftx_base = int(0.25 * mask.shape[1])
    rightx_base = int(0.75 * mask.shape[1])

    # Set height of windows
    window_height = int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    #margin = 150
    global windows_centers
    #Logger.log(windows_centers, Logger.REPORT_INFO)
    if(windows_centers.shape[1] != nwindows):
        windows_centers = np.array([[leftx_base] * nwindows, [rightx_base] * nwindows])
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        if window != 0:
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
        else:
            win_xleft_low = leftx_current - (margin + 30)
            win_xleft_high = leftx_current + (margin + 30)
            win_xright_low = rightx_current - (margin + 30)
            win_xright_high = rightx_current + (margin + 30)
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            if(window == 0):
                weight_left = -abs(good_left_inds - leftx_current)
                mean_left = np.average(nonzerox[good_left_inds], weights = weight_left)                
                leftx_current = int(mean_left)
            else:
                weight_left = -abs(good_left_inds - leftx_current)
                mean_left = np.average(nonzerox[good_left_inds], weights = weight_left)                
                leftx_current = int(mean_left)
        if len(good_right_inds) > minpix:        
            if(window == 0):
                weight_right = -abs(good_right_inds - rightx_current)
                mean_right = np.average(nonzerox[good_right_inds], weights = weight_right)                
                rightx_current = int(mean_right)
            else:
                weight_right = -abs(good_right_inds - rightx_current)
                mean_right = np.average(nonzerox[good_right_inds], weights = weight_right)
                rightx_current = int(mean_right)

    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fail safe for when no good points exists inside the windows (by: Igor), can be improved
    if len(leftx)<1 or len(rightx)<1:
        leftx  = np.full(nwindows,int(0.25 * mask.shape[1]))
        lefty  = np.arange(0,mask.shape[0], mask.shape[0]/nwindows, dtype=np.uint8)
        rightx = np.full(nwindows,int(0.75 * mask.shape[1]))
        righty = np.arange(0,mask.shape[0], mask.shape[0]/nwindows, dtype=np.uint8)

    # Fit a second order polynomial to each
    global left_fit, right_fit
    if(left_fit.shape[0] == 0):
        left_fit = np.polyfit(lefty, leftx, order)
        right_fit = np.polyfit(righty, rightx, order)
    else:
        left_fit2 = np.polyfit(lefty, leftx, order)
        right_fit2 = np.polyfit(righty, rightx, order)
        for i in range(len(left_fit)):
            left_fit[i] = left_fit[i] * 0.9 + left_fit2[i] * 0.1
            right_fit[i] = right_fit[i] * 0.9 + right_fit2[i] * 0.1
    
    # Fit a second order polynomial to each
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, order)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, order)
    
    return (left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)

def visualizeLanes(image, mask,order = 2):
    """
    Visualize the windows and fitted lines for `image`.
    Returns (`left_fit` and `right_fit`)
    """
    left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy = findLines(image, mask, order = order)
    # Visualization
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    out_img = out_img.astype(np.uint8)

    return left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy

def drawLine(img, left_fit, right_fit, order = 2):
    """
    Draw the lane lines on the image `img` using the poly `left_fit` and `right_fit`.
    """
    yMax = img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(img).astype(np.uint8)
    
    # Calculate points.
    left_fitx = np.asarray(calculateline(left_fit, ploty, order),dtype=np.uint32)
    right_fitx = np.asarray(calculateline(right_fit, ploty, order),dtype=np.uint32)
    for pixel in range(img.shape[0]-1):
        if (left_fitx[pixel] < img.shape[1]-1) and (right_fitx[pixel] < img.shape[1]-1):
            color_warp[pixel,left_fitx[pixel]] = (0,0,255)
            color_warp[pixel,left_fitx[pixel]+1] = (0,0,255)
            color_warp[pixel,left_fitx[pixel]-1] = (0,0,255)
            color_warp[pixel,right_fitx[pixel]] = (255,0,0)
            color_warp[pixel,right_fitx[pixel]+1] = (255,0,0)
            color_warp[pixel,right_fitx[pixel]-1] = (255,0,0)

    newwarp = adjustPerspective(color_warp, toBird=False)
    return cv2.addWeighted(img, 1, newwarp, 0.8, 0)

def drawLaneOnImage(img, order):
    """
    Find and draw the lane lines on the image `img`.
    """
    left_fit, right_fit, left_fit_m, right_fit_m, _, _, _, _, _ = findLines(img, order = order)
    output = drawLine(img, left_fit, right_fit, order)
    return cv2.cvtColor( output, cv2.COLOR_BGR2RGB )

def calculateline(polynomial, yMax, order=2):
    if(order == 2):
        return polynomial[0]*yMax**2 + polynomial[1]*yMax + polynomial[2]
    elif(order == 1):
        return polynomial[0]*yMax + polynomial[1]
    else:
        assert 0, "ERROR"

def udacity_pipeline(img, mask, order = 2):
    """
    Find and draw the lane lines on the image `img`.
    """
    left_fit, right_fit, left_fit_m, right_fit_m, _, _, out_img, _, _ = visualizeLanes(img, mask, order)
    output = drawLine(img, left_fit, right_fit, order)
    
    # Calculate vehicle center
    xMax = mask.shape[1]*xm_per_pix
    yMax = mask.shape[0]*ym_per_pix
    vehicleCenter = xMax / 2
    lineLeft = calculateline(left_fit, yMax, order)
    lineRight = calculateline(right_fit, yMax, order)
    lineMiddle = lineLeft + (lineRight - lineLeft)/2
    diffFromVehicle = lineMiddle - vehicleCenter

    lineLeftMiddle = calculateline(left_fit, yMax/2, order)
    lineRightMiddle = calculateline(right_fit, yMax/2, order)

    lineLeft_midpoint = calculateline(left_fit, yMax//2, order)
    lineRight_midpoint = calculateline(right_fit, yMax//2, order)
    lineMiddle_midpoint = lineLeft_midpoint + (lineRight_midpoint - lineLeft_midpoint)/2
    diffFromVehicle_midpoint = lineMiddle_midpoint - vehicleCenter
    theta = diffFromVehicle_midpoint/mask.shape[1]

    lateral_offset = (lineMiddle - 0.5*mask.shape[1])/mask.shape[1]

    return output, out_img, lateral_offset, theta
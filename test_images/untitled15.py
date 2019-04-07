#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 23:35:44 2018

@author: lvfuxiang
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
def yellow_white(img):
    img = np.copy(img)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_blured = cv2.GaussianBlur(img_HSV,(9,9),0)
    #yellow and white color selection
    lower_yellow = np.array([20,55,70])
    upper_yellow = np.array([28,255,255])
    y_thresh = [lower_yellow,upper_yellow]
    yellow = cv2.inRange(img_blured,y_thresh[0],y_thresh[1])
    yellow[yellow > 0] = 1
    lower_white = np.array([180,180,180])
    upper_white = np.array([255,255,255])
    w_thresh = [lower_white,upper_white]
    white = cv2.inRange(img_blured,w_thresh[0],w_thresh[1])
    white[white > 0] = 1
    yellow_white = yellow | white
    
    return yellow_white
'''
def yellow_white(img):
    #yellow and white color selection
    lower_yellow = np.array([10,0,0])
    upper_yellow = np.array([30,255,255])
    y_thresh = [lower_yellow,upper_yellow]
    yellow = cv2.inRange(img,y_thresh[0],y_thresh[1])
    
    lower_white = np.array([200,200,200])
    upper_white = np.array([255,255,255])
    w_thresh = [lower_white,upper_white]
    white = cv2.inRange(img,w_thresh[0],w_thresh[1])
    yellow_white = yellow | white
    
    return yellow_white

def white(img):
    #yellow and white color selection
    lower_white = np.array([200,200,200])
    upper_white = np.array([255,255,255])
    
    
    
    white = cv2.inRange(img,lower_white,upper_white)
  
    #yellow_white = cv2.bitwise_and(yellow,white)
    return white
def yellow(img):
    #yellow and white color selection
    
    lower_yellow = np.array([10,0,0])
    upper_yellow = np.array([30,255,255])
    
   
    yellow = cv2.inRange(img,lower_yellow,upper_yellow)
    #yellow_white = cv2.bitwise_and(yellow,white)
    return yellow

def saturation_threshold(img, s_thresh, sx_thresh):
    img = np.copy(img)
    img_blured = cv2.GaussianBlur(img,(9,9),0)
      
    
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img_blured, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    

	# combine the two binary
    binary = sxbinary | s_binary
    #mask = np.zeros_like(combined_binary)
    #vertices = np.array([[(100,720),(545,470),(755,470),(1100,720)]],dtype=np.int32)
    #cv2.fillPoly(mask,vertices,1)
    
    
    #vertices_o = np.array([[(200,720),(545,470),(770,470),(1120,720)]],dtype=np.int32)
    #vertices_i = np.array([[(340,720),(613,470),(714,470),(1013,720)]],dtype=np.int32)
    #cv2.fillPoly(mask,[vertices_o,vertices_i],1)
    #binary = cv2.bitwise_and(combined_binary,mask)
    return binary
'''
img = mpimg.imread('test4.jpg')
yellow_white =yellow_white(img)

#result = saturation_threshold(yellow_white, s_thresh, sx_thresh)

#plt.imshow(yellow_white)
#yellow = yellow(img)
#yellow_white = yellow | white
#img_d = np.dstack((yellow_white,yellow_white,yellow_white))
s_thresh=(170,255)
sx_thresh=(20, 100)
#t = saturation_threshold(img, s_thresh, sx_thresh)
#binary = yellow_white
#result = saturation_threshold(yellow_white, s_thresh, sx_thresh)
plt.imshow(yellow_white,cmap = 'gray')
plt.show()
print(yellow_white.shape)


#######################
'''
def find_lane_pixels_v2(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 4) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 4) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img
'''
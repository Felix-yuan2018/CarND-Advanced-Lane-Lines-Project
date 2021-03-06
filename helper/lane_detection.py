import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob


def find_lane_pixels(binary_warped):
	"""
	find lane in a binary_warped image
	input: binary_warped image
	output: left/right lane pixel poistion and a drawed search image
	"""

	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

	# plt.plot(histogram)
	# plt.show()

	# Create an output image to draw on and visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

	# Find the peak of the left and right havleve of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]//2) # 1280/2=640
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# HYPERPARAMETERS
	# Choose the number of sliding windows
	nWindows = 9
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimu number of pixels found to recenter window
	min_pix = 50
	# Set height of windows - based on nWindows above adn image shape
	window_height = np.int(binary_warped.shape[0]//nWindows)
	# Identify the x and y positions of all nonzero(i.e. activated) pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0]) # y is row, x is col
	nonzerox = np.array(nonzero[1])
	# Current postions to be updated later for each window in n_window
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	for window in range(nWindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height

		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Draw the windows on the visulization image
		cv2.rectangle(out_img, (win_xleft_low, win_y_low),
			(win_xleft_high, win_y_high), (0,255,0),2)
		cv2.rectangle(out_img, (win_xright_low, win_y_low),
			(win_xright_high, win_y_high), (0,255,0),2)
		
		# plt.imshow(out_img)
		# plt.show()

		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]	# nonzero() return a tuple, get the list for tuple
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]	# nonzero() return a tuple, get the list for tuple

		# Append these indices to hte lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# # update the window center for next round window scan
		if len(good_left_inds) > min_pix:
			leftx_current = int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > min_pix:
			rightx_current = int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices (previously was a list of list)
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
	# print(len(nonzerox))
	return (leftx, lefty, rightx, righty, out_img)

def fit_polynomial(leftx, lefty, rightx, righty, out_img):
    # Find our lane pixels first
    #leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # check if there is search failure
    if leftx.size == 0 or lefty.size == 0:
        cv2.putText(out_img,"Search failure", (50,60), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
        return out_img 
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img


def get_polynomial(leftx, lefty, rightx, righty, img_size):


    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)#多项式系数
    
    left_lane_fun = np.poly1d(left_fit) # 多项式方程
    right_lane_fun = np.poly1d(right_fit)

    ploty = np.linspace(0, img_size[0]-1, img_size[0])
    left_fitx = left_lane_fun(ploty)
    right_fitx = right_lane_fun(ploty)
    
    return left_fitx,right_fitx,ploty
    

def lane_sanity_check(left_fitx, right_fitx, ploty):
    
	'''
	1. checking that they have similar curvature margin 10%
	2. checking that they are separated by approximately right distance horizontally
	tranform calibration distence 1280/2 margin 640, 5%(610-670) is good search, 15%(545-730) is detected
	3. Checking that they are roughly parallel, check the another side if 1280/4 margin 10%
	mannully adjust the threshold bot(480-600), mid(350-500), top(100-500)
	'''
	flag = True
	lane_distance_bot = right_fitx[720] - left_fitx[720]
	lane_distance_mid = right_fitx[320] - left_fitx[320]
	lane_distance_top = right_fitx[0] - left_fitx[0]
	
	# tranform calibration distence 1280/2 is 640, 5%(610-670) is good search, 15%(545-730) is detected
	if ((lane_distance_bot < 480) or (lane_distance_bot > 600)): flag = False
	if ((lane_distance_mid < 350) or (lane_distance_mid > 500)): flag = False
	if ((lane_distance_top < 150) or (lane_distance_top > 500)): flag = False

	return flag, lane_distance_bot, lane_distance_mid, lane_distance_top

###############################################################################
def test():
	# read the image and change to binary(when write binary to RGB *255)
    binary_warped = mpimg.imread('../output_images/wraped/test6.jpg')
    binary_warped = binary_warped[:,:,0]# get 1 channel, three channel is same

    out_img = fit_polynomial(binary_warped)
    
    

    plt.imshow(out_img)
    plt.show()

def test(binary_image_file):
	# read the image and change to binary(when write binary to RGB *255)
	binary_warped = mpimg.imread(binary_image_file)
	binary_warped = binary_warped[:,:,0] / 255 # get 1 channel, three channel is same
	plt.figure(),plt.imshow(binary_warped, cmap='gray'),plt.show()
	from measure_curve import measure_curv
	img_size = (binary_warped.shape[1], binary_warped.shape[0])

	leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
	
	# plt.figure(),plt.imshow(out_img)

	if len(leftx)==0 or len(rightx) == 0:
		print("Search Failure")
		return 

	left_fitx,right_fitx,ploty = get_polynomial(leftx, lefty, rightx, righty, img_size)
	left_curverad, right_curverad = measure_curv(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700)
	flag, lane_distance_bot, lane_distance_mid, lane_distance_top = lane_sanity_check(left_fitx,right_fitx,ploty)

	cur_left = "left: {}".format(int(left_curverad))
	cur_right = "right: {}".format(int(right_curverad))
	info_str = "{}, {}, {}, {}".format(flag, int(lane_distance_bot), int(lane_distance_mid), int(lane_distance_top))
	out_img = fit_polynomial(leftx, lefty, rightx, righty, out_img)
	cv2.putText(out_img,cur_left, (50,580), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
	cv2.putText(out_img,cur_right, (50,640), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
	cv2.putText(out_img,info_str, (50,700), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
 
	plt.figure(),plt.imshow(out_img)
	plt.show()

if __name__ == '__main__':
	#test('../output_images/project/warped/test6.jpg')
	test('../examples/test6.jpg')   
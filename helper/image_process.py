import numpy as np
import cv2
import matplotlib.image as mpimg
import glob




def yellow_white_thresh(img, y_low, y_high, w_low, w_high):

	yellow_filtered = yellow_filter(img, y_low, y_high)
	yellow_filtered[yellow_filtered > 0] = 1 # transfer to binary

	white_filtered = white_filter(img, w_low, w_high) # transfer to binary
	white_filtered[white_filtered > 0] = 1

	# combine the two binary, right and left
	yellow_filtered[:,640:] = 0 # use left side of yellow filtered
	white_filtered[:,:640] = 0 # use the right side of white filtered

	# plt.figure(),plt.imshow(yellow_filtered, cmap="gray")
	# plt.figure(),plt.imshow(white_filtered, cmap="gray")
	# plt.show()

	binary = yellow_filtered | white_filtered

	return binary

def yellow_filter(image,y_low, y_high):
	"""
	filter the left side yellow line out
	"""
	image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	image_filtered = cv2.inRange(image_HSV,y_low,y_high)

	return image_filtered

def white_filter(image,w_low,w_high):
	"""
    filter the right side white line out
	"""
	image_filtered = cv2.inRange(image,w_low,w_high)

	return image_filtered

def saturation_threshold(img,s_thresh=(170,255),sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the l and s channels
    #blur = cv2.GaussianBlur(img,(9,9),0)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
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
    # combine the two binary, right and left
    #sxbinary[:,640:] = 0 # use left side of yellow filtered
    #s_binary[:,:640] = 0 # use the right side of white filtered
    
	# combine the two binary
    saturation_binary = sxbinary | s_binary  

    return saturation_binary

def binary_combined(img):
    binary1 = yellow_white_thresh(img, y_low=(10,50,0), y_high=(30,255,255), w_low=(180,180,180), w_high=(255,255,255))
    binary2 = saturation_threshold(img,s_thresh=(170,255),sx_thresh=(20, 100))  
    binary = binary1 | binary2
    binary = np.uint8(binary)	# transfer to uint8
    binary[binary > 0] = 1 # transfer to binary again
    
    return binary


def draw_lane_fit(undist, warped ,Minv, left_fitx, right_fitx, ploty):
	# Drawing
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

	# Warp the blank back to original image space using inverse perspective matrix(Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result

def draw_lane_find(undist, warped, Minv, leftx, lefty, rightx, righty):
	
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	color_warp[lefty, leftx] = [255, 0, 0]
	color_warp[righty, rightx] = [0 ,0 , 255]

	newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result

def test_thresh_images(src, dst):
    """
    apply the thresh to images in a src folder and output to dst foler
    """
    image_files = glob.glob(src+"*.jpg")
    for idx, file in enumerate(image_files):
        print(file)
        img = mpimg.imread(file)
        image_threshed = binary_combined(img)
        file_name = file.split("/")[-1]
        print(file_name)
        out_image = dst+file_name
        print(out_image)
        # convert  binary to RGB, *255, to visiual, 1 will not visual after write to file
        image_threshed = image_threshed*255
        #image_threshed = cv2.cvtColor(image_threshed, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(out_image, image_threshed)
if __name__ == "__main__":
	# these lib just use in test() functions
	
	test_thresh_images("../test_images/", "../output_images/image_process/")
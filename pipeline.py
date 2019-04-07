# the image process pipeline for the lane detection.

# import the libary
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pickle
import glob

print("Import the functions used in pipeline...")
from helper.image_process import binary_combined,draw_lane_find,draw_lane_fit
from helper.lane_detection import find_lane_pixels, get_polynomial, fit_polynomial,lane_sanity_check
from helper.measure_curve import measure_curv, measure_offset



# class to keep left/right line
class Line():
	def __init__(self):
		# was the line detected in the last iteration?
		self.detected = False
		# x values of the last n fits of the line
		self.recent_xfitted = []
		# average x values of the fitted line over the last n iterations
		self.bestx = None
		# polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]
		# radius of curvature of the line in some units
		self.radius_of_curvature = []
		# distance in meters of vechicle center from the line
		self.line_base_pos = []
		# difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float')
		# x values for detected line pixels
		self.allx = []
		# y values for detected line pixels
		self.ally = []



class Pipeline():
    '''
    pipeline class with has left/right line class to hold the related information
    '''
    # Get cal and tranform parameter
    print("Import the camera calbration parameter & perspective tranform parameter...")
    pickle_file = open("./helper/perspective_matrix.pk", "rb")
    M_pickle = pickle.load(pickle_file)
    M = M_pickle["M"]
    Minv = M_pickle["Minv"]

    pickle_file = open("./camera_cal/camera_cal.pk", "rb")
    dist_pickle = pickle.load(pickle_file)
    mtx = dist_pickle["mtx"]  
    dist = dist_pickle["dist"]

    def __init__(self, left, right):
        '''
        initial with left, right line class
        '''
        self.left = left 	# the Line() class to keep the left line info
        self.right = right 	# the Line() class to keep the right line info
        self.image_counter = 0 		# the image counter
        self.fit_fail_counter = 0	# this is used record the fit lane not meet the requirement
        self.fit_ok = False		# flag use to record fit lane is ok or not
        self.search_fail_counter = 0 # this is used to record lane pixel search failure.
        self.search_ok = False	# flag use to record lane search is ok or not
        self.smooth_number = 15 # use to average the radius valude to let the screen number not so jump
        self.debug_window = False # Option if turn on the debug window in the pipeline
        self.radius = [] # store the radius data, arverage of left/right lane cur
        self.offset = [] # store the car center offset from lane center
        self.quick_search = False # not implement, use last time fit line to quick search the lane points
        self.search_method = 1 # not implement 1/slid window, 2/convelustional, 
	
    def project_debug_window(self, image_warped, left_curverad, right_curverad, lane_check_result, search_result):
        """
        return a 360*640 debug window
        """
        # unpack the parameters
        leftx, lefty, rightx, righty, out_img = search_result
        detected,lane_distance_bot, lane_distance_mid, lane_distance_top = lane_check_result
				
        # check if search is OK
        if self.search_ok == True:		
            # the right-up corner window for debug.
            leftx, lefty, rightx, righty, out_img = search_result
            fit_image = fit_polynomial(leftx, lefty, rightx, righty, out_img)
			
            # santiy check window			
            cur_left = "left: {}".format(int(left_curverad))
            cur_right = "right: {}".format(int(right_curverad))
            #info_str = "{}".format(flag)

            cv2.putText(fit_image,cur_left, (50,580), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
            cv2.putText(fit_image,cur_right, (50,640), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
            #cv2.putText(fit_image,info_str, (50,700), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
			
            # set the debug window size
            fit_image_resize = cv2.resize(fit_image, (640, 360))
            return fit_image_resize

        else:	# if search failure, just put "search fail" text on img and return the img
            cv2.putText(out_img,'search fail', (50,50), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
            out_img_resize = cv2.resize(out_img, (640, 360))
            return out_img_resize

    def store_search(self, leftx, lefty, rightx, righty):
        """
        update the search result
        """
        self.left.allx.append(leftx)
        self.left.ally.append(lefty)
        self.right.allx.append(rightx)
        self.right.ally.append(righty)

    def get_recent_search(self):
        """
        output recent search result
        """
        leftx = self.left.allx[-1]
        lefty = self.left.ally[-1]
        rightx = self.right.allx[-1]
        righty = self.right.ally[-1]

        return leftx, lefty, rightx, righty

    def store_fit(self, left_fitx, right_fitx):

        self.left.recent_xfitted.append(left_fitx)
        self.right.recent_xfitted.append(right_fitx)

    def get_recent_fit(self):
        left_fitx = self.left.recent_xfitted[-1]
        right_fitx = self.right.recent_xfitted[-1]

        return left_fitx, right_fitx

    def project_fit_lane_info(self, image, color=(255,97,0)):
        """
        project the fited lane information to the image
        use last 15 frame average data to avoid the number quick jump on screen.
        """
        offset = np.mean(self.offset[-15:-1]) if len(self.offset) > self.smooth_number else np.mean(self.offset)
        curverad = np.mean(self.radius[-15:-1]) if len(self.radius) > self.smooth_number else np.mean(self.radius)
        direction = "right" if offset < 0 else "left"
        str_cur = "Radius of Curvature = {}(m)".format(int(curverad))
        str_offset = "Vehicle is {0:.2f}m ".format(abs(offset)) + "{} of center".format(direction)
        cv2.putText(image, str_cur, (60,100), cv2.FONT_HERSHEY_SIMPLEX,2,color,4)
        cv2.putText(image, str_offset, (60,160), cv2.FONT_HERSHEY_SIMPLEX,2,color,4)

    
    def pipeline(self, image):
        '''
		 # pipeline for image processing
		 '''	
		 # counter the image
        self.image_counter += 1
		
        img_size = (image.shape[1], image.shape[0])

		 # distort the image
        image_undist = cv2.undistort(image, Pipeline.mtx, Pipeline.dist, None, Pipeline.mtx)
        # apply the image threshhold
        image_threshed = binary_combined(image_undist)
    
		# apply the view_perspective tranform
        image_warped = cv2.warpPerspective(image_threshed, Pipeline.M, img_size, flags=cv2.INTER_LINEAR)
        
		# find lane pixels
        search_result = find_lane_pixels(image_warped)
        leftx, lefty, rightx, righty, out_img = search_result

		# check the pixels search result, if the leftx or lefty is empyt, use recent data, if there is no recent data, return the image it self
        if leftx.size == 0 or rightx.size == 0:
            self.search_ok = False
            self.search_fail_counter += 1
			
            if self.left.allx == []:
                return image # logical choise, only happend first frame search failed, or first n frame search failed
            else:	# use recent search result
                leftx, lefty, rightx, righty = self.get_recent_search()
        else: # store the search result
            self.search_ok = True
            self.store_search(leftx, lefty, rightx, righty)

		# get the polynomial line points
        left_fitx,right_fitx,ploty = get_polynomial(leftx,lefty,rightx,righty,img_size)

		# sanity Check
        lane_check_result = lane_sanity_check(left_fitx,right_fitx,ploty)
        self.fit_ok,lane_distance_bot, lane_distance_mid, lane_distance_top = lane_check_result= lane_check_result

        if self.fit_ok:
			# store the fit result
            self.store_fit(left_fitx, right_fitx)
		
        elif self.left.recent_xfitted == []:	# if there is no good fit, just skip and use what ever detect
            self.fit_fail_counter += 1

        else:	# use previous data
            self.fit_fail_counter += 1
			# use the recent_xfitted
            left_fitx, right_fitx = self.get_recent_fit()
			# use the recent pionts
            leftx, lefty, rightx, righty = self.get_recent_search()

		# measure the curverad and offset
        left_curverad, right_curverad = measure_curv(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700)
        curverad = (left_curverad + right_curverad) / 2
        offset = measure_offset(leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix=3.7/700)

		# store the lane data for furthre caculation
        self.radius.append(curverad)
        self.offset.append(offset)

		# draw the lane to the undist image
        if self.fit_ok:	# 
            result = draw_lane_find(image_undist, image_warped, Pipeline.Minv, leftx, lefty, rightx, righty)
            result = draw_lane_fit(result, image_warped, Pipeline.Minv, left_fitx, right_fitx, ploty)
        else:	# if the fit not OK, just draw the last fit line
            result = draw_lane_fit(image_undist, image_warped, Pipeline.Minv, left_fitx, right_fitx, ploty)

		# write curverad and offset on to result image
        self.project_fit_lane_info(result, color=(255,97,0))

        if self.debug_window:
            debug_window = self.project_debug_window(image_warped, left_curverad, right_curverad, lane_check_result, search_result)
            result[:360,640:]=debug_window

			# write the fit_failure image for ananlysis
            if (not self.fit_ok) or (not self.search_ok):
                fileName = "output_video/temp/temp_image/"+str(self.image_counter)+".jpg"
                write_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fileName, write_img)
	
        return result

##############################################################################
'''
def image_test_tracker(image_file,video, debug_window=False):
    '''
    #test the pipeline in one picture and show the result
'''
    image = mpimg.imread(image_file, video)

    left = Line()
    right = Line()
    pipeline = Pipeline(left, right)

    # checkif debug_window if turn on
    pipeline.debug_window = True if debug_window else False
    result = pipeline.pipeline(image)
    print("processed", pipeline.image_counter, "images")
    print("fit_fail Failure: ", pipeline.fit_fail_counter)
    print("Search Failure: ", pipeline.search_fail_counter)

    plt.imshow(result)
    plt.show()

    f, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.imshow(image), ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(result), ax2.set_title('Processed Image', fontsize=15)
    plt.show()
'''
def images_test_tracker(src, dst, video, debug_window=False):
    '''
    test the pipeline on src folder's images 
    write the result to the dst folder
    '''
    # create pipeline instance
    left = Line()
    right = Line()
    pipeline = Pipeline(left, right)
	
    # checkif debug_window if turn on
    pipeline.debug_window = True if debug_window else False
    image_files = glob.glob(src+"*.jpg")
    for idx, file in enumerate(image_files):
        print("handle on: ", file)
        img = mpimg.imread(file)
        result = pipeline.pipeline(img)		
        file_name = file.split("/")[-1]
        out_image = dst+file_name
        image_dist = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_image,image_dist)
		
    print("processed", pipeline.image_counter, "images")
    print("fit_fail Failure: ", pipeline.fit_fail_counter)
    print("Search Failure: ", pipeline.search_fail_counter)
    print("write the processed image to: ",dst)

##############################################################################
if __name__ == '__main__':
    """
    #image_test_tracker(), test pipeline on one image and show the image on screen
    images_test_tracker(), test pipeline on images and write the result to related folder
    """
    images_test_tracker("test_images/", "output_images/Lane_fit/", "project", debug_window=False)
	
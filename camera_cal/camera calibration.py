# Camera calibration file, include two function.
# main(), do the camera cal and write the result to a pickle file for further use
# test(), check whether the pickle file and the calibration parameters work
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

#camera calibrate function 
"""
mtx-represents 3D to 2D transformation 
dist-represents undistortion coef
rvecs-the spin of camera
tvecs-the offset of the camera in the real world
"""
def camera_cal(drawncorners=True):
    
    # set nx,ny based chessboard picture
    nx = 9  # the number of inner corners in the x direction of chessboard 
    ny = 6  # the number of inner corners in the y direction of chessboard 
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32) # Construct a matrix of 54 rows and 3 columns
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # turns arrays into grid order 

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./calibration*.jpg')
    print("Reading the calibration file...")
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        
        img = cv2.imread(fname)
        
        #convert image to gray that the 'cv2.findChessboardCorners' needed
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Finding chessboard corners (for an 9×️6 board)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
        print('image:',fname,'ret = ',ret)
        
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
                     
            img = cv2.drawChessboardCorners(img, (nx,ny), corners,ret)

            if drawncorners:
                write_name = 'corners_found'+str(idx)+'.jpg'
                cv2.imwrite(write_name, img)
                #cv2.imshow('img', img)
                #cv2.waitKey(500)
                plt.figure(figsize = (8,8))
                plt.imshow(img)
                plt.show()
            
    #cv2.destroyAllWindows()
            
    # Get image size
    img_size = (img.shape[1],img.shape[0])  # same as img.shape[::-1]
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Store the camera calibration pameraters in pickle file (rvecs / tvecs not used in this project)
    print("Saving the parameter to file...>>camera_cal.pk")
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle_file = open("camera_cal.pk", "wb")
    pickle.dump(dist_pickle, pickle_file)
    pickle_file.close()
   
def test():
	"""
	read the pickle file on disk and implement undistor on image
	show the oringal/undistort image
	"""
	print("Reading the pickle file...")
	pickle_file = open("./camera_cal.pk", "rb")
	dist_pickle = pickle.load(pickle_file)
	mtx = dist_pickle["mtx"]  
	dist = dist_pickle["dist"]
	pickle_file.close()

	print("Reading the sample image...")
	img = mpimg.imread('./corners_found2.jpg')
	img_size = (img.shape[1],img.shape[0])
	dst = cv2.undistort(img, mtx, dist, None, mtx)

	# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
	# Visualize undistortion
	print("Visulize the result...")
	f, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
	ax1.imshow(img), ax1.set_title('Original Image', fontsize=15)
	ax2.imshow(dst), ax2.set_title('Undistored Image', fontsize=15)
	plt.show()

if __name__ == '__main__':
	camera_cal(drawncorners=True)	# read the chessboard file and get mtx, dist and write to pickle fiel
	# calibrate(drawconer=True)
	test()	# read the pickle file and undistort an image.


    
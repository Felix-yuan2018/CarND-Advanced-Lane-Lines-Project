import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#cal_undistort function that takes an image, and camarea parameters(mtx,dist)
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(src, dst,plot=False):
    
    # Read in the saved objpoints and imgpoints
    print("Reading the pickle file...")
    pickle_file = open("../camera_cal/camera_cal.pk", "rb")
    dist_pickle = pickle.load(pickle_file)
    mtx = dist_pickle["mtx"]  
    dist = dist_pickle["dist"]
    pickle_file.close()
    
    # Read in an image
    image_files = glob.glob(src+"*.jpg")
    for idx, file in enumerate(image_files):
        print("handle on: ", file)
        img = mpimg.imread(file)
        # Use cv2.undistort() to distortion correction
        undist = cv2.undistort(img,mtx,dist,None,mtx)
        undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
        file_name = file.split("/")[-1]
        out_image = dst+file_name
        cv2.imwrite(out_image,undist)
        
    
        if plot == True:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title(file_name +'\n Original Image' , fontsize=15)
            ax2.imshow(undist)       
            ax2.set_title(file_name + '\n Undistorted Image', fontsize=15)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()

if __name__ == '__main__':
    cal_undistort("../test_images/", "../output_images/undistort/",plot=False)
    

        



    
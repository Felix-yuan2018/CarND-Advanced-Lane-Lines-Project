# handle the video use the pipeline
print("import lib...")
from moviepy.editor import VideoFileClip
import os
print("import pipeline...")
from pipeline import *

def get_video_tracker(video, subclip=False, debug_window=False):
    '''
    handle the project_video use pipeline function
    '''
    # Create pipeline instance
    print("build pipeline instance...")
    left = Line()
    right = Line()
    pipeline = Pipeline(left,right)

    # checkif debug_window if turn on
    pipeline.debug_window = True if debug_window else False
	
    # check if handle the subclip
    if subclip:
        print("test on 5 second video")
        clip = VideoFileClip("test_video/%s" %video).subclip(0,5)
    else:
        print("handle the whole video")
        clip = VideoFileClip("test_video/%s" %(video))
    white_clip = clip.fl_image(pipeline.pipeline)
    white_output = "./output_video/temp/"+ video
    white_clip.write_videofile(white_output, audio=False)

	# write the information to the consel
    print("processed {} images".format(pipeline.image_counter))
    print("Detected Failure: {}".format(pipeline.fit_fail_counter))
    print("Search Failure: {}".format(pipeline.search_fail_counter))
    print("The video is at ./output_video/temp/")

    # write the information to log file
    with open("./output_video/temp/log.txt", "w") as text_file:
        print("processed {} images".format(pipeline.image_counter), file=text_file)
        print("Detected Failure: {}".format(pipeline.fit_fail_counter), file=text_file)
        print("Search Failure: {}".format(pipeline.search_fail_counter), file=text_file)

if __name__ == "__main__":
	"""
	choise one line to uncoment and run the file, get the video.
	the video will be output to ./outpu_videos/temp/
	option: subclip = True, just use (0-5) second video, False, use total long video.
	option: debug_window = True, project the debug window on the up-right corner of the screen to visualize the image handle process
								and write the fit lane failure/search lane failure image to ./output_videos/temp/images
	"""
	

	#get_video_tracker("project_video.mp4", subclip=False, debug_window=True) 
	#get_video_tracker("project_video.mp4", subclip=False, debug_window=False)

	#get_video_tracker("challenge_video.mp4", subclip=False, debug_window=True) 
	#get_video_tracker("challenge_video.mp4", subclip=False, debug_window=False)
	
	#get_video_tracker("harder_challenge_video.mp4", subclip=False, debug_window=True)
	get_video_tracker("harder_challenge_video.mp4", subclip=False, debug_window=False)
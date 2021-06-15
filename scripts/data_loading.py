import cv2

def extract_frames(video_path,frame_dir):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(frame_dir+"/frame{:0>4d}.jpg".format(count), image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('{}\tRead a new frame: {}'.format(count,success))
        count += 1
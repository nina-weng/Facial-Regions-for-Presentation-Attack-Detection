import os
import cv2
import random
import numpy as np

def extract_frames(video_path,frame_dir,num_frames):
    vidcap = cv2.VideoCapture(video_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_choice = random.sample(list(np.arange(0,length)),num_frames)
    print(random_choice)
    success, image = vidcap.read()
    count = 0
    while success:
        if count in random_choice:
            cv2.imwrite(frame_dir+"/frame{:0>4d}.jpg".format(count), image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('{}\tRead a new frame: {}'.format(count,success))
        count += 1
    print('video from:{}\tnum of frames:{}'.format(video_path, count))
    return None


if __name__ == '__main__':

    casia_data_folder = '..\..\..\..\Casia-Face-AntiSpoofing'
    print(os.listdir(casia_data_folder))

    test_dir = casia_data_folder+'/test_release/test_release/'
    train_dir = casia_data_folder+'/train_release/train_release/'

    test_subject_number = len(os.listdir(test_dir))
    train_subject_number = len(os.listdir(train_dir))
    print('number of subjects in training set: {}\nnumber of subjects in testing set: {}'.format(train_subject_number,
                                                                                                 test_subject_number))


    for subject_id in os.listdir(test_dir):
        subject_dir = test_dir+'/{}/'.format(subject_id)
        video_list = os.listdir(subject_dir)

        for video_name in video_list:
            video_id = video_name.split('.avi')[0]
            # load video
            video_path = os.path.join(subject_dir,video_name)
            print('current video is from path: {}'.format(video_path))

            # videos to frames
            frame_train_dir = casia_data_folder+'/test_frames/'
            subject_frame_dir = os.path.join(frame_train_dir,str(subject_id))
            if os.path.exists(subject_frame_dir) == False:
                os.mkdir(subject_frame_dir)

            video_frame_dir = os.path.join(subject_frame_dir,str(video_id))
            if os.path.exists(video_frame_dir) == False:
                os.mkdir(video_frame_dir)

            extract_frames(video_path,video_frame_dir,5)

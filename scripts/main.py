import os



if __name__ == '__main__':
    """
    casia_data_folder = '..\..\..\Casia-Face-AntiSpoofing'
    print(os.listdir(casia_data_folder))

    test_dir = casia_data_folder+'/test_release/test_release/'
    train_dir = casia_data_folder+'/train_release/train_release/'

    test_subject_number = len(os.listdir(test_dir))
    train_subject_number = len(os.listdir(train_dir))
    print('number of subjects in training set: {}\nnumber of subjects in testing set: {}'.format(train_subject_number,
                                                                                                 test_subject_number))

    # choose one subject for the following steps
    subject_id = 17
    subject_dir = train_dir+'/{}/'.format(subject_id)
    video_list = os.listdir(subject_dir)

    video_id = 1

    video_path = os.path.join(subject_dir,video_list[video_id-1])
    print('current video is from path: {}'.format(video_path))

    # load in videos

    # videos to frames
    frame_train_dir = casia_data_folder+'/train_frames/'
    subject_frame_dir = os.path.join(frame_train_dir,str(subject_id))
    if os.path.exists(subject_frame_dir) == False:
        os.mkdir(subject_frame_dir)

    video_frame_dir = os.path.join(subject_frame_dir,str(video_id))
    if os.path.exists(video_frame_dir) == False:
        os.mkdir(video_frame_dir)

    extract_frames(video_path,video_frame_dir)


    for each_frame in os.list(video_frame_dir):
        # face detection
        pass
        # face alignment

        # face regions extraction
    
"""
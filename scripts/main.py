import os





if __name__ == '__main__':

    casia_data_folder = '..\..\..\Casia-Face-AntiSpoofing'
    print(os.listdir(casia_data_folder))

    test_dir = casia_data_folder+'/test_release/test_release/'
    train_dir = casia_data_folder+'/train_release/train_release/'

    test_subject_number = len(os.listdir(test_dir))
    train_subject_number = len(os.listdir(train_dir))
    print('number of subjects in training set: {}\nnumber of subjects in testing set: {}'.format(train_subject_number,
                                                                                                 test_subject_number))

    # choose one subject for the following steps
    subject_dir = train_dir+'/17/'
    video_list = os.listdir(subject_dir)

    video_path = os.path.join(subject_dir,video_list[0])
    print('current video is from path: {}'.format(video_path))

    # read in videos

    # videos to frames

    # face detection

    # face alignment

    # face regions extraction
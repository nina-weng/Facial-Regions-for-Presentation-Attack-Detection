import os
from tqdm import tqdm
import random


"""
get the train & test list as well as the labels 
"""

FRAME_INTERVAL = 20


def get_imgpath_label_txt(dirs,txt_path,num_frames=None):
    '''

    :param dirs: dirs for data
    :param txt_path: the stored txt path
    :param num_frames: is None, then choose all frames; otherwise, randomly get x frames from each video
    :return:
    '''
    f = open(txt_path, "a")
    for dir in dirs:
        video_list = os.listdir(dir)
        for video_id in tqdm(video_list):
            if video_id == '1' or video_id == '2' or video_id == 'HR_1':
                label = 0
            else: label = 1
            video_dir = os.path.join(dir,video_id)
            img_list = os.listdir(video_dir)
            if num_frames == None or len(img_list) < num_frames:
               for idx,img_fname in enumerate(img_list):
                    f.write('{},{}\n'.format(os.path.join(video_dir,img_fname),label))
            else:
                random_list= random.sample(img_list,num_frames)
                for idx, img_fname in enumerate(random_list):
                    f.write('{},{}\n'.format(os.path.join(video_dir, img_fname), label))
            # random_img = random.choice(img_list)
            # f.write('{},{}\n'.format(os.path.join(video_dir, random_img), label))

    f.close()


if __name__ == '__main__':

    # casia_data_folder = '..\..\..\..\Casia-Face-AntiSpoofing'
    casia_data_folder = 'D:\\NinaWeng\\DTU\\learning\\02238_biometric\\RPA\\Casia-Face-AntiSpoofing\\'
    frame_dir = casia_data_folder + '/train_frames/'
    train_normalized_dir = casia_data_folder + '/train_normalized/'
    test_normalized_dir = casia_data_folder + '/test_normalized/'
    region_chosen = 'face_ISOV'
    train_region_dir = casia_data_folder+'/train_face_region/'+ region_chosen
    test_region_dir = casia_data_folder + '/test_face_region/'+ region_chosen

    # temporarily we use the first 15 subjects in train set as training data, the last 5 subjects as testing data
    train_set_dirs =[os.path.join(train_region_dir,dir_) for dir_ in os.listdir(train_region_dir)]
    test_set_dirs = [os.path.join(test_region_dir,dir_) for dir_ in os.listdir(test_region_dir)]

    txt_dir = '..\..\\train_test_info'
    train_txt_fname = 'train_faceisov_20_1.txt'
    test_txt_fname='test_faceisov_30_1.txt'

    get_imgpath_label_txt(train_set_dirs,os.path.join(txt_dir,train_txt_fname),num_frames=1)
    get_imgpath_label_txt(test_set_dirs,os.path.join(txt_dir,test_txt_fname),num_frames=1)




import os
from tqdm import tqdm
import random


"""
get the train & test list as well as the labels 
"""

FRAME_INTERVAL = 20


def get_imgpath_label_txt(dirs,txt_path):
    f = open(txt_path, "a")
    for dir in dirs:
        video_list = os.listdir(dir)
        for video_id in tqdm(video_list):
            if video_id == '1' or video_id == '2' or video_id == 'HR_1':
                label = 0
            else: label = 1
            video_dir = os.path.join(dir,video_id)
            img_list = os.listdir(video_dir)
            # for idx,img_fname in enumerate(img_list):
            #     if idx%FRAME_INTERVAL == 0:
            #         f.write('{},{}\n'.format(os.path.join(video_dir,img_fname),label))
            random_img = random.choice(img_list)
            f.write('{},{}\n'.format(os.path.join(video_dir, random_img), label))

    f.close()


if __name__ == '__main__':

    # casia_data_folder = '..\..\..\..\Casia-Face-AntiSpoofing'
    casia_data_folder = 'D:\\NinaWeng\\DTU\\learning\\02238_biometric\\RPA\\Casia-Face-AntiSpoofing\\'
    frame_dir = casia_data_folder + '/train_frames/'
    normalized_dir = casia_data_folder + '/train_normalized/'
    region_dir = casia_data_folder+'/train_face_region/both_eyes/'

    # temporarily we use the first 15 subjects in train set as training data, the last 5 subjects as testing data
    train_set_dirs = [os.path.join(region_dir,str(i)) for i in range(1,16)]
    test_set_dirs = [os.path.join(region_dir, str(i)) for i in range(16, 21)]

    txt_dir = '..\..\\train_test_info'
    train_txt_fname = 'train_15_r1.txt'
    test_txt_fname='test_5_r1.txt'

    get_imgpath_label_txt(train_set_dirs,os.path.join(txt_dir,train_txt_fname))
    get_imgpath_label_txt(test_set_dirs,os.path.join(txt_dir,test_txt_fname))




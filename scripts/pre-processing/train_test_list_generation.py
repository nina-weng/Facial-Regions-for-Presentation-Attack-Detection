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
        print(dir.split('/'))
        subject_id = dir.split('/')[-1]
        video_list = os.listdir(dir)
        for video_id in tqdm(video_list):
            if video_id == '1' or video_id == '2' or video_id == 'HR_1':
                label = 0
            else: label = 1
            video_dir = os.path.join(dir,video_id)
            img_list = os.listdir(video_dir)
            if num_frames == None or len(img_list) < num_frames:
               for idx,img_fname in enumerate(img_list):
                    img_id = img_fname.split('.jpg')[0].split('normalized')[1]
                    f.write('{},{},{},{}\n'.format(subject_id,video_id,img_id,label))
            else:
                random_list= random.sample(img_list,num_frames)
                for idx, img_fname in enumerate(random_list):
                    # f.write('{},{}\n'.format(os.path.join(video_dir, img_fname), label))
                    img_id = img_fname.split('.jpg')[0].split('normalized')[1]
                    f.write('{},{},{},{}\n'.format(subject_id, video_id, img_id, label))
            # random_img = random.choice(img_list)
            # f.write('{},{}\n'.format(os.path.join(video_dir, random_img), label))

    f.close()


def get_list_from_chosen_id(chosenid_fpath,f_dir,region_type,txt_path):
    info = []

    with open(chosenid_fpath,'r') as f:
        for line in f:
            if line.endswith('\n'):
                line= line[:-1]
            contents = line.split(',')
            subject_id,video_id,img_id,label = contents[0],contents[1],contents[2],int(contents[3])
            img_new_path = '{}/{}/{}/{}{}.jpg'.format(f_dir,subject_id,video_id,region_type,img_id)
            info.append('{},{}\n'.format(img_new_path,label))

    f_txt = open(txt_path,'a')
    for l in info:
        f_txt.write(l)
    f.close()





def get_list_from_chosen_id_for_all_regions():

    # casia_data_folder = '..\..\..\..\Casia-Face-AntiSpoofing'
    casia_data_folder = '../../../../Casia-Face-AntiSpoofing'
    frame_dir = casia_data_folder + '/train_frames/'
    train_normalized_dir = casia_data_folder + '/train_normalized'
    test_normalized_dir = casia_data_folder + '/test_normalized'

    txt_dir = '../../train_test_info/numf5'
    train_chosenid_fname = 'train_chosen_id_numf5.txt'
    test_chosenid_fname = 'test_chosen_id_numf5.txt'

    train_chosenid_path = os.path.join(txt_dir, train_chosenid_fname)
    test_chosenid_path = os.path.join(txt_dir, test_chosenid_fname)


    face_region_dir = casia_data_folder+'/train_face_region_numf5/'

    for f_region in os.listdir(face_region_dir):

        region_chosen = f_region
        train_region_dir = casia_data_folder+'/train_face_region_numf5/'+ region_chosen
        test_region_dir = casia_data_folder + '/test_face_region/'+ region_chosen


        train_txt_fname = 'train_{}_20_1.txt'.format(region_chosen)
        test_txt_fname = 'test_{}_30_1.txt'.format(region_chosen)

        train_txt_path = os.path.join(txt_dir, train_txt_fname)
        test_txt_path = os.path.join(txt_dir, test_txt_fname)

        get_list_from_chosen_id(train_chosenid_path,train_region_dir,region_chosen,train_txt_path)
        get_list_from_chosen_id(test_chosenid_path, test_region_dir, region_chosen, test_txt_path)

#     region_chosen = 'normalized'
#     train_region_dir = casia_data_folder + '/train_face_region/' + region_chosen
#     test_region_dir = casia_data_folder + '/test_face_region/' + region_chosen
# #
#
#     train_txt_fname = 'train_{}_20_1.txt'.format(region_chosen)
#     test_txt_fname = 'test_{}_30_1.txt'.format(region_chosen)
#
#     train_txt_path = os.path.join(txt_dir, train_txt_fname)
#     test_txt_path = os.path.join(txt_dir, test_txt_fname)
#
#
#     get_list_from_chosen_id(train_chosenid_path, train_normalized_dir, region_chosen, train_txt_path)
#     get_list_from_chosen_id(test_chosenid_path, test_normalized_dir, region_chosen, test_txt_path)






if __name__ == '__main__':
    get_list_from_chosen_id_for_all_regions()


    #####################################################################
    # code for generate chosen_id with certain number of frames         #
    #####################################################################
    # casia_data_folder = '../../../../Casia-Face-AntiSpoofing'
    #
    # train_normalized_dir = casia_data_folder + '/train_normalized'
    # test_normalized_dir = casia_data_folder + '/test_normalized'
    #
    # txt_dir = '../../train_test_info/numf5'
    # train_chosenid_fname = 'train_chosen_id_numf5.txt'
    # test_chosenid_fname = 'test_chosen_id_numf5.txt'
    #
    # train_dirs = [train_normalized_dir+'/'+subject_id for subject_id in os.listdir(train_normalized_dir)]
    # test_dirs = [test_normalized_dir+'/'+subject_id for subject_id in os.listdir(test_normalized_dir)]
    #
    # # get the chosen_id list with the number of frame as x
    # get_imgpath_label_txt(train_dirs, os.path.join(txt_dir,train_chosenid_fname), num_frames=5)
    # get_imgpath_label_txt(test_dirs, os.path.join(txt_dir, test_chosenid_fname), num_frames=5)
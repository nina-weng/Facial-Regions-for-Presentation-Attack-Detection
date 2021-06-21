"""
for extracting images with chosen id from the original folder
"""

import os
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    source_dir = '../../../../Casia-Face-AntiSpoofing/train_face_region'
    target_dir = '../../../../Casia-Face-AntiSpoofing/train_face_region_2'

    chosen_id_txt_fpath = '../../train_test_info/train_chosen_id.txt'

    with open(chosen_id_txt_fpath,'r') as f:
        for line in tqdm(f):
            if line.endswith('\n'):
                line= line[:-1]
            contents = line.split(',')
            subject_id,video_id,img_id,label = contents[0],contents[1],contents[2],int(contents[3])

            for face_regions in os.listdir(source_dir):

                source_fr_dir = os.path.join(source_dir,face_regions)
                target_fr_dir = os.path.join(target_dir,face_regions)

                if os.path.exists(target_fr_dir) == False:
                    os.mkdir(target_fr_dir)

                source_subject_dir = os.path.join(source_fr_dir,subject_id)
                target_subject_dir = os.path.join(target_fr_dir,subject_id)

                if os.path.exists(target_subject_dir) == False:
                    os.mkdir(target_subject_dir)

                source_video_dir = os.path.join(source_subject_dir,video_id)
                target_video_dir = os.path.join(target_subject_dir,video_id)

                if os.path.exists(target_video_dir) == False:
                    os.mkdir(target_video_dir)

                source_img_path = os.path.join(source_video_dir,'{}{}.jpg'.format(face_regions,img_id))
                target_img_path =os.path.join(target_video_dir,'{}{}.jpg'.format(face_regions,img_id))

                shutil.copyfile(source_img_path, target_img_path)

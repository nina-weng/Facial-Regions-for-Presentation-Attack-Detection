import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

FACE_REGIONS_INFO ={
    'chin':{'determined_lm':[8],'determined_type':'center','bb':(75,181)},
    'left_ear':{'determined_lm':[1,2],'determined_type':'center','bb':(75,51)}, #1
    'right_ear':{'determined_lm':[14,15],'determined_type':'center','bb':(75,51)}, #15
    'left_eyebrow':{'determined_lm':[17,18,19,20,21],'determined_type':'center','bb':(51,75)},
    'right_eyebrow':{'determined_lm':[22,23,24,25,26],'determined_type':'center','bb':(51,75)},
    'both_eyebrows':{'determined_lm':[21,22],'determined_type':'center','bb':(51,151)},
    'left_eye':{'determined_lm':[36,37,38,39,40,41],'determined_type':'center','bb':(51,51)},
    'right_eye': {'determined_lm': [42,43,44,45,46,47], 'determined_type': 'center', 'bb': (51, 51)},
    'both_eyes': {'determined_lm': [39,42], 'determined_type': 'center', 'bb': (51, 151)},
    'face_ISOV': {'determined_lm': [30], 'determined_type': 'center', 'bb': (192, 168)},
    'forehead': {'determined_lm': [21,22], 'determined_type': 'bottom_center', 'bb': (101, 151)},
    'left_middle_face': {'determined_lm': [30], 'determined_type': 'right_center', 'bb': (173, 106)},
    'right_middle_face': {'determined_lm': [30], 'determined_type': 'left_center', 'bb': (173, 106)},
    'mouth': {'determined_lm': [61,62,63,65,66,67], 'determined_type': 'center', 'bb': (51, 101)},
    'nose': {'determined_lm': [29], 'determined_type': 'center', 'bb': (101, 75)},

}

# visualize the feature map for only one region
def vis_feature_map(face_region,index,data_source,region_idx=None):
    if face_region in FACE_REGIONS_INFO.keys():
        target_size = FACE_REGIONS_INFO[face_region]['bb'][::-1]
    else:  # normalized
        target_size = (400, 300)[::-1]

    fm = np.load(os.path.join(fm_dir, fname))
    # index = 202  # 33,202
    if region_idx != None:
        new_index = int(region_idx * len(fm)/num_of_regions +index)
        fm_chosen = fm[new_index][0].mean(0)
    else:
        fm_chosen = fm[index][0].mean(0)
    # fm_chosen = np.resize(fm_chosen,target_size)

    # get the img
    test_id_fpath = '../../train_test_info/' + data_source + '/test_' + face_region + '_30_1.txt'
    img_path_info = []
    with open(test_id_fpath, 'r') as f:
        for line in f:
            if line.endswith('\n'):
                line = line[:-1]
            img_path_info.append(line)

    # correspond image path
    img_path, label = img_path_info[index].split(',')
    face_region_img = cv2.imread(img_path)
    face_region_img = cv2.cvtColor(face_region_img, cv2.COLOR_BGR2RGB)

    # resize
    fm_chosen = cv2.resize(fm_chosen, target_size, interpolation=cv2.INTER_LINEAR)

    plt.figure(figsize=(9, 9))
    plt.imshow(face_region_img)
    plt.imshow(fm_chosen, alpha=0.5)
    plt.title('img_path:\n{}\nlabel:{} (0-live,1-fake)'.format(img_path.split('/')[-1], label))
    # plt.axis('off')
    plt.show()
    # cv2.waitKey(0)

    plt.figure(figsize=(9, 9))
    plt.imshow(face_region_img)
    # plt.imshow(fm_chosen, alpha=0.5)
    plt.title('img_path:\n{}\nlabel:{} (0-live,1-fake)'.format(img_path.split('/')[-1], label))
    # plt.axis('off')
    plt.show()




if __name__ == '__main__':
    data_type = 'rm_fusion'  #'rm_single'
    data_source = 'rm'      #'rm','numf1'
    fm_dir = '../../results/feature_maps/' + data_type
    # fname = 'both_eyes_20210627164935.npy'
    # fname = 'normalized_20210627164922.npy'
    # fname = 'fusion_forehead-face_ISOV_20210627175630.npy'
    fname = 'fusion_both_eyebrows-right_eye_20210627181505.npy'

    index = 166

    # for single region
    if data_type == 'rm_single' or data_type == 'single':
        face_region = fname.split('_2021')[0]
        vis_feature_map(face_region,index,data_source)


    elif data_type =='region_fusion' or data_type == 'rm_fusion':
        # get region name
        regions = fname.split('_2021')[0].split('fusion_')[1].split('-')
        num_of_regions = len(regions)
        for idx,region in enumerate(regions):
            vis_feature_map(region,index,data_source,idx)


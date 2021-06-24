import dlib
import os
from tqdm import tqdm
import cv2
import numpy as np

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



def extract_region(region_name,img,landmarks):
    bb = FACE_REGIONS_INFO[region_name]['bb']
    w = bb[1]
    h = bb[0]
    determined_type = FACE_REGIONS_INFO[region_name]['determined_type']
    determined_lm = FACE_REGIONS_INFO[region_name]['determined_lm']

    ref_center = np.mean(landmarks[determined_lm, :], axis=0).astype(int)

    if determined_type =='center':
        pass
    elif determined_type == 'bottom_center':
        ref_center[1] -= int(h/2)
    elif determined_type == 'right_center':
        ref_center[0] -= int(w/2)
    elif determined_type == 'left_center':
        ref_center[0] += int(w/2)
    else:
        raise Exception('determinted type error: {}'.format(determined_type))

    x = int(ref_center[0] - w/2)
    y = int(ref_center[1] - h/2)

    region_img = img[y:y + h, x:x + w]
    # print(x,y,w,h)
    # for i, (x, y) in enumerate(landmarks):
    #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    # cv2.circle(img, (ref_center[0], ref_center[1]), 1, (0, 255, 0), -1)
    # cv2.imshow("Output", img)
    # cv2.waitKey(0)
    return region_img



def shape2np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def landmark_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) == 0:
        isDetected = False
        print('landmark detection fail')
        return isDetected, img

    index = 0
    # if detected areas are more than 2, choose the biggest area
    if len(rects) >= 2:
        bigest_area = (rects[0].bottom() - rects[0].top()) * (rects[0].right() - rects[0].left())
        for i in range(1, len(rects)):
            area = (rects[i].bottom() - rects[i].top()) * (rects[i].right() - rects[i].left())
            if area > bigest_area:
                bigest_area = area
                index = i

    rect = rects[index]
    shape = predictor(gray, rect)
    landmarks = shape2np(shape)
    return True,landmarks

if __name__ == '__main__':

    dataset_type = 'replay-mobile'

    if dataset_type == 'Casia':
        dataset_folder = '..\..\..\..\Casia-Face-AntiSpoofing'
    elif dataset_type == 'replay-mobile':
        dataset_folder = '..\..\..\..\RM'
    else:
        raise Exception('dataset type not implemented')


    normalized_dir = dataset_folder + '/test_normalized/'

    detector = dlib.get_frontal_face_detector()
    predictor_path = '..\..\pretrained_model\shape_predictor_81_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    # store path
    face_regions_dir =  dataset_folder + '/test_face_region/'
    if os.path.exists(face_regions_dir) == False:
        os.mkdir(face_regions_dir)

    # mkdir for all regions
    for region_name in FACE_REGIONS_INFO.keys():
        region_dir = os.path.join(face_regions_dir,region_name)
        if os.path.exists(region_dir) == False:
            os.mkdir(region_dir)

    if dataset_type == 'Casia':
        for subjects in os.listdir(normalized_dir):

            subject_dir = os.path.join(normalized_dir,subjects)
            video_list = os.listdir(subject_dir)

            for region_type_chosen in FACE_REGIONS_INFO.keys():
                region_subject_dir = os.path.join(face_regions_dir, region_type_chosen, subjects)
                if os.path.exists(region_subject_dir) == False:
                    os.mkdir(region_subject_dir)


            for video_id in video_list:

                video_dir = os.path.join(subject_dir, video_id)
                face_list = os.listdir(video_dir)

                for region_type_chosen in FACE_REGIONS_INFO.keys():
                    region_subject_video_dir = os.path.join(face_regions_dir, region_type_chosen, subjects, video_id)
                    if os.path.exists(region_subject_video_dir) == False:
                        os.mkdir(region_subject_video_dir)

                print('processing frames from:{}'.format(video_dir))
                for face_fname in tqdm(face_list):
                    face_path = os.path.join(video_dir, face_fname)

                    frame_id = face_fname.split('.jpg')[0].split('normalized')[1]

                    img = cv2.imread(face_path)
                    isDetected, landmarks = landmark_detection(img)

                    if isDetected:
                        for region_type_chosen in FACE_REGIONS_INFO.keys():
                            # region_type_chosen = 'chin'
                            region_subject_video_dir = os.path.join(face_regions_dir,region_type_chosen,subjects,video_id)
                            region_img = extract_region(region_type_chosen,img,landmarks)
                            cv2.imwrite(region_subject_video_dir + "/{}{}.jpg".format(region_type_chosen,frame_id), region_img)

    elif dataset_type == 'replay-mobile':
        for ap_bp_types in os.listdir(normalized_dir):
            ap_bp_types_dir = os.path.join(normalized_dir, ap_bp_types)
            img_list = os.listdir(ap_bp_types_dir)

            for region_type_chosen in FACE_REGIONS_INFO.keys():
                region_abtype_dir = os.path.join(face_regions_dir, region_type_chosen, ap_bp_types)
                if os.path.exists(region_abtype_dir) == False:
                    os.mkdir(region_abtype_dir)

            print('processing frames from:{}'.format(ap_bp_types_dir))
            for face_fname in tqdm(img_list):
                face_path = os.path.join(ap_bp_types_dir, face_fname)

                img_info = face_fname.split('.jpg')[0].split('normalized_')[1]

                img = cv2.imread(face_path)
                isDetected, landmarks = landmark_detection(img)

                if isDetected:
                    for region_type_chosen in FACE_REGIONS_INFO.keys():
                        region_subject_video_dir = os.path.join(face_regions_dir, region_type_chosen,ap_bp_types)
                        region_img = extract_region(region_type_chosen, img, landmarks)
                        # cv2.imshow('',region_img)
                        # cv2.waitKey()
                        write_path = os.path.join(region_subject_video_dir,"{}_{}.jpg".format(region_type_chosen, img_info))
                        # print(write_path)
                        cv2.imwrite(os.path.join(region_subject_video_dir,"{}_{}.jpg".format(region_type_chosen, img_info)),
                                    region_img)



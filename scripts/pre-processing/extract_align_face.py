from mtcnn import MTCNN
from tqdm import tqdm
import math
import numpy as np
import cv2
import dlib
import os

NORMALIZED_WIDTH = 300


def get_pixel_distance(p1,p2):
    return int(np.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2))


def rect2bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape2bb(shape):
    # number_nodes= shape.shapes[0]
    left = np.min(shape[:,0])
    right = np.max(shape[:,0])
    top= np.min(shape[:,1])
    bottom = np.max(shape[:,1])

    x = left
    y= top
    w = right - left
    h = bottom - top

    # extend

    return (x,y,w,h)



def shape2np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def get_affined_landmarks(landmarks,M):
    a_landmarks = []
    for point in landmarks:
        # Convert to homogenous coordinates in np array format first so that you can pre-multiply M
        rotated_point = M.dot(np.concatenate([point,[1]]))
        a_landmarks.append([int(rotated_point[0]),int(rotated_point[1])])
    return np.array(a_landmarks)

def crop_face(face_img,affined_landmarks):
    left_eye_center = np.mean(affined_landmarks[36:42, :], axis=0)
    right_eye_center = np.mean(affined_landmarks[42:48, :], axis=0)
    eye_center = (int((right_eye_center[0] + left_eye_center[0]) * 0.5),int((right_eye_center[1] + left_eye_center[1]) * 0.5))
    x = int(eye_center[0] - NORMALIZED_WIDTH/2)
    w = NORMALIZED_WIDTH
    h = int(NORMALIZED_WIDTH/0.75)
    y = int(eye_center[1] - NORMALIZED_WIDTH*0.6-1)

    # print(left_eye_center,right_eye_center,eye_center)
    crop_img = face_img[y:y + h, x:x + w]
    return crop_img

def face_alignment(img,landmarks):
    """
    get the aligned face via landmarks, according to ISO/IEC 19794-5:2011
    :param img:
    :param landmarks: [81 * 2] landmarks
    :return: cropped & aligned face
    """
    # for i,(x, y) in enumerate(landmarks):
    #     if i<=47 and i>=36:
    #         cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    left_eye_center = np.mean(landmarks[36:42,:],axis=0).astype(int)
    right_eye_center = np.mean(landmarks[42:48,:],axis=0).astype(int)
    # print(left_eye_center,right_eye_center)
    # cv2.circle(img, (left_eye_center[0],left_eye_center[1]), 1, (0, 255, 0), -1)
    # cv2.circle(img, (right_eye_center[0],right_eye_center[1]), 1, (0, 255, 0), -1)
    # cv2.imshow("Output", img)
    # cv2.waitKey(0)
    eye_center = ((right_eye_center[0]+left_eye_center[0])*0.5,(right_eye_center[1]+left_eye_center[1])*0.5)
    print(eye_center)
    dx = (right_eye_center[0] - left_eye_center[0])
    dy = (right_eye_center[1] - left_eye_center[1])
    # compute angle
    angle = math.atan2(dy, dx) * 180. / math.pi
    print(angle)
    pupil_dis = get_pixel_distance(left_eye_center,right_eye_center)
    print('pupil_dis:{}'.format(pupil_dis))
    scale = NORMALIZED_WIDTH*0.25/pupil_dis
    print('scale:{}'.format(scale))
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)

    aligned_face = cv2.warpAffine(img.copy(), RotateMatrix, (int(img.shape[1]),int(img.shape[0])))

    landmark_affined =  get_affined_landmarks(landmarks,RotateMatrix)

    for i,(x, y) in enumerate(landmark_affined):
        cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow('',aligned_face)
    cv2.waitKey(0)

    normalized_face = crop_face(aligned_face,landmark_affined)

    cv2.imshow('', normalized_face)
    cv2.waitKey(0)


    return normalized_face




def landmark_detection(img):
    # image = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
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

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    # (x, y, w, h) = shape2bb(shape)
    # print(x,y,w,h)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # # show the face number
    # cv2.putText(image, "Face #{}".format(index+1), (x - 10, y - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # # loop over the (x, y)-coordinates for the facial landmarks
    # # and draw them on the image
    # for (x, y) in shape:
    #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    # # show the output image with the face detections + facial landmarks
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)
    return True,landmarks

if __name__ == '__main__':
    # try

    casia_data_folder = '..\..\..\..\Casia-Face-AntiSpoofing'

    # detector = MTCNN()
    detector = dlib.get_frontal_face_detector()
    predictor_path = '..\..\pretrained_model\shape_predictor_81_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    frame_dir = casia_data_folder+'/train_frames/'

    normalized_dir = casia_data_folder+'/train_normalized/'

    for subjects in os.listdir(frame_dir):
        if subjects != '1':
            continue
        subject_dir = os.path.join(frame_dir,subjects)
        video_list = os.listdir(subject_dir)

        normalized_subject_dir = os.path.join(normalized_dir,subjects)
        if os.path.exists(normalized_subject_dir)== False:
            os.mkdir(normalized_subject_dir)

        for video_id in video_list:
            if video_id != '2':
                continue
            video_dir = os.path.join(subject_dir,video_id)
            frame_list = os.listdir(video_dir)

            normalized_video_dir = os.path.join(normalized_subject_dir, video_id)
            if os.path.exists(normalized_video_dir) == False:
                os.mkdir(normalized_video_dir)

            print('processing frames from:{}'.format(video_dir))
            for frame_name in tqdm(frame_list):
                frame_path = os.path.join(video_dir,frame_name)
                img = cv2.imread(frame_path)
                # print(img.shape)

                isDetected,landmarks = landmark_detection(img)
                if isDetected:
                    normolized_face = face_alignment(img,landmarks)

                    frame_id = frame_name.split('.jpg')[0].split('frame')[1]

                    cv2.imwrite(normalized_video_dir + "/normalized{}.jpg".format(frame_id), normolized_face)



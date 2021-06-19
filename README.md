# Facial-Regions-for-Presentation-Attack-Detection
02238 course project

**Task description**: see [task-description](./RPA-task-description.pdf)



## Dataset

related paper: https://ieeexplore-ieee-org.proxy.findit.dtu.dk/stamp/stamp.jsp?tp=&arnumber=6199754

There are 50 subjects in this dataset. For each subject, there are 12 videos with low, mid, high quality. And it contains 3 kinds of attack: warped photo attack, cut photo attack, and video attack.





## Mission Completion Step-by-step

1. scripts for {video2frames, face detection, facial alignment}

   1. extract frames using cv2

      ​	store in *train_frames* folder

   2. extract normalized face image using landmarks

      * detect 81 landmarks (ref: https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)
      * align the face image first according to interpupil distance and angle (affine operation)
      * crop the image to normalized one according to ISO/IEC 19794-5

      store in *train_normalized* folder

      **crop error on video type 2,5,6**

2. scripts for extract different facial regions 

   normalized face with shape (width: 300, height: 400)

   

   | ID   | Facial Region     | landmark(s) for determining the center | bounding box |
   | :--- | ----------------- | -------------------------------------- | ------------ |
   |      |                   |                                        |              |
   | 1    | Chin              | 5                                      | 75 x 181     |
   | 2    | Left ear          | (1,2)                                  | 75 x 51      |
   | 3    | Right ear         | (14,15)                                | 75 x 51      |
   | 4    | left eyebrow      | (17-21)                                | 51 x 75      |
   | 5    | right eyebrow     | (22-26)                                | 51 x 75      |
   | 6    | both eyebrows     | (21,22)                                | 51 x 151     |
   | 7    | left eye          | (36-41)                                | 51 x 51      |
   | 8    | right eye         | (42-47)                                | 51 x 51      |
   | 9    | both eyes         | (39,42)                                | 51 x 151     |
   | 10   | Face ISOV         | (30)                                   | 192 x 168    |
   | 11   | Forehead          | (21,22) as bottom center               | 101 x 151    |
   | 12   | Left middle face  | 30 as right center                     | 173 x 106    |
   | 13   | Right middle face | 30 as left center                      | 173 x 106    |
   | 14   | Mouth             | (61-63,65-67)                          | 51 x 101     |
   | 15   | Nose              | 29                                     | 101 x 75     |

   

3. scripts for PAD approach (better not too computational consuming) 

   1. some reference ideas:
      * [use only a single frame, based on color space histogram](https://github.com/ee09115/spoofing_detection) might not be so suitable for our case
      * the easiest way: extract features (not sure what features) from frames and than use classifier.
      * we could start with for example 10 video (or even 5) for training and 5 for testing, to reduce the computational cost for now. (notice the video quality and PA type)
      * Handbook might has some clues for where to start
   2. possible ideas (steps)
      1. DL structure: resnet18 
      2. reduce data size (one from every 10 frames)
      3. only use the train set for pilot trails, 15 subjects for training & 5 for testing
      4. on normalized face & 15 regions respectively 
   3. updated to-do-list (6.18 night)
      1. extract frame and face regions for testing set (5 frames each video maybe?)
      2. run the resnet18 model and see whether it is too good
      3. whether need other dataset

4. experiment design 

   1. ideas: known attack + challenges on unknow attack types

   2. ablation study on facial regions

   3. fusion study (refer on the first reference)

      

5. analyze on the result (DET, EER etc.)






## Pilot experiment results

| model    | parameters                                                   | regions/n_face  | accuracy after 3 epochs             |
| -------- | ------------------------------------------------------------ | --------------- | ----------------------------------- |
| resnet18 | train-batch = 32, test-batch = 1, size = 256, lr = 1e-4, num_frames_perv = 1 | normalized_face | 1e: 0.944; 2e: 0.9916; 3e:0.988     |
|          |                                                              | both_eyes       | 1e: 0.5888; 2e: 0.86944; 3e: 0.8916 |
|          |                                                              | face_ISOV       | 1e: 0.8333; 2e: 0.9388; 3e: 0.9694  |
|          |                                                              |                 |                                     |
|          |                                                              |                 |                                     |





## Might-interesting Ideas

- fusion on facial regions 
- unknown attack + facial region 
- cross database, might be hard and time-consuming 

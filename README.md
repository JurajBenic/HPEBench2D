# HPEBench2D
Multi-Person 2D Human Pose Estimation: A Benchmark for Real-Time Applications


# Table of Contents
- [HPEBench2D](#hpebench2d)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Matching predicted keypoints with ground truth](#matching-predicted-keypoints-with-ground-truth)
    - [Folder and file structure](#folder-and-file-structure)
    - [Structure of json files](#structure-of-json-files)



# Introduction
Human pose estimation (HPE) is a critical component of computer vision, enabling real-time applications across various fields. This study benchmarks five state-of-the-art frameworks:
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), 
- [YOLO](https://docs.ultralytics.com/), 
- [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), 
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose),
- [Sapiens](https://github.com/facebookresearch/sapiens),
  
in the context of 2D multi-person pose estimation in videos for real-time applications. Using the Panoptic dataset and standardized metrics, including Object Keypoint Similarity (OKS), Average Precision (AP), and Average Recall (AR), we evaluate their accuracy, speed, and hardware efficiency under realistic conditions such as occlusion and crowded scenes.

Our analysis highlights the strengths and limitations of each framework, providing valuable insights to help researchers and practitioners select and deploy reliable HPE solutions in real-world applications.



# Matching predicted keypoints with ground truth
The matchPeople.py script is designed to match predicted person keypoints with ground-truth person keypoints. It supports the [Panoptic dataset](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) exclusively and requires the input results to be preformatted as specified in the documentation.
|Parameter|Description|
| :---: | :---:|
| -v | Print progress. |
| -path | Path to the folder of results. The structure of the folder has to be similar to the paper one.|
| -datasets | Names of datasets/folders to process. If left blank it will process all datasest in the given -path directory. |
| -methods | Names of methods/folders to process. If left blank it will process all methods in the given path/dataset directory.|
| -out | File to save the results. If left blank then the results will not be saved.|

    

### Folder and file structure
```
.
|───results
│   │───method1
│   |   │   method1_dataset1_camera1.json   #example: rtmo_160906_ian1_hd_00_01.json
│   |   │   method1_dataset1_camera2.json
│   |   │   method1_dataset2_camera15.json
│   |   │   method1_dataset2_camera16.json
│   |   │   ...
│   │───method2
│   |   │   method2_dataset1_camera1.json   #example: yolo_160906_ian1_hd_00_10.json
│   |   │   method2_dataset1_camera2.json
│   |   │   method1_dataset2_camera15.json
│   |   │   method1_dataset2_camera16.json
│   |   │   ...
│   └───...
│       │   ...
│   
|───dataset1
|   |   hdPose2D_camera1.json   #example: hdPose2D_00_00.json
|   │   hdPose2D_camera2.json
|   │   ...
|───dataset2
|   |   hdPose2D_camera15.json
|   │   hdPose2D_camera16.json
|   │   ...
└───...
    |   ...
```

### Structure of json files 
```
{
    "frameNumber" :{
        "bodies": [
            {
                "id": 0,    # id of the person
                "keypoints": [  # coco keypoint structure
                    [x1, y1],
                    [x2, y2],
                    ...
                    [x17, y17]
                ],
                "keypoint_scores": [2, 2, 1, 0, .. 0]
            },
        ]
    },
    ...
}
```
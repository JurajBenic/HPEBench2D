# HPEBench2D
Multi-Person 2D Human Pose Estimation: A Benchmark for Real-Time Applications


## Introduction
Human pose estimation (HPE) is a critical component of computer vision, enabling real-time applications across various fields. This study benchmarks five state-of-the-art frameworks:
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), 
- [YOLO](https://docs.ultralytics.com/), 
- [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), 
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose),
- [Sapiens](https://github.com/facebookresearch/sapiens),
  
in the context of 2D multi-person pose estimation in videos for real-time applications. Using the Panoptic dataset and standardized metrics, including Object Keypoint Similarity (OKS), Average Precision (AP), and Average Recall (AR), we evaluate their accuracy, speed, and hardware efficiency under realistic conditions such as occlusion and crowded scenes.

Our analysis highlights the strengths and limitations of each framework, providing valuable insights to help researchers and practitioners select and deploy reliable HPE solutions in real-world applications.


## Highlights

- Comprehensive benchmarking of five state-of-the-art 2D human pose estimation frameworks: OpenPose, YOLO11X, RTMO, RTMPose, and Sapiens.
- Evaluation conducted on the Panoptic dataset, ensuring diverse and realistic testing conditions.
- Detailed analysis of accuracy, speed, and hardware efficiency using standardized metrics such as Object Keypoint Similarity (OKS), Average Precision (AP), and Average Recall (AR).
- RTMO achieved the highest accuracy with a mAP of 0.61, while YOLO11X demonstrated the fastest performance at 47.8 FPS.
- Insights into the trade-offs between precision, speed, and hardware utilization for real-time applications.
- Recommendations for selecting the most suitable framework based on specific application needs and hardware constraints.


## Usage


To use your own human pose estimation model with this benchmark, follow these steps:

1. **Prepare Your Model Output**: Ensure that your model's output is formatted similarly to the example JSON structure provided in the "Structure of json files" section. Each frame should contain an array of detected bodies, each with an ID, keypoints, and keypoint scores. Also, ground truth has the same structure.

2. **Organize Your Results**: Place your model's output files in a directory structure that matches the example provided in the "Folder and file structure" section. Create a folder for your method under the `results` directory and place the JSON files for each dataset and camera within this folder.

3. **Run the Matching Script**: Use the `matchPeople.py` script to match your model's predicted keypoints with the ground truth keypoints. Execute the script with the appropriate parameters:
    ```sh
    python matchPeople.py -path /path/to/results -datasets dataset1 dataset2 -methods your_method_name -out /path/to/output/results.json
    ```
    Replace `/path/to/results` with the path to your results directory, `dataset1 dataset2` with the names of the datasets you want to process, `your_method_name` with the name of your method's folder, and `/path/to/output/results.json` with the desired output file path.

4. **Evaluate Performance**: Analyze interactively the output results using `evaluation.ipynb` to evaluate the performance of your model using the standardized metrics such as Object Keypoint Similarity (OKS), Average Precision (AP), and Average Recall (AR).

By following these steps, you can benchmark your own human pose estimation model against the provided frameworks and gain insights into its performance in real-time applications.

### Matching predicted keypoints with ground truth
The `matchPeople.py` script is designed to match predicted person keypoints with ground-truth person keypoints. It supports the [Panoptic dataset](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) exclusively and requires the input results to be preformatted as specified in the documentation.
|Parameter|Description|
| :--- | :---|
| -v | Print progress. |
| -path | Path to the folder of results. The structure of the folder has to be similar to the paper one.|
| -datasets | Names of datasets/folders to process. If left blank it will process all datasest in the given -path directory. |
| -methods | Names of methods/folders to process. If left blank it will process all methods in the given path/dataset directory.|
| -out | File to save the results. If left blank then the results will not be saved.|

### Folder and file structure
```
.
|───results
│   │───method1 #example: rtmo
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
|───dataset1    #example: 160906_ian1
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
```json
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

## Authors
## Authors
- [Juraj Benić](https://github.com/JurajBenic) - School of Applied Mathematics and Computer Science, University of Osijek
- [Tomislav Prusina](https://github.com/tomo61098) - School of Applied Mathematics and Computer Science, University of Osijek; Meet Intelligent Innovations LLC
- [Domagoj Ševerdija](https://github.com/dseverdi) - School of Applied Mathematics and Computer Science, University of Osijek; Meet Intelligent Innovations LLC
- [Domagoj Matijević](https://github.com/dmatijev) - School of Applied Mathematics and Computer Science, University of Osijek; Meet Intelligent Innovations LLC

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


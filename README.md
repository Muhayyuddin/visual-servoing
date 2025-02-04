# Vision for Navigation in Extreme Marine Conditions
Abstract:
Visual perception is an important component for autonomous navigation of unmanned surface vessels (USV), particularly for the tasks related to autonomous inspection and tracking. These tasks involve vision-based navigation techniques to identify the target for navigation. Reduced visibility under extreme weather conditions in marine environments makes it difficult for vision-based approaches to work properly. To overcome these issues, this paper presents an autonomous vision-based navigation framework for tracking target objects in extreme marine conditions. The proposed framework consists of an integrated perception pipeline that uses a generative adversarial network (GAN) to remove noise and highlight the object features before passing them to the object detector (i.e., YOLOv5). The detected visual features are then used by the USV to track the target. The proposed framework has been thoroughly tested in simulation under extremely reduced visibility due to sandstorms and fog. The results are compared with state-of-the-art de-hazing methods across the benchmarked MBZIRC simulation dataset, on which the proposed scheme has outperformed the existing methods across various metrics.

## Dataset Generation

To generate the simulated dataset for sand storm and extreme fog  first we need to install the MBZIRC simulator 

Install MBZIRC simulator by following the instructions provided here.
https://github.com/osrf/mbzirc


## Environment

Create virtual environment

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

## Dataset

We have provided the sample dataset for training, the dataset can be found in the follwing folder 
```
USV_Dataset/*   * Foggy / Sand_Storm
```
## Training

Inorder to train on the existing dataset run the following command  

```
python main.py -- data_dir USV_Dataset/*
```
Once the train will be completed, the trained weights will be stored in saved_models directory.

For testing purposes, the trained weights for sand and fog can be found in the following directories 
- Fog_Dehazed
- Sand_Dehazed
- 
## Demo
To generate results in the result folder  run 

```
python DEMO1.py 
```
The output video will be stored in the output_video directory

Please cite the article as below 
```cite
@INPROCEEDINGS{muhayy2023,
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Vision-Based Autonomous Navigation for Unmanned Surface Vessel in Extreme Marine Conditions},
  author={Ahmed, Muhayyuddin and Bakht, Ahsan Baidar and Hassan, Taimur and Akram, Waseem and Humais, Ahmed and Seneviratne, Lakmal and He, Shaoming and Lin, Defu and Hussain, Irfan},
  year={2023},
  pages={7097-7103},
  doi={10.1109/IROS55552.2023.10341867}}

```

# Vision for Navigation in Extreme Marine Conditions

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

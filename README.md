# Vision for Navigation in Extreme Marine Conditions

## Dependencies

Install MBZIRC simulator by following the instructions provided here.
https://github.com/osrf/mbzirc


Python 3.5 or 3.6 are recommended.

```
tqdm==4.19.9
numpy==1.17.3
tensorflow==1.12.0
tensorboardX
torch==1.0.0
Pillow==6.2.0
torchvision==0.2.2
```

## Environment

Create virtual environment

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

## Dataset

will be updated soon

## Training

Inorder to train the GANs on the existing dataset run the following command  

```
python3 main.py 
```
## Demo
To generate results in the result folder  run 

```
python3 DEMO1.py 
```

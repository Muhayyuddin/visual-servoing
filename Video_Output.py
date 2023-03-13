import argparse
import cv2
from model import Generator
from PIL import Image
from torch.autograd import Variable
from utils import *
import os
import numpy as np
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='Inputs/Video/Sand_Storm.mkv' , help='path to input video')#change input video name
parser.add_argument('--output', type=str,default='Output_Video/Sand_out.mkv',help='path to output video')#Change output video name
parser.add_argument('--model_path', type=str, default='*/latest_model.pt', help='path to the pre-trained model')# Select Sand_Dehazed or Fog_Dehazed
parser.add_argument('--gpu', type=int, default=0 , help='gpu index')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

my_model = Generator()
my_model.cuda()
my_model.load_state_dict(torch.load(args.model_path))
my_model.eval()
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to a multiple of 32
        scale = 32
        frame = cv2.resize(frame, (width // scale * scale, height // scale * scale))
        # Convert the frame to a tensor and move it to the GPU
        frame = rgb_to_tensor(frame)
        frame = frame.unsqueeze(0)
        frame = Variable(frame.cuda())
        # Apply the model to the frame and convert it back to an image
        with torch.no_grad():
            output = my_model(frame)
        output = tensor_to_rgb(output)
        out_frame = Image.fromarray(np.uint8(output), mode='RGB')
        # Resize the output frame to the original size and convert it to BGR
        out_frame = out_frame.resize((width, height), resample=Image.BICUBIC)
        out_frame = cv2.cvtColor(np.array(out_frame), cv2.COLOR_RGB2BGR)
        # Write the output frame to the video file
        out.write(out_frame)
        # Show the output frame
        cv2.imshow('frame',out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()

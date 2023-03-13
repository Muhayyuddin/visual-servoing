import argparse
from model import Generator
from PIL import Image
from torch.autograd import Variable
from utils import *
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='image-dehazing')

parser.add_argument('--model', default='/home/mbzirc/Downloads/AhsanBB/Pytorch-Image-Dehazing-master/results_Final/Net1/model/96/latest_model.pt', help='training directory')
parser.add_argument('--images', nargs='+', type=str, default='inputs/20.png', help='path to hazy folder')
parser.add_argument('--outdir', default='outputs', help='data save directory')
parser.add_argument('--gpu', type=int, default=0 , help='gpu index')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

my_model = Generator()
my_model.cuda()
my_model.load_state_dict(torch.load(args.model))
my_model.eval()
image = Image.open(args.images).convert('RGB')
plt.imshow(image)
width, height = image.size
scale = 32
image = image.resize((width // scale * scale, height // scale * scale))
image = rgb_to_tensor(image)
image = image.unsqueeze(0)
image = Variable(image.cuda())
with torch.no_grad():
    output = my_model(image)
output = tensor_to_rgb(output)
out = Image.fromarray(np.uint8(output), mode='RGB')
out = out.resize((width, height), resample=Image.BICUBIC)
plt.imshow(out)
directory,filename=os.path.split(args.images)
basename,extension=os.path.splitext(filename)
print(extension)
print(Name)
# output_path = os.path.join('/home/mbzirc/Downloads/AhsanBB/Pytorch-Image-Dehazing-master/outputs', f'{args.images}_output.jpg')
path=f"{args.outdir}/"+f"{basename}_Dehazed.jpg"
print(path)
out.save(f"{args.outdir}/"+f"{basename}_Dehazed.jpg")


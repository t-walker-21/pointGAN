from __future__ import print_function
from show3d_balls import *
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointGen, PointGenC
import torch.nn.functional as F
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print (opt)

gen = PointGen(num_points = 2500)
gen.load_state_dict(torch.load(opt.model))


base = 5
span = 100

sim_noise = Variable(torch.randn(base, 100))
sim_noises = Variable(torch.zeros(span * base,100))

for j in range(base):
    for i in range(span):
        x = (1-i/span)
        sim_noises[i + span * j] = sim_noise[j] * x + sim_noise[(j+1) % base] * (1-x)

points = gen(sim_noises)
point_np = points.transpose(2,1).data.numpy()
print(point_np.shape)

for i in range(base * span):
    print(i)
    frame = showpoints_frame(point_np[i])
    plt.imshow(frame)
    plt.axis('off')
    plt.savefig('%s/%04d.png' %('out_wgan', i), bbox_inches='tight')
    plt.clf()

#showpoints(point_np)

#sim_noise = Variable(torch.randn(1000, 100))
#points = gen(sim_noise)
#point_np = points.transpose(2,1).data.numpy()
#print(point_np.shape)

#np.savez('gan.npz', points = point_np)

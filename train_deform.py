"""
This script implements a deformation network that regresses per point offsets from instance to canonical

"""

from __future__ import print_function
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
from pointnet import DeformNet
import torch.nn.functional as F
from pytorch3d.loss.chamfer import chamfer_distance as criterion
import open3d as o3d
import sys
from tensorboardX import SummaryWriter
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='ae_deform',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default = 4096,  help='number of points')
parser.add_argument('--dataset', type=str, required=True, help='dataset root')
parser.add_argument('--latent_size', type=int, default=100, help='bottleneck size')
parser.add_argument('--class_choice', type=str, required=True, help='class choice')
parser.add_argument('--dont_save_model', action='store_true', help='save model progress')
parser.add_argument('--test', action='store_true', help='use test set')
parser.add_argument('--viz', action='store_true', help='visualization')
parser.add_argument('--no_rot', action='store_true', help='no rotation on points', default=False)

opt = parser.parse_args()
print (opt)


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = opt.dataset, class_choice = [opt.class_choice], canonicalize=True, npoints = opt.num_points, train=not opt.test)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

opt.outf += "_" + opt.class_choice

# Create summary logger
writer = SummaryWriter("runs/" + opt.outf + "_" + str(time.time()))

cudnn.benchmark = True

num_classes = len(dataset.classes)
print('classes', num_classes)


try:
    os.makedirs(opt.outf)
except OSError:
    pass


ae = DeformNet(num_points = opt.num_points)


if opt.model != '':
    print ("loading pretrained model")
    ae.load_state_dict(torch.load(opt.model))

print(ae)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if opt.model == '':
    #print ("initializing weights!!")
    ae.apply(weights_init)

ae.cuda()

optimizer = optim.Adam(ae.parameters(), lr = 0.0001)

num_batch = len(dataset)/opt.batchSize

n_iter = 0

for epoch in range(1, opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()

        # Dataloader returns (point_canonical, points_rand_instance, _)
        points_canon, points, _ = data

        # Apply random rotation to batch
        if opt.no_rot:
            rot = np.eye(3)
        
        else:
            rot = rand_rotation_matrix()

        # Make rotation matrix into tensor and repeat for each point
        rot_tensor = torch.Tensor(rot).view(1, 9)
        rot_tensor = rot_tensor.repeat(1, points.shape[0]).view(points.shape[0], 3, 3)

        # Apply rotation to both instance cloud and canonical cloud
        points_r = torch.bmm(points, rot_tensor)
        points_canon_r = torch.bmm(points_canon, rot_tensor)

        # Make these pointclouds as tensors
        points = Variable(torch.Tensor(points_r))
        points_canon = Variable(torch.Tensor(points_canon_r))

        # Sub sample point cloud to train network on unseen points
        #choice = np.random.choice(4096, 4096, replace=True)
        #down = points[:, choice, :]

        down = points.cuda()
        down = down.transpose(2,1)

        points = points.cuda()
        points = points.transpose(2, 1)

        bs = points.size()[0]

        points_canon = points_canon.transpose(2,1)
        points_canon = points_canon.cuda()
        
        # Infer per-point translation from down sampled points
        translation = ae(down)

        #translation = torch.zeros_like(translation)
        
        points_canon = points_canon.transpose(2, 1).contiguous()
        down = down.transpose(2, 1).contiguous()
        #points = points.transpose(2, 1).contiguous()

        gen = down + translation

        #print (translation[0][0])
        
        #print(gen.size(), points.size(), dist1.size())
        
        loss = criterion(gen, points_canon)[0]

        # Log loss
        writer.add_scalar('Losses/chamfer_loss', loss.item(), n_iter)

        n_iter += 1

        loss.backward()
        optimizer.step()
        
        
        print('[%d: %d/%d] train loss %f' %(epoch, i, num_batch, loss.item()))

        if opt.viz and loss.item() < 0.004:

            #print (translation[0][0])

            down_sampeled = down[0].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            vec = o3d.utility.Vector3dVector(down_sampeled)
            pcd.points = vec
            o3d.visualization.draw_geometries([pcd])

            target = points_canon[0].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            vec = o3d.utility.Vector3dVector(target)
            pcd.points = vec
            o3d.visualization.draw_geometries([pcd])

            pred = gen[0].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            vec = o3d.utility.Vector3dVector(pred)
            pcd.points = vec
            o3d.visualization.draw_geometries([pcd])

    
    if not opt.dont_save_model:
        torch.save(ae.state_dict(), '%s/model_ae_%d.pt' % (opt.outf, epoch))
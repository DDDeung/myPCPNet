import torch
import numpy as np
import torch.nn as nn
import utils
import torch.nn.functional as F

#Spatial Transformer Network
class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.dim = dim
        self.num_points = num_points
        self.num_scales = num_scales
        self.sym_op = sym_op

        #parameters from pointnet-T-nets
        self.localization_net = nn.Sequential(
            nn.Conv1d(self.dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, self.dim*self.dim)
        )

        self.maxpool1 = nn.MaxPool1d(num_points)

        if self.num_scales > 1:
            self.nn = nn.Sequential(
                nn.Linear(1024*self.num_scales, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        batchsize = x.size()[0]

        x = self.localization_net(x)

        # symmetric operation
        if self.num_scales == 1:
            x = self.maxpool1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.maxpool1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = self.nn(x)

        x = self.fc_loc(x)

        # pointnet - T-Net
        eye = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batchsize, 1)
        x = x + eye
        x = x.view(-1, self.dim, self.dim)

        return x

"""
    we constrain the first spatial transformer to the domain of rotations and we exchange the max symmetric operation
    with a sum.
"""
class QSTN(nn.Module):
    def __init__(self,num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.num_points = num_points
        self.num_scales = num_scales
        self.sym_op = sym_op

        self.localization_net = nn.Sequential(
            nn.Conv1d(self.dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),

        )

        self.maxpool1 = nn.MaxPool1d(num_points)

        if self.num_scales > 1:
            self.nn = nn.Sequential(
                nn.Linear(1024 * self.num_scales, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        batchsize = x.size()[0]

        x = self.localization_net(x)

        # symmetric operation
        if self.num_scales == 1:
            x = self.maxpool1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024 * self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s * 1024:(s + 1) * 1024, :] = self.maxpool1(
                    x[:, :, s * self.num_points:(s + 1) * self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = self.nn(x)

        x = self.fc_loc(x)

        # add identity quaternion
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        x = utils.batch_quat_to_rotmat(x)

        # convert quaternion to rotation matrix
        return x

class PointNetFeat(nn.Module):
    def __init__(self, num_scales=1, num_points=500, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(PointNetFeat, self).__init__()
        self.num_scales = num_scales
        self.num_points = num_points
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals
        self.point_tuple = point_tuple

        if self.use_point_stn:
            self.stn1 = QSTN(num_scales=self.num_scales, num_points=num_points*self.point_tuple, dim=3, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(3*self.point_tuple, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
        )

        if self.num_scales > 1:
            self.nn = nn.Sequential(
                nn.Conv1d(1024, 1024*self.num_scales,1),
                nn.BatchNorm1d(1024*self.num_scales)
            )

        if self.sym_op == 'max':
            self.maxpool1 = nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.maxpool1 = None
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

    def forward(self, x):
        # input transfrom
        if self.use_point_stn:
            #from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3*self.point_tuple, -1)

        else:
            trans = None

        # MLP(3->64)
        x = self.mlp1(x)

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        #MLP(64->1024)
        x = self.mlp2(x)

        #MLP(1024, 1024*num_scales)
        if self.num_scales > 1:
            x = self.nn(F.relu(x))

        if self.get_pointfvals:
            pointfvals = x
        else:
            pointfvals = None

        # symmetric operation
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.maxpool1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
        else:
            x_scales = x.new_empty(x.size(0), 1024 * self.num_scales ** 2, 1)
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s * self.num_scales * 1024:(s + 1) * self.num_scales * 1024, :] = self.maxpool1(
                        x[:, :, s * self.num_points:(s + 1) * self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s * self.num_scales * 1024:(s + 1) * self.num_scales * 1024, :] = torch.sum(
                        x[:, :, s * self.num_points:(s + 1) * self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = x_scales

        x = x.view(-1, 1024*self.num_scales**2)

        return x, trans, trans2, pointfvals
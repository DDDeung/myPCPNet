import basisComponents as bC
import torch.nn as nn

class SPCPNet(nn.Module):
    def __init__(self, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(SPCPNet, self).__init__()
        self.num_points = num_points

        self.feat = bC.PointNetFeat(
            num_scales = 1,
            num_points = num_points,
            use_point_stn = use_point_stn,
            use_feat_stn = use_feat_stn,
            sym_op = sym_op,
            get_pointfvals = get_pointfvals,
            point_tuple = point_tuple
        )

        self.nn = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256,output_dim)
        )

    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = self.nn(x)

        return x, trans, trans2, pointfvals



class MPCPNet(nn.Module):
    def __init__(self,num_scales=2, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(MPCPNet, self).__init__()
        self.num_points = num_points
        self.feat = bC.PointNetFeat(
            num_scales = num_scales,
            num_points = num_points,
            use_point_stn = use_point_stn,
            use_feat_stn = use_feat_stn,
            sym_op = sym_op,
            get_pointfvals = get_pointfvals,
            point_tuple = point_tuple
        )

        self.nn = nn.Sequential(
            nn.Linear(1024*num_scales**2,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256,output_dim)
        )

    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = self.nn(x)

        return x, trans, trans2, pointfvals


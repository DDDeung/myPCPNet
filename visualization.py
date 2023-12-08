import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import utils
import torch

# number_points = 5
# pcd = np.random.rand(number_points, 3)  # uniform distribution over [0, 1)
# print(pcd)

# Create Figure:
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.scatter3D(pcd[:, 0], pcd[:, 1], pcd[:, 2])
# # label the axes
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Random Point Cloud")
# # display:
# plt.show()

# visualize point clouds
# pd_root = '/Users/koala/Desktop/cuhksz2023sem1/cv/PCPNet/pclouds'
# pd_shape_name = 'boxunion2100k'
# points_data = np.loadtxt(os.path.join(pd_root, pd_shape_name + '.xyz'), delimiter=" ", dtype=np.float32)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points_data[:, :3])
# o3d.visualization.draw_geometries([pcd])
#
# norm_root = '/Users/koala/Desktop/cuhksz2023sem1/cv/PCPNet/results/single_scale_normal'
# norm_shape_name =  'boxunion2100k'
# normals_data = np.loadtxt(os.path.join(norm_root, norm_shape_name + '.normals'), delimiter=" ", dtype=np.float32)
# print(normals_data)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.scatter3D(points_data[:, 0], points_data[:, 1], points_data[:, 2])
# # label the axes
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Visualize Point Cloud")
# # display:
# plt.show()
#

# colors = plt.cm.viridis

def get_target_features(opt):
    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in opt.pcpnet.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
        elif o == 'neighbor_normals':
            target_features.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
        else:
            raise ValueError('Unknown output: %s' % (o))

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')

    return target_features, output_target_ind, output_pred_ind, output_loss_weight, pred_dim


def cal_eachpoints_RMS(object_name):
    predict_root = '/Users/koala/Desktop/cuhksz2023sem1/cv/PCPNet/results/single_scale_normal'
    predict_name = object_name
    predict_data = np.loadtxt(os.path.join(predict_root, predict_name + '.normals'), delimiter=" ", dtype=np.float32)

    target_root = '/Users/koala/Desktop/cuhksz2023sem1/cv/PCPNet/pclouds'
    target_name = object_name
    target_data = np.loadtxt(os.path.join(target_root, target_name + '.normals'), delimiter=" ", dtype=np.float32)

    # load model and training parameters
    trainopt = torch.load('checkpoints/my_single_scale_normal_params.pth')
    target_features, output_target_ind, output_pred_ind, output_loss_weight, pred_dim = get_target_features(trainopt)

    c = predict_data - target_data
    d = torch.from_numpy((c))
    d = d.pow(2)
    d = d.sum(1)
    d = d.mean()

    if trainopt.pcpnet.outputs[0] == 'unoriented_normals':
        if trainopt.train.normal_loss == 'ms_euclidean':
            loss = torch.min(torch.from_numpy((predict_data - target_data)).pow(2).sum(1),
                             torch.from_numpy((predict_data + target_data)).pow(2).sum(1)) * \
                   output_loss_weight[0]
        elif trainopt.train.normal_loss == 'ms_oneminuscos':
            loss = (1 - torch.abs(utils.cos_angle(predict_data, target_data))).pow(2) * output_loss_weight[
                0]
        else:
            raise ValueError('Unsupported loss type: %s' % (trainopt.train.normal_loss))
    elif trainopt.train.normal_loss == 'oriented_normals':
        if trainopt.train.normal_loss == 'ms_euclidean':
            loss = (predict_data - target_data).pow(2).sum(1) * output_loss_weight[0]
        elif trainopt.train.normal_loss == 'ms_oneminuscos':
            loss = (1 - utils.cos_angle(predict_data, target_data)).pow(2) * output_loss_weight[0]
        else:
            raise ValueError('Unsupported loss type: %s' % (trainopt.train.normal_loss))
    else:
        raise ValueError('Unsupported output type: %s' % (trainopt.pcpnet.outputs[0]))

    return loss

def visualize(object_name):
    loss = cal_eachpoints_RMS(object_name)
    # Assign colors based on error size
    colors = plt.cm.viridis((loss -  torch.min(loss))/( torch.max(loss) -  torch.min(loss)))

    pd_root = '/Users/koala/Desktop/cuhksz2023sem1/cv/PCPNet/pclouds'
    pd_shape_name = object_name
    points_data = np.loadtxt(os.path.join(pd_root, pd_shape_name + '.xyz'), delimiter=" ", dtype=np.float32)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter3D(points_data[:, 0], points_data[:, 1], points_data[:, 2],c=colors)
    # label the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Visualize Point Cloud")
    # display:
    plt.show()
visualize('Liberty100k')

# def draw3d(object_name):
#     normals = np.loadtxt(f"results/single_scale_normal/{object_name}.normals")
#     points = np.loadtxt(f"pclouds/{object_name}.xyz")
#
#     # Create a point cloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.normals = o3d.utility.Vector3dVector(normals)
#     o3d.visualization.draw_plotly([pcd])
#
# draw3d('pipe100k')

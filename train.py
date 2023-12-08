import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from PCPNet import SPCPNet, MPCPNet
from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
import utils
import random
import os
import logging
from tqdm import tqdm

class PCPNetTrainer(object):
    def __init__(self, args, wb_logger):
        super().__init__()

        # train parameters
        self.nepoch = args.train.nepoch
        self.batchsize = args.train.batchsize
        self.lr = args.train.lr
        self.patch_radius = args.train.patch_radius
        self.patch_center = args.train.patch_center
        self.patch_point_count_std = args.train.patch_point_count_std
        self.patches_per_shape = args.train.patches_per_shape
        self.workers = args.train.workers
        self.cache_capacity = args.train.cache_capacity
        self.seed = args.train.seed
        self.training_order = args.train.training_order
        self.identical_epochs = args.train.identical_epochs
        self.momentum = args.train.momentum
        self.use_pca = args.train.use_pca
        self.normal_loss = args.train.normal_loss


        # model parameters
        self.outputs = args.pcpnet.outputs
        self.use_point_stn = args.pcpnet.use_point_stn
        self.use_feat_stn = args.pcpnet.use_feat_stn
        self.sym_op = args.pcpnet.sym_op
        self.point_tuple = args.pcpnet.point_tuple
        self.points_per_patch = args.pcpnet.points_per_patch

        self.data_root = args.data.root
        self.trainset = args.data.train
        self.testset = args.data.test
        self.name = args.name
        self.output_path = args.output_path
        self.wb_logger = wb_logger
        self.checkpoint_save_interval = args.train.checkpoint_save_interval

        # save parameters
        params_filename = os.path.join(self.output_path, '%s_params.pth' % (self.name))
        torch.save(args, params_filename)


        # get indices in targets and predictions corresponding to each output
        target_features = []
        self.output_target_ind = []
        self.output_pred_ind = []
        self.output_loss_weight = []
        pred_dim = 0
        for o in self.outputs:
            if o == 'unoriented_normals' or o == 'oriented_normals':
                if 'normal' not in target_features:
                    target_features.append('normal')
                self.output_target_ind.append(target_features.index('normal'))
                self.output_pred_ind.append(pred_dim)
                self.output_loss_weight.append(1.0)
                pred_dim += 3
            elif o == 'max_curvature' or o == 'min_curvature':
                if o not in target_features:
                    target_features.append(o)

                self.output_target_ind.append(target_features.index(o))
                self.output_pred_ind.append(pred_dim)
                if o == 'max_curvature':
                    self.output_loss_weight.append(0.7)
                else:
                    self.output_loss_weight.append(0.3)
                pred_dim += 1
            else:
                raise ValueError('Unknown output: %s' % (o))

        if pred_dim <= 0:
            raise ValueError('Prediction is empty for the given outputs.')

        # create model
        if len(self.patch_radius) == 1:
            self.model = SPCPNet(
                num_points=self.points_per_patch,
                output_dim=pred_dim,
                use_point_stn=self.use_point_stn,
                use_feat_stn=self.use_feat_stn,
                sym_op=self.sym_op,
                point_tuple=self.point_tuple)
        else:
            self.model = MPCPNet(
                num_scales=len(self.patch_radius),
                num_points=self.points_per_patch,
                output_dim=pred_dim,
                use_point_stn=self.use_point_stn,
                use_feat_stn=self.use_feat_stn,
                sym_op=self.sym_op,
                point_tuple=self.point_tuple
            )

        # create train dataset loaders
        train_dataset = PointcloudPatchDataset(
            root=self.data_root,
            shape_list_filename=self.trainset,
            patch_radius=self.patch_radius,
            points_per_patch=self.points_per_patch,
            patch_features=target_features,
            point_count_std=self.patch_point_count_std,
            seed=self.seed,
            identical_epochs=self.identical_epochs,
            use_pca=self.use_pca,
            center=self.patch_center,
            point_tuple=self.point_tuple,
            cache_capacity=self.cache_capacity)
        if self.training_order == 'random':
            train_datasampler = RandomPointcloudPatchSampler(
                train_dataset,
                patches_per_shape=self.patches_per_shape,
                seed=self.seed,
                identical_epochs=self.identical_epochs)
        elif self.training_order == 'random_shape_consecutive':
            train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
                train_dataset,
                patches_per_shape=self.patches_per_shape,
                seed=self.seed,
                identical_epochs=self.identical_epochs)
        else:
            raise ValueError('Unknown training order: %s' % (self.training_order))

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_datasampler,
            batch_size=self.batchsize,
            num_workers=int(self.workers))

        # create test dataset loader
        test_dataset = PointcloudPatchDataset(
            root=self.data_root,
            shape_list_filename=self.testset,
            patch_radius=self.patch_radius,
            points_per_patch=self.points_per_patch,
            patch_features=target_features,
            point_count_std=self.patch_point_count_std,
            seed=self.seed,
            identical_epochs=self.identical_epochs,
            use_pca=self.use_pca,
            center=self.patch_center,
            point_tuple=self.point_tuple,
            cache_capacity=self.cache_capacity)
        if self.training_order == 'random':
            test_datasampler = RandomPointcloudPatchSampler(
                test_dataset,
                patches_per_shape=self.patches_per_shape,
                seed=self.seed,
                identical_epochs=self.identical_epochs)
        elif self.training_order == 'random_shape_consecutive':
            test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
                test_dataset,
                patches_per_shape=self.patches_per_shape,
                seed=self.seed,
                identical_epochs=self.identical_epochs)
        else:
            raise ValueError('Unknown training order: %s' % (self.training_order))

        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            sampler=test_datasampler,
            batch_size=self.batchsize,
            num_workers=int(self.workers)
        )

    def train_one_epoch(self,epoch):
        # switch to train mode
        self.model.train()
        losses = 0.0
        loader = tqdm(self.train_dataloader)
        train_enum = enumerate(loader, 0)

        for i, data in train_enum:
            # get data batch and upload to device
            points = data[0]
            target = data[1:-1]
            points = points.transpose(2,1)
            points = points.to(self.device)
            target = tuple(t.to(self.device) for t in target)

            # zero gradients
            self.optimizer.zero_grad()

            # forward pass
            pred, trans, _, _ = self.model(points)

            # compute loss
            loss = self.compute_loss(
                pred = pred,
                target=target,
                outputs=self.outputs,
                output_pred_ind=self.output_pred_ind,
                output_target_ind=self.output_target_ind,
                output_loss_weight=self.output_loss_weight,
                patch_rot=trans if self.use_point_stn else None,
                normal_loss=self.normal_loss
            )
            losses += loss.item()
            #bp
            loss.backward()
            self.optimizer.step()

            self.train_fraction_done = (i + 1) / self.train_num_batch

            # update learning rate
            self.scheduler.step(epoch * self.train_num_batch + i)

        return losses/len(self.train_dataloader)


    def test_one_epoch(self,test_enum):
        # switch to eval mode
        self.model.eval()
        test_batchind, data = next(test_enum)

        # get data batch and upload to device
        points = data[0]
        target = data[1:-1]
        points = points.transpose(2, 1)
        points = points.to(self.device)
        target = tuple(t.to(self.device) for t in target)

        # forward pass
        with torch.no_grad():
            pred, trans, _, _ = self.model(points)

        loss = self.compute_loss(
            pred=pred,
            target=target,
            outputs=self.outputs,
            output_pred_ind=self.output_pred_ind,
            output_target_ind=self.output_target_ind,
            output_loss_weight=self.output_loss_weight,
            patch_rot=trans if self.use_point_stn else None,
            normal_loss=self.normal_loss)

        self.test_fraction_done = (test_batchind + 1) / self.test_num_batch

        return test_batchind,loss.item()


    def train(self):
        if self.seed < 0:
            self.seed = random.randint(1, 10000)

        print("Random Seed: %d" % (self.seed))
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # device - use "mps" if possible
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        # learning rate
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[],
                                             gamma=0.1)  # milestones in number of optimizer iterations
        self.model.to(self.device)

        self.train_num_batch = len(self.train_dataloader)
        self.test_num_batch = len(self.test_dataloader)



        min_loss = 100000
        for epoch in range(self.nepoch):
            train_batchind = -1
            self.train_fraction_done = 0.0

            test_batchind = -1
            self.test_fraction_done = 0.0
            test_enum = enumerate(self.test_dataloader, 0)

            train_loss = self.train_one_epoch(epoch)
            print(
                f"[ Train | {epoch + 1:03d}/{self.nepoch:03d} ] train_loss = {train_loss:.5f}")

            test_losses = 0.0
            while self.test_fraction_done <= self.train_fraction_done and test_batchind + 1 < self.test_num_batch:
                test_batchind, test_loss = self.test_one_epoch(test_enum)
                test_losses += test_loss
            test_losses /= len(self.test_dataloader)
            print(
                f"[ Test | {epoch + 1:03d}/{self.nepoch:03d} ] test_loss = {test_losses:.5f}")
            log_info = {"epoch": epoch, "learning_rate": self.lr, "train_loss":train_loss, "test_loss": test_losses}
            self.wb_logger.log(log_info)
            if test_losses < min_loss:
                min_loss = test_losses
                is_best = True
            else:
                is_best = False

            if is_best or (epoch + 1) % self.checkpoint_save_interval == 0:
                self.save_checkpoint(
                    state={
                        'epoch': epoch + 1,
                        'loss': test_losses,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    },
                    is_best=is_best,
                    root=self.output_path,
                    filename=f"checkpoint_epoch_{epoch + 1}.pth.tar"
                )


    def compute_loss(self, pred, target, outputs, output_pred_ind, output_target_ind, output_loss_weight, patch_rot,
                     normal_loss):

        loss = 0
        for oi, o in enumerate(outputs):
            if o == 'unoriented_normals' or o == 'oriented_normals':
                o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi] + 3]
                o_target = target[output_target_ind[oi]]

                if patch_rot is not None:
                    # transform predictions with inverse transform
                    # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                    o_pred = torch.bmm(o_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)

                if o == 'unoriented_normals':
                    if normal_loss == 'ms_euclidean':
                        loss += torch.min((o_pred - o_target).pow(2).sum(1), (o_pred + o_target).pow(2).sum(1)).mean() * \
                                output_loss_weight[oi]
                    elif normal_loss == 'ms_oneminuscos':
                        loss += (1 - torch.abs(utils.cos_angle(o_pred, o_target))).pow(2).mean() * output_loss_weight[
                            oi]
                    else:
                        raise ValueError('Unsupported loss type: %s' % (normal_loss))
                elif o == 'oriented_normals':
                    if normal_loss == 'ms_euclidean':
                        loss += (o_pred - o_target).pow(2).sum(1).mean() * output_loss_weight[oi]
                    elif normal_loss == 'ms_oneminuscos':
                        loss += (1 - utils.cos_angle(o_pred, o_target)).pow(2).mean() * output_loss_weight[oi]
                    else:
                        raise ValueError('Unsupported loss type: %s' % (normal_loss))
                else:
                    raise ValueError('Unsupported output type: %s' % (o))
            elif o == 'max_curvature' or o == 'min_curvature':
                o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi] + 1]
                o_target = target[output_target_ind[oi]]

                # Rectified mse loss: mean square of (pred - gt) / max(1, |gt|)
                normalized_diff = (o_pred - o_target) / torch.clamp(torch.abs(o_target), min=1)
                loss += normalized_diff.pow(2).mean() * output_loss_weight[oi]

            else:
                raise ValueError('Unsupported output type: %s' % (o))

        return loss

    def save_checkpoint(self, state, is_best, root, filename='checkpoint.pth.tar'):
        if is_best:
            torch.save(state, os.path.join(root, 'model_best.pth.tar'))
        else:
            torch.save(state, os.path.join(root, filename))

    def save(self, filename="saved"):
        path = os.path.join(self.output_path, filename + ".pth")
        torch.save(self.model.state_dict(), path)
        logging.info(f"Saved in {path}")


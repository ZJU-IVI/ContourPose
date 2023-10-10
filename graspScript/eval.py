import threading
import cv2
import numpy as np
import os
from network import ContourPose
from torch import nn
import torch
import matplotlib.pyplot as plt


def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0
    if not os.path.exists(model_dir):
        return 0
    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir) if "pkl" in pth]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch

    print("Load model: {}".format(os.path.join(model_dir, "{}.pkl".format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, "{}.pkl".format(pth)))
    try:
        net.load_state_dict(pretrained_model['net'], strict=strict)
    except KeyError:
        net.load_state_dict(pretrained_model, strict=strict)
    return pth


# config
K = np.array([[679.89393628, 0, 323.41658954],
                      [0, 679.84846281, 278.94291854],
                      [0, 0, 1]])



class evaluator(threading.Thread):
    def __init__(self, img_queue, res_queue, obj_id):
        super().__init__()
        self.img_queue = img_queue
        self.res_queue = res_queue
        self.obj_id = obj_id
        self.ContourPoseNet = self.getContourNet()
        self.ContourPoseNet.eval()

    def getContourNet(self):
        class_type = "obj{}".format(self.obj_id)
        data_path = "/home/lqz/liquanzhi/allDataset/data"
        model_path = "/home/lqz/liquanzhi/Edge_Pose/model"
        model_epoch = 140
        self.corner = np.loadtxt(os.path.join(data_path, "train", class_type, "{}.txt".format(class_type)))
        ContourPoseNet = ContourPose(heatmap_dim=self.corner.shape[0])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ContourPoseNet = nn.DataParallel(ContourPoseNet, device_ids=[0, 1])
        ContourPoseNet = ContourPoseNet.to(device)
        # load model
        model_path = os.path.join(model_path, class_type)
        load_network(ContourPoseNet, model_path, epoch=model_epoch)
        return ContourPoseNet

    def run(self):
        while True:
            image = self.img_queue.get()
            if image is None:
                break
            # 新增一个batch_size通道
            img_tensor = image.unsqueeze(0)
            with torch.no_grad():
                pred_heatmap, pred_edge = self.ContourPoseNet(img_tensor)
                # self.visualize_heatmap(img_tensor, pred_heatmap)
                predict_2d = self.map2points(pred_heatmap)
                result = self.calculatePose(img_tensor, predict_2d)
            self.res_queue.put(result)

    def map2points(self, pred_heatmap):
        pred = pred_heatmap.view(pred_heatmap.shape[0], pred_heatmap.shape[1], -1)
        width = pred_heatmap.shape[3]
        pred_max = torch.argmax(pred, dim=2)
        pred_x = pred_max / width
        pred_x = pred_x.int()
        pred_y = pred_max % width
        pred_x = pred_x.unsqueeze(dim=2)
        pred_y = pred_y.unsqueeze(dim=2)
        pred_points = torch.cat((pred_y, pred_x), dim=2)
        return pred_points

    def calculatePose(self, img, predict2d):
        batch_size = img.shape[0]
        if isinstance(img, torch.Tensor):
            img = img.permute(0, 2, 3, 1).detach().cpu().numpy()
            img *= [0.184, 0.206, 0.197]
            img += [0.419, 0.427, 0.424]
            img = img * 255.0
        # img = img.astype(np.uint8)
        predict = predict2d[0].detach().cpu().numpy().reshape(predict2d.shape[1], -1)
        k = K
        pred_pose = self.pnp(self.corner, predict, k)
        print(self.obj_id)
        print(pred_pose)
        # self.visual(img, pred_pose, k)
        return pred_pose

    def pnp(self, points_3d, points_2d, camera_matrix):  # SOLVEPNP_ITERATIVE

        try:
            dist_coeffs = self.dist_coeffs
        except:
            dist_coeffs = np.zeros(shape=[5, 1], dtype="float64")

        assert (
                points_3d.shape[0] == points_2d.shape[0]
        ), "points 3D and points 2D must have same number of vertices"
        points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
        points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
        camera_matrix = camera_matrix.astype(np.float64)

        if points_2d.shape[0] < 5:
            _, R_exp, t = cv2.solvePnP(
                points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            points_3d = np.expand_dims(points_3d, 0)
            points_2d = np.expand_dims(points_2d, 0)
            _, R_exp, t, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, camera_matrix, dist_coeffs, iterationsCount=200, reprojectionError=5,
                flags=cv2.SOLVEPNP_ITERATIVE)  # , reprojectionError=1.2
            # )

        R, _ = cv2.Rodrigues(R_exp)
        return np.concatenate([R, t], axis=-1)

    def visualize_heatmap(self, rgb, pred_heatmap):
        batch_size = rgb.shape[0]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.permute(0, 2, 3, 1).detach().cpu().numpy()
            rgb *= [0.184, 0.206, 0.197]
            rgb += [0.419, 0.427, 0.424]
            rgb = rgb * 255.0
        rgb = rgb.astype(np.uint8)
        for i in range(batch_size):
            pred_heatmap_test = pred_heatmap[i].detach().cpu().numpy()
            all_pred = np.zeros((pred_heatmap_test.shape[1], pred_heatmap_test.shape[2]))
            for j in range(pred_heatmap_test.shape[0]):
                all_pred = all_pred + pred_heatmap_test[j]
            img_add_pred = cv2.addWeighted(rgb[i, :, :, 0], 0.1, all_pred, 30, 1, dtype=cv2.CV_32F)

            plt.imshow(img_add_pred)
            plt.draw()
            plt.pause(0.5)
            plt.close()

import torch
import tqdm
import numpy as np
import os
import os.path as osp
from utils.utils import load_ply, project
import cv2
from scipy import spatial
from itertools import combinations
from random import choice
import math
from config import rtDic, diameters

cuda = torch.cuda.is_available()

class evaluator:
    def __init__(self, args, model, test_loader, device):
        self.args = args
        self.model = model
        self.cad_path = osp.join(os.getcwd(), "cad")
        self.mesh_model = load_ply(
            osp.join(
                self.cad_path, "{}.ply".format(args.class_type)
            )
        )
        corners = np.loadtxt(os.path.join(self.cad_path, '{}.txt'.format(args.class_type)))
        self.corners = corners
        self.keyponits = np.loadtxt(
            os.path.join(args.data_path, "train", self.args.class_type, "{}.txt".format(args.class_type)))
        self.device = device
        self.pts_3d = self.mesh_model["pts"] * 1000
        self.data_loader = test_loader
        self.valid_3d = np.loadtxt(osp.join(os.getcwd(), "Valid3D/{}.txt".format(args.class_type)))
        self.index = 1
        self.threshold = args.threshold
        self.proj_2d = []
        self.proj_2d_mean = []
        self.add = []
        self.x_error_all = []
        self.y_error_all = []
        self.z_error_all = []
        self.alpha_error_all = []
        self.beta_error_all = []
        self.gama_error_all = []
        # ablation
        # self.error_num = 0
        self.diameter = diameters[args.class_type] / 1000.0

        # for PECP
        example = ""
        for i in range(0, self.keyponits.shape[0]):
            if i < 10:
                example = example + str(i)
            if i >= 10:
                example = example + chr(97 + i - 10)
        self.list_all = list(combinations(example, 4))

    def evaluate(self):
        self.model.eval()

        with torch.no_grad():
            for data in tqdm.tqdm(self.data_loader, leave=False, desc="val"):
                if cuda:
                    img, heatmap, K, pose = [x.to(self.device) for x in data]
                else:
                    img, heatmap, K, pose = data

                pred_heatmap, pred_contour = self.model(img)
                keypoints_2d, predict_2d = self.map_2_points(heatmap, pred_heatmap)
                self.calculate_metric(keypoints_2d, predict_2d, K)
                # self.calculate_metric_PECP(keypoints_2d, predict_2d, K, pose, pred_contour)

            proj_2d_mean = np.mean(self.proj_2d)
            add_mean = np.mean(self.add)
            x = np.mean(self.x_error_all)
            y = np.mean(self.y_error_all)
            z = np.mean(self.z_error_all)
            alpha = np.mean(self.alpha_error_all)
            beta = np.mean(self.beta_error_all)
            gamma = np.mean(self.gama_error_all)
            print("model class type:{}:2D- {}  ADD-{}".format(self.args.class_type, proj_2d_mean, add_mean))
            print('x error:{} mm, y error:{} mm, z error:{} mm'.format(x, y, z))
            print("translation error:{} mm".format((x ** 2 + y ** 2 + z ** 2) ** 0.5))
            print('alpha error:{} °, beta error:{} °, gamaa error:{} °'.format(alpha, beta, gamma))
            print("rotation error:{} mm".format((alpha ** 2 + beta ** 2 + gamma ** 2) ** 0.5))

    def map_2_points(self, heatmap, pred_heatmap):

        def extract_coords(input_map):
            flat_map = input_map.view(input_map.shape[0], input_map.shape[1], -1)
            max_idx = torch.argmax(flat_map, dim=2)
            width = input_map.shape[3]
            x = (max_idx / width).int().unsqueeze(dim=2)
            y = (max_idx % width).unsqueeze(dim=2)
            return torch.cat((y, x), dim=2)

        gt_points = extract_coords(heatmap)
        pred_points = extract_coords(pred_heatmap)
        return gt_points, pred_points

    def calculate_tra_and_rot(self, pose, pred_pose):
        if self.add[-1] == False:
            return 0
        pred_pose = self.pose_reverse(pred_pose, pose)
        rot = pose[:, :3]
        tra = pose[:, 3:].reshape(1, 3)
        pred_rot = pred_pose[:, :3]
        pred_tra = pred_pose[:, 3:].reshape(1, 3)
        tra_error = (tra - pred_tra) * 1000

        x_error = math.fabs(tra_error[:, 0])
        y_error = math.fabs(tra_error[:, 1])
        z_error = math.fabs(tra_error[:, 2])

        self.x_error_all.append(x_error)
        self.y_error_all.append(y_error)
        self.z_error_all.append(z_error)

        sy = math.sqrt(rot[2, 1] * rot[2, 1] + rot[2, 2] * rot[2, 2])
        alpha = math.atan2(rot[2, 1], rot[2, 2])
        beta = math.atan2(-rot[2, 0], sy)
        gamma = math.atan2(rot[1, 0], rot[0, 0])

        pred_sy = math.sqrt(pred_rot[2, 1] * pred_rot[2, 1] + pred_rot[2, 2] * pred_rot[2, 2])
        pred_alpha = math.atan2(pred_rot[2, 1], pred_rot[2, 2])
        pred_beta = math.atan2(-pred_rot[2, 0], pred_sy)
        pred_gamma = math.atan2(pred_rot[1, 0], pred_rot[0, 0])

        alpha_error = math.fabs((math.fabs(alpha) - math.fabs(pred_alpha)) * 180 / math.pi)
        beta_error = math.fabs((math.fabs(beta) - math.fabs(pred_beta)) * 180 / math.pi)
        gamma_error = math.fabs((math.fabs(gamma) - math.fabs(pred_gamma)) * 180 / math.pi)

        self.alpha_error_all.append(alpha_error)
        self.beta_error_all.append(beta_error)
        self.gama_error_all.append(gamma_error)

    def calculate_metric(self, keypoints2d, predict2d, K):

        batch_size = keypoints2d.shape[0]
        keypoints_num = keypoints2d.shape[1]

        for i in range(batch_size):
            keypoints = keypoints2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            predict = predict2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            #self.set_error_points(predict)
            k = K[i].detach().cpu().numpy()
            gt_pose = self.pnp(self.keyponits, keypoints, k)
            pred_pose = self.pnp(self.keyponits, predict, k)
            self.projection_2d(pred_pose, gt_pose, k)
            if self.args.class_type in ["obj1", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29', 'obj33']:
                self.add_metric(pred_pose, gt_pose, syn=True)
            else:
                self.add_metric(pred_pose, gt_pose)
            self.calculate_tra_and_rot(gt_pose, pred_pose)

    def set_error_points(self, predict):
        error_list = np.random.choice(10, self.error_num, replace=False)
        for num in error_list:
            w = np.random.randint(0, 480)
            h = np.random.randint(0, 640)
            predict[num] = [w, h]

    def calculate_metric_PECP(self, keypoints2d, predict2d, K, pose, pred_contour):

        batch_size = keypoints2d.shape[0]
        keypoints_num = keypoints2d.shape[1]

        for i in range(batch_size):
            predict = predict2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            # self.set_error_points(predict)
            k = K[i].detach().cpu().numpy()
            gt_pose = pose[i].detach().cpu().numpy()
            contour = pred_contour[i][0].detach().cpu().numpy()
            contour[contour >= 0] = 1
            contour[contour < 0] = 0
            contour = np.asarray(contour).astype(np.uint8)
            foreground = np.sum(contour)
            if foreground > 1000:
                # Gaussian convolution can be used here
                edge_heatmap = contour
                pred_pose = self.PECP(self.keyponits, predict, k, edge_heatmap, self.list_all)
            else:
                pred_pose = self.pnp(self.keyponits, predict, k)
            self.projection_2d(pred_pose, gt_pose, k)
            if self.args.class_type in ["obj1", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29', 'obj33']:
                self.add_metric(pred_pose, gt_pose, syn=True)
            else:
                self.add_metric(pred_pose, gt_pose)
            self.calculate_tra_and_rot(gt_pose, pred_pose)

    def PECP(self, points_3d, points_2d, K, target_contour, list_all):
        match_dict = {}
        for i in range(points_3d.shape[0]):
            match_keypoints = np.concatenate((points_2d[i], points_3d[i]), axis=0)
            # 1 2 3 4 5 6 7 8 9 a b c
            if i >= 10:
                match_dict[chr(97 + i - 10)] = match_keypoints
            else:
                match_dict[str(i)] = match_keypoints

        list_score = np.zeros(points_2d.shape[0])

        # Number of iterations obtained by calculation
        iteration_time = 400

        for i in range(iteration_time):
            temp_list = choice(list_all)
            keypoints_2d = np.zeros((temp_list.__len__(), 2))
            keypoints_3d = np.zeros((temp_list.__len__(), 3))
            for j in range(temp_list.__len__()):
                keypoints_2d[j] = match_dict[temp_list[j]][:2]
                keypoints_3d[j] = match_dict[temp_list[j]][2:]
            _, R_exp, t = cv2.solvePnP(keypoints_3d, keypoints_2d, K, distCoeffs=np.zeros(shape=[5, 1], dtype="float64"),
                                       flags=cv2.SOLVEPNP_EPNP)
            R, _ = cv2.Rodrigues(R_exp)
            pose = np.concatenate([R, t], axis=-1)
            # 2d points
            valid_2d = project(self.valid_3d, K, pose).astype(int)
            sum, valid_2d = self.get_confidence(target_contour, valid_2d)
            score = sum - 0.33 * valid_2d.shape[0]
            if score > 0:
                for t in temp_list:
                    if t >= 'a':
                        index = ord(t) - 97 + 10
                    else:
                        index = int(t)
                    list_score[index] = list_score[index] + score
        max = 0
        total = points_2d.shape[0] + 1
        error_num = 0
        for k in range(4, total):
            k = k - error_num
            top_index = self.top_K_idx(list_score, k)
            keypoints_2d = np.zeros((top_index.shape[0], 2))
            keypoints_3d = np.zeros((top_index.shape[0], 3))
            j = 0
            for idx in top_index:
                if idx >= 10:
                    temp_idx = chr(97 + idx - 10)
                else:
                    temp_idx = str(idx)
                keypoints_2d[j] = match_dict[temp_idx][:2]
                keypoints_3d[j] = match_dict[temp_idx][2:]
                j = j + 1
            if top_index.shape[0] == 4:
                _, R_exp, t = cv2.solvePnP(keypoints_3d, keypoints_2d, K,
                                           distCoeffs=np.zeros(shape=[5, 1], dtype="float64"),
                                           flags=cv2.SOLVEPNP_EPNP)
                R, _ = cv2.Rodrigues(R_exp)
                pose = np.concatenate([R, t], axis=-1)
            else:
                pose = self.pnp(keypoints_3d, keypoints_2d, K)
            valid_2d = project(self.valid_3d, K, pose).astype(int)  # 得到2d点
            sum, valid_2d = self.get_confidence(target_contour, valid_2d)
            if sum >= max:
                max = sum
                pose_final = pose
            else:
                list_score[top_index[-1]] = -1
                total = total - 1
                error_num = error_num + 1
                continue
        ransac_pose = self.pnp(points_3d, points_2d, K)
        valid_2d = project(self.valid_3d, K, ransac_pose).astype(int)  # 得到2d点
        sum, valid_2d = self.get_confidence(target_contour, valid_2d)
        if sum >= max:
            pose_final = ransac_pose
        return pose_final

    def get_confidence(self, target_contour, valid_2d):
        valid_2d[valid_2d[:, 0] >= 640] = 0
        valid_2d[valid_2d[:, 0] < 0] = 0
        valid_2d[valid_2d[:, 1] >= 480] = 0
        valid_2d[valid_2d[:, 1] < 0] = 0
        valid_2d = np.unique(valid_2d, axis=0)
        sum = np.sum(target_contour[valid_2d[:, 1], valid_2d[:, 0]])
        return sum, valid_2d

    def pnp(self, points_3d, points_2d, camera_matrix):

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
                points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
            )
        else:
            if self.args.class_type in ["obj2", "obj32"]:
                _, R_exp, t, inliers = cv2.solvePnPRansac(
                    points_3d, points_2d, camera_matrix, dist_coeffs, iterationsCount=1000, reprojectionError=5,
                    flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                _, R_exp, t, inliers = cv2.solvePnPRansac(
                    points_3d, points_2d, camera_matrix, dist_coeffs, iterationsCount=1000, reprojectionError=5,
                    flags=cv2.SOLVEPNP_EPNP)

        R, _ = cv2.Rodrigues(R_exp)
        return np.concatenate([R, t], axis=-1)

    def projection_2d(self, pose_pred, pose_targets, K):
        # rot pos in z
        pose_pred = self.pose_reverse(pose_pred, pose_targets)

        model_2d_pred = project(self.mesh_model["pts"], K, pose_pred)
        model_2d_targets = project(self.mesh_model["pts"], K, pose_targets)
        proj_mean_diff = np.mean(
            np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1)
        )

        if proj_mean_diff < self.threshold:
            self.proj_2d_mean.append(proj_mean_diff)
        self.proj_2d.append(proj_mean_diff < self.threshold)

    def pose_reverse(self, pose_pred, pose_targets):
        if self.args.class_type in ["obj1", "obj2", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29',
                                    'obj33']:
            rot = rtDic[self.args.class_type]
            pose_pred2 = np.dot(pose_pred, rot)
            ori = np.linalg.norm(pose_targets - pose_pred)
            new = np.linalg.norm(pose_targets - pose_pred2)
            if new < ori:
                pose_pred = pose_pred2
        return pose_pred

    def add_metric(self, pose_pred, pose_targets, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = (
                np.dot(self.mesh_model["pts"], pose_pred[:, :3].T) + pose_pred[:, 3]
        )
        model_targets = (
                np.dot(self.mesh_model["pts"], pose_targets[:, :3].T) + pose_targets[:, 3]
        )

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        self.add.append(mean_dist < diameter)

    def top_K_idx(self, data, k):
        data = np.array(data)
        idx = data.argsort()[-k:][::-1]
        return idx

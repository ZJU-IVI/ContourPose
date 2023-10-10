import random
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import glob, pickle
import numpy as np
import os.path as osp
from utils.utils import project
import yaml
import cv2
from utils.transforms import rotate_img
from heatmap import generate_heatmap
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
blender_K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])

class MyDataset(Dataset):
    def __init__(self, root, cls, is_train=True, scene=None, index=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.test_root = osp.join(self.root, "test")
        self.cls = cls
        self.objID = self.cls[3:]
        self.data_paths = []
        self.is_training = is_train

        self.scene = scene
        self.index = index
        self.corners = np.loadtxt(os.path.join(root, "train", cls, "{}.txt".format(cls)))  # KEYPOINTS
        self.element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.bg_imgs_path = os.path.join(os.getcwd(), "bg_imgs.npy")
        self.sun_path = os.path.join(root, "SUN2012pascalformat")
        self.get_bg_imgs()
        self.bg_imgs = np.load(self.bg_imgs_path).astype(np.str)

        if is_train:
            self.path = osp.join(self.root, "train", self.cls)
            self.K = yaml.load(open(osp.join(self.path, 'Intrinsic.yml'), 'r'))
            self.train_pose = yaml.load(open(osp.join(self.path, 'gt.yml'), 'r'))
            self.data_paths = self.get_train_data_path(self.root, self.cls)

        else:
            self.path = osp.join(self.root, "test/scene{}".format(str(self.scene)))
            self.K = yaml.load(open(osp.join(self.path, 'Intrinsic.yml'), 'r'))
            self.train_pose = yaml.load(open(osp.join(self.path, 'gt.yml'), 'r'))
            self.data_paths = self.get_test_data_path()

    def get_train_data_path(self, root, cls):
        paths_list = []
        paths = {}

        count = os.listdir(osp.join(self.path, "photo_cut"))
        train_inds = [int(ind.replace(".png", "")) for ind in count]
        train_inds.sort()

        train_img_path = osp.join(self.path, "photo_cut")
        mask_path = osp.join(self.path, "mask")  # use test data train
        edge_path = osp.join(self.path, "gtEdge")

        render_dir = osp.join(root, "train", "renders", cls)
        render_edge_dir = osp.join(root, "train", "renders", "Render_edge", cls)
        render_num = len(glob.glob(osp.join(render_dir, "*.pkl")))

        for idx in train_inds:
            idx = int(idx)
            img_name = "{}.png".format(idx)
            mask_name = "{}.png".format(idx)
            paths["img_path"] = osp.join(train_img_path, img_name)
            paths["mask_path"] = osp.join(mask_path, mask_name)
            paths["edge_path"] = osp.join(edge_path, img_name)
            paths["type"] = "true"
            paths_list.append(paths.copy())

        for idx in range(render_num):
            img_name = "{}.jpg".format(idx)
            edge_name = "{}.png".format(idx)
            mask_name = "{}_depth.png".format(idx)
            pose_name = "{}_RT.pkl".format(idx)

            paths["img_path"] = osp.join(render_dir, img_name)
            paths["mask_path"] = osp.join(render_dir, mask_name)
            paths["pose_path"] = osp.join(render_dir, pose_name)
            paths["edge_path"] = osp.join(render_edge_dir, edge_name)
            paths["type"] = "render"
            paths_list.append(paths.copy())

        return paths_list

    def get_test_data_path(self):
        paths_list = []
        paths = {}

        train_img_path = osp.join(self.path, "photo_cut")
        mask_path = osp.join(self.path, "mask")
        edge_path = osp.join(self.path, "edge")

        for idx in range(1, 417):
            idx = int(idx)
            img_name = "{}_{}.png".format(idx, self.objID)
            mask_name = "{}_{}.png".format(idx, self.objID)
            edge_name = "{}_{}.png".format(idx, self.objID)
            paths["img_path"] = osp.join(train_img_path, img_name)
            paths["mask_path"] = osp.join(mask_path, mask_name)
            paths["edge_path"] = osp.join(edge_path, edge_name)

            paths["type"] = "test"
            paths_list.append(paths.copy())
        return paths_list

    def get_data(self, path):
        img = np.array(Image.open(path["img_path"]))
        if path["type"] == "true":
            gt_contour = np.array(Image.open(path["edge_path"]))
            gt_contour = cv2.dilate(gt_contour, kernel=self.element)
            mask = (np.asarray(cv2.imread(path["img_path"], 0)) != 0).astype(np.uint8)
            idx = int(osp.basename(path["img_path"]).replace(".png", ""))
            instance_gt = self.train_pose[str(idx)][0]
            K = np.array(self.K[str(idx)]).reshape(3, 3)
            R = np.array(instance_gt['m2c_R']).reshape(3, 3)
            t = np.array(instance_gt['m2c_T']).reshape(3, 1)
            pose = np.concatenate([R, t], axis=1)  # 合成3x4位姿矩阵
            return img, mask, pose, K, gt_contour
        if path["type"] == "render":
            with open(path["pose_path"], "rb") as f:
                pose = pickle.load(f)["RT"]
            K = blender_K
            mask = (np.asarray(Image.open(path["mask_path"]))).astype(np.uint8)
            gt_contour = np.array(Image.open(path["edge_path"]))
            gt_contour = cv2.dilate(gt_contour, kernel=self.element)
            return img, mask, pose, K, gt_contour

        elif path["type"] == "test":
            idx = int(osp.basename(path["img_path"]).replace("_{}.png".format(self.objID), ""))
            K = np.array(self.K[str(idx)][self.index][int(self.objID)]).reshape(3, 3)
            instance_gt = self.train_pose[str(idx)][self.index]  # test
            R = np.array(instance_gt['m2c_R']).reshape(3, 3)
            t = np.array(instance_gt['m2c_T']).reshape(3, 1)
            pose = np.concatenate([R, t], axis=1)  # 合成3x4位姿矩阵
            return img, pose, K

    def get_bg_imgs(self):
        if os.path.exists(self.bg_imgs_path):
            return

        img_paths = glob.glob(os.path.join(self.sun_path, 'JPEGImages/*'))
        bg_imgs = []

        for img_path in tqdm(img_paths):
            bg_imgs.append(img_path)

        np.save(self.bg_imgs_path, bg_imgs)

    def random_background(self, img, mask):
        self.get_bg_imgs()
        random_img_path = random.choice(self.bg_imgs)
        random_img = cv2.imread(random_img_path)
        row, col = img.shape[:2]
        if row < 481 or col < 641:
            random_img = cv2.resize(random_img, dsize=(960, int((960 / row) * col)))
        random_img = self.random_crop(random_img)  # crop 480*640
        random_img = self.random_filp(random_img)  # random flip

        mask_img = cv2.bitwise_and(img, img, mask=mask)
        maskCopy = np.array(mask)
        np.place(maskCopy, maskCopy > 0, 255)
        mask_inv = cv2.bitwise_not(maskCopy)
        background = cv2.bitwise_and(random_img, random_img, mask=mask_inv)
        background = cv2.medianBlur(background, 3)
        last_img = cv2.add(mask_img, background)
        last_img = cv2.medianBlur(last_img, 3)
        return last_img

    def random_translation(self, img, edge, mask, heatmap, is_render=False):
        if is_render:
            random_y = random.randint(-80, 80)
            random_x = random.randint(-100, 120)
        else:
            random_y = random.randint(-100, 150)
            random_x = random.randint(-150, 200)
        M = np.float32([[1, 0, random_x], [0, 1, random_y]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        edge = cv2.warpAffine(edge, M, (edge.shape[1], edge.shape[0]))
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        for i in range(heatmap.shape[0]):
            heatmap[i] = cv2.warpAffine(heatmap[i], M, (edge.shape[1], edge.shape[0]))
        return img, edge, mask, heatmap

    def random_rotation_and_resize(self, img, edge, mask, heatmap, is_render=False):
        if not is_render:
            ratio = random.uniform(0.8, 1.1)
        else:
            ratio = random.uniform(0.7, 1.2)
        flag = random.randint(1, 4)
        if flag == 1:
            random_angle = random.randint(0, 360)
        else:
            random_angle = 0
        height = img.shape[1]
        width = img.shape[0]
        mat = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), random_angle, ratio)
        img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))
        edge = cv2.warpAffine(edge, mat, (edge.shape[1], edge.shape[0]))
        mask = cv2.warpAffine(mask, mat, (mask.shape[1], mask.shape[0]))
        for i in range(heatmap.shape[0]):
            heatmap[i] = cv2.warpAffine(heatmap[i], mat, (edge.shape[1], edge.shape[0]))
        return img, edge, mask, heatmap

    def random_crop(self, img):
        h, w = img.shape[:2]
        y = np.random.randint(0, h - 480)
        x = np.random.randint(0, w - 640)
        image = img[y:y + 480, x:x + 640, :]
        return image

    def random_filp(self, image):
        flip_prop = np.random.randint(low=0, high=3)
        axis = np.random.randint(0, 2)
        if flip_prop == 0:
            image = cv2.flip(image, axis)
        return image

    def augment(self, img, mask, gt_contour, pose, K):
        img = np.asarray(img).astype(np.uint8)
        if True:
            # randomly mask out to add occlusion
            R = np.eye(3, dtype=np.float32)
            R_orig = pose[:3, :3]
            T_orig = pose[:3, 3]

            img, mask, gt_contour, R = rotate_img(img, mask, gt_contour, T_orig, K, -30, 30)

            new_R = np.dot(R, R_orig)
            pose[:3, :3] = new_R

        return img, mask, gt_contour, pose

    def get_heatmap(self, pose, K, keypoints, img):
        keypoints_2d = project(keypoints, K, pose)
        heatmap = generate_heatmap(keypoints_2d, img.shape[0], img.shape[1])
        return heatmap


    def __getitem__(self, index):

        path = self.data_paths[index]
        if self.is_training:
            img, mask, pose, K, gt_contour = self.get_data(path)
        else:
            img, pose, K = self.get_data(path)
            heatmap = self.get_heatmap(pose, K, self.corners, img)

        if self.is_training:
            img, mask, gt_contour, pose = self.augment(img, mask, gt_contour, pose, K)
            heatmap = self.get_heatmap(pose, K, self.corners, img)
            if path["type"] == "true":
                # random rotation and resize
                img, gt_contour, mask, heatmap = self.random_rotation_and_resize(img, gt_contour, mask, heatmap)
                # random translation
                img, gt_contour, mask, heatmap = self.random_translation(img, gt_contour, mask, heatmap)
            else:
                img, gt_contour, mask, heatmap = self.random_rotation_and_resize(img, gt_contour, mask, heatmap, is_render=True)
                img, gt_contour, mask, heatmap = self.random_translation(img, gt_contour, mask, heatmap, is_render=True)
            # random background
            img = self.random_background(img, mask)

            # random light
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-5, 5)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if self.is_training:
            gt_contour = gt_contour / 255
            gt_contour = np.expand_dims(gt_contour, axis=2)

        img = img / 255.0
        img -= [0.419, 0.427, 0.424]
        img /= [0.184, 0.206, 0.197]

        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        img = torch.tensor(img, dtype=torch.float32).permute((2, 0, 1))
        if self.is_training:
            gt_contour = torch.tensor(gt_contour, dtype=torch.int8).permute((2, 0, 1))
            return img, heatmap, K, pose, gt_contour
        return img, heatmap, K, pose

    def __len__(self):
        return len(self.data_paths)

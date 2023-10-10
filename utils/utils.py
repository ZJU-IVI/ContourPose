# import PIL
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import torch
#from torch_geometric.data import Data
from sklearn.neighbors import KDTree
from scipy.linalg import logm
from numpy import linalg as LA

def load_ply(path):
    """
    Loads a 3D mesh model from a PLY file.

    :param path: Path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
    'faces' (mx3 ndarray) - the latter three are optional.
    """
    f = open(path, 'r')

    n_pts = 0
    n_faces = 0
    face_n_corners = 3 # Only triangular faces are supported
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False

    # Read header
    while True:
        line = f.readline().rstrip('\n').rstrip('\r') # Strip the newline character(s)
        if line.startswith('element vertex'):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith('element face'):
            n_faces = int(line.split()[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith('element'): # Some other element
            header_vertex_section = False
            header_face_section = False
        elif line.startswith('property') and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split()[-1], line.split()[-2]))
        elif line.startswith('property list') and header_face_section:
            elems = line.split()
            if elems[-1] == 'vertex_indices':
                # (name of the property, data type)
                face_props.append(('n_corners', elems[2]))
                for i in range(face_n_corners):
                    face_props.append(('ind_' + str(i), elems[3]))
            else:
                print('Warning: Not supported face property: ' + elems[-1])
        elif line.startswith('format'):
            if 'binary' in line:
                is_binary = True
        elif line.startswith('end_header'):
            break

    # Prepare data structures
    model = {}
    model['pts'] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

    pt_props_names = [p[0] for p in pt_props]
    is_normal = False
    if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
        is_normal = True
        model['normals'] = np.zeros((n_pts, 3), np.float)

    is_color = False
    if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
        is_color = True
        model['colors'] = np.zeros((n_pts, 3), np.float)

    is_texture = False
    if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):
        is_texture = True
        model['texture_uv'] = np.zeros((n_pts, 2), np.float)

    formats = { # For binary format
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uchar': ('B', 1)
    }

    # Load vertices
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                      'red', 'green', 'blue', 'texture_u', 'texture_v']
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model['pts'][pt_id, 0] = float(prop_vals['x'])
        model['pts'][pt_id, 1] = float(prop_vals['y'])
        model['pts'][pt_id, 2] = float(prop_vals['z'])

        if is_normal:
            model['normals'][pt_id, 0] = float(prop_vals['nx'])
            model['normals'][pt_id, 1] = float(prop_vals['ny'])
            model['normals'][pt_id, 2] = float(prop_vals['nz'])

        if is_color:
            model['colors'][pt_id, 0] = float(prop_vals['red'])
            model['colors'][pt_id, 1] = float(prop_vals['green'])
            model['colors'][pt_id, 2] = float(prop_vals['blue'])

        if is_texture:
            model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
            model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

    # Load faces
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == 'n_corners':
                    if val != face_n_corners:
                        print('Error: Only triangular faces are supported.')
                        print('Number of face corners: ' + str(val))
                        exit(-1)
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == 'n_corners':
                    if int(elems[prop_id]) != face_n_corners:
                        print('Error: Only triangular faces are supported.')
                        print('Number of face corners: ' + str(int(elems[prop_id])))
                        exit(-1)
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model['faces'][face_id, 0] = int(prop_vals['ind_0'])
        model['faces'][face_id, 1] = int(prop_vals['ind_1'])
        model['faces'][face_id, 2] = int(prop_vals['ind_2'])

    f.close()

    return model

def read_ply_to_data(path):

    model = load_ply(path)
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]
    x = model['colors']

    x = x / 255.0
    x -= mean
    x /= std
    x = np.concatenate([x,model['pts'],model['normals']],axis=-1)
    x = torch.tensor(x,dtype=torch.float32)
    
    pos = torch.tensor(model['pts'],dtype=torch.float32)
    face = torch.tensor(model['faces'],dtype=torch.long).transpose(1,0)
    data = Data(x = x, pos=pos,face = face)
    return data

def read_mask(path, split, cls_idx=1):
    if split == "train" or split == "test":
        return (np.asarray(Image.open(path))[:, :, 0] != 0).astype(np.uint8)
    elif split == "fuse":
        return (np.asarray(Image.open(path)) == cls_idx).astype(np.uint8)
    elif split == "render":
        return (np.asarray(Image.open(path))).astype(np.uint8)


def mask_iou(self, output, batch):
    mask_pred = torch.argmax(output["seg"], dim=1)[0].detach().cpu().numpy()
    mask_gt = batch["mask"][0].detach().cpu().numpy()
    iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
    self.mask_ap.append(iou > 0.7)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def cal_error(S, y, img_shape=(480, 640)):
    S = S[:, y[0, :, 0], :]
    S = S.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    S = np.argmax(S, axis=-1)
    S = S.reshape(-1)
    y = y[:, :, 1].reshape(-1)

    gt_pos = []
    for idx in y:
        v = math.floor(idx / img_shape[1])
        u = idx - img_shape[1] * v
        gt_pos.append([u, v])

    est_pos = []
    for idx in S:
        v = math.floor(idx / (img_shape[1] / 2)) * 2
        u = (idx - img_shape[1] / 2 * (v / 2)) * 2
        est_pos.append([u, v])
 
    gt_pos = np.array(gt_pos, dtype=np.float32)
    est_pos = np.array(est_pos, dtype=np.float32)
    error = np.abs(gt_pos - est_pos)
    dist = np.sqrt(error[0] ** 2 + error[1] ** 2)
    avg_error = np.mean(dist)
    sigma = np.std(dist)

    return avg_error, sigma
def compute_add_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred
    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
        mean_distances[i] = np.mean(distance)

    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def compute_add(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt = pose_gt[:,:3]
    t_gt = pose_gt[:,3:]
    R_pred = pose_pred[:, :3]
    t_pred = pose_pred[:, 3:]
    pts_xformed_gt = R_gt * pts3d.transpose() + t_gt
    pts_xformed_pred = R_pred * pts3d.transpose() + t_pred
    distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
    mean_distances = np.mean(distance)

    threshold = diameter * percentage
    if mean_distances < threshold:
        return True
    else:
        return False


def compute_add_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred
    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
        mean_distances[i] = np.mean(distance)

    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score
def compute_adds_score(pts3d, diameter, pose_gt, pose_pred,percentage=0.1):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred

    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            mean_distances[i] = np.inf
            continue
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
        distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
        mean_distances[i] = np.mean(distance)
    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def mesh_project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = xyz.astype(np.float32)
    K = K.astype(np.float32)
    RT = RT.astype(np.float32)
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    z = xyz[:, 2].copy()
    xyz = np.dot(xyz, K.astype(np.float32).T)
    xyz = xyz / xyz[:, 2:]

    xyz[:, 2] = z
    return xyz

def find_neighborhold_node(model):
    pts = model["pts"]
    faces = model["faces"]
    neighbors = [[] for i in range(pts.shape[0])]
    for i in range(pts.shape[0]):
        dim0, dim1 = np.where(faces == i)
        for idx in faces[dim0]:
            for id in idx:
                if id not in neighbors[i] and id != i:
                    neighbors[i].append(id)

    return neighbors


def bbox_from_mask(mask_img, stride=0):

    mask_img = np.array(mask_img)
    mask = mask_img[:, :, 0]
    img_shape = mask.shape
    coor = np.nonzero(mask)
    coor[0].sort()
    xmin = coor[0][0]
    xmax = coor[0][-1]
    coor[1].sort()
    ymin = coor[1][0]
    ymax = coor[1][-1]

    if xmin >= stride:
        xmin -= stride
    else:
        xmin = 0
    if xmax + stride <= img_shape[0]:
        xmax += stride
    else:
        xmax = img_shape[0]

    if ymin >= stride:
        ymin -= stride
    else:
        ymin = 0

    if ymax + stride <= img_shape[1]:
        ymax += stride
    else:
        ymax = img_shape[1]

    return xmax, ymax, xmin, ymin


def concate_graph(x, edge, attribute):

    batch_size = x.shape[0]
    x_num = 0
    if x.ndim == 3:
        x_num = x.shape[1]
    elif x.ndim == 4:
        x_num = x.shape[1] * x.shape[2]
    x = x.reshape(-1, x.shape[-1])
    for i in range(batch_size):
        edge[i, :, :] += i * x_num

    edge = edge.permute(0, 2, 1)
    edge = edge.reshape(-1, 2)
    edge = edge.permute(1, 0)
    attribute = attribute.reshape(-1, attribute.shape[-1])

    return [x, edge, attribute]

def adjust_learning_rate_tless(optimizer, epoch, init_lr,loss):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.5 ** (epoch // 20))
    print("LR:{}".format(lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.5 ** (epoch // 20))
    print("LR:{}".format(lr))
    writer1.add_scalar('lr', lr, global_step=epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def draw_error(S, y, image):
    torch.cuda.is_available()
    S = S[:, y[0, :, 0], :]
    S = S.detach().cpu().numpy()
    batch_size = S.shape[0]
    y = y.detach().cpu().numpy()
    img = image.detach().cpu().numpy()[0]

    S = np.argmax(S, axis=-1)
    S = S.reshape(-1)
    y = y[:, :, 1].reshape(-1)
    gt_pos = []
    for idx in y:
        v = math.floor(idx / img.shape[1])
        u = idx - img.shape[1] * v
        gt_pos.append([u, v])
    est_pos = []
    for idx in S:
        v = math.floor(idx / (img.shape[1] / 2)) * 2
        u = (idx - img.shape[1] / 2 * (v / 2)) * 2
        est_pos.append([u, v])
    gt_pos = np.array(gt_pos, dtype=np.float32)
    est_pos = np.array(est_pos, dtype=np.float32)


if __name__ == "__main__":

    img = plt.imread("/home/reflex/data/LINEMOD/ape/mask/0000.png")
    img = np.array(img)
    bbox_from_mask(img)

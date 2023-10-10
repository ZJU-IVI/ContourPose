# zhz
# 这个用来简单输入姿态和图片进行渲染，看看预测对不对
#

import pygame
from pygame.locals import *
import cv2
import scipy.misc
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os
import time
import shutil
from ruamel import yaml
import copy
import random
from read_face_stl import stl_model
from tqdm import tqdm


def creatC2m(rM2c, tM2c):  # 输入为r 3*3   t为 1*3
    R = rM2c
    T = np.dot(-rM2c.T, tM2c.reshape(3, 1)).reshape(1, 3)
    return R, T


def T2vec(R):  # 矩阵分解为旋转矩阵和平移向量
    r = R[:3, :3]
    t = R[:3, 3]
    return r, t


def vec2T(r, t):  # 将旋转和平移向量转换为rt矩阵  r和t都是3*1
    t = t.reshape(3, 1)
    R = cv2.Rodrigues(r)
    rtMatrix = np.c_[np.r_[R[0], np.array([[0, 0, 0]])], np.r_[t, np.array([[1]])]]
    return rtMatrix


def cube(tri):  # 提取一堆点
    glBegin(GL_TRIANGLES)  # 绘制多个三角形
    for Tri in tri:
        glColor3fv(Tri['colors'])
        glVertex3fv(
            (Tri['p0'][0], Tri['p0'][1], Tri['p0'][2]))
        glVertex3fv(
            (Tri['p1'][0], Tri['p1'][1], Tri['p1'][2]))
        glVertex3fv(
            (Tri['p2'][0], Tri['p2'][1], Tri['p2'][2]))

    glEnd()  # 实际上以三角面片的形式保存


def draw_cube_test(worldOrientation, worldLocation, tri, window, display):

    glPushMatrix()


    pos = worldLocation[0]

    rm = worldOrientation.T

    rm[:, 0] = -rm[:, 0]
    rm[:, 1] = -rm[:, 1]

    xx = np.array([rm[0, 0], rm[1, 0], rm[2, 0]])
    yy = np.array([rm[0, 1], rm[1, 1], rm[2, 1]])
    zz = np.array([rm[0, 2], rm[1, 2], rm[2, 2]])
    obj = pos + zz

    gluLookAt(pos[0], pos[1], pos[2], obj[0], obj[1], obj[2], yy[0], yy[1], yy[2])
    cube(tri)
    glPopMatrix()
    pygame.display.flip()

    string_image = pygame.image.tostring(window, 'RGB')
    temp_surf = pygame.image.fromstring(string_image, display, 'RGB')

    tmp_arr = pygame.surfarray.array3d(temp_surf)

    #print(np.sum(tmp_arr))

    return (tmp_arr)  # 得到最后的图


def readYaml(yaml_file):
    gt_dic = {}
    f = open(yaml_file, 'r', encoding='utf-8')
    cfg = f.read()
    d = yaml.safe_load(cfg)
    return d


#def init(width, height):



    #return display, window
def init2():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #glClearColor(1,1,1,0.0)
    scale = 0.0001
    fx = intrinsic[0][2]  # 相机标定矩阵的值
    fy = intrinsic[1][2]
    cx = intrinsic[0][0]
    cy = intrinsic[1][1]
    glFrustum(-fx * scale, (width - fx) * scale, -(height - fy) * scale, fy * scale, (cx + cy) / 2 * scale, 20)  # 透视投影

    glMatrixMode(GL_MODELVIEW)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)  # 设置深度测试函数
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POINT_SMOOTH)
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_FILL)





def creatImg(obj_index,info):
    W_Rm2c = info[:, :3]
    W_Lm2c = info[:, 3:].reshape(1,-1)
    W_Rm2c = cv2.Rodrigues(np.matrix(W_Rm2c))[0]
    W_Lm2c = np.array([[W_Lm2c[0][0], W_Lm2c[0][1], W_Lm2c[0][2]]]).reshape(3, 1)
    rt = vec2T(W_Rm2c, W_Lm2c)
    W_Rm2c, W_Lm2c = T2vec(rt)
    tri = stl_model(os.path.join(stlPath, obj_index)).tri

    W_Rc2m, W_Lc2m = creatC2m(W_Rm2c, W_Lm2c)
    im = draw_cube_test(W_Rc2m, W_Lc2m, tri, window, display)
    im2 = np.zeros((height, width, 3))
    for m in range(height):
        for n in range(width):
            im2[m, n] = im[n, m]  # 彩色的边缘面

    return im2


def creatBottom(width, height):  # 生成要求大小的bottom图片
    img = np.zeros((height, width), dtype=np.uint8)
    bottom = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(3):
        bottom[:, :, i] = 0
    return bottom


pic_num = 3  # 用来做验证的图片数量

trainDic = {'head': 'obj', 'imgNum': 660, 'dirNum': 38, 'gtFile': 'gt.yml'}
testDic = {'head': 'scene', 'imgNum': 416, 'dirNum': 32, 'gtFile': 'gt.yml'}

listDirName = ['Testing images(2448#2048)', 'Testing images(512#512)', 'Training images(512#512)',
               'TrainingglFrustum(-fx * scale, (width - fx) * scale, -(height - fy) * scale, fy * scale, (cx + cy) / 2 * scale, 20)  # 透视投影 images(2448#2048)']
# listDirName = ['Testing images(512#512)']

'------------------------pygame初始化------------------------------'
width = 2448
height = 2048
#creatBottom(width, height)draw
pygame.init()
display = (width, height)
window = pygame.display.set_mode(display, DOUBLEBUF | OPENGLBLIT | OPENGL)


# display, window = init(width, height)#初始化



'------------------------pygame初始化结束------------------------------'
stlPath = os.path.join("/home/lqz/liquanzhi/Edge_Pose/stl/face-stl")
img = cv2.imread("received_image.jpg")

intrinsic = np.array([[2337.1354059537,0,1231.74452654278],[0,2336.97909090459,1048.86628248792],[0,0,1]])

# train_pose = yaml.load(open(os.path.join("/home/lqz/liquanzhi/PoseDataset(1)", 'gt.yml'), 'r'))
# instance_gt = train_pose["1"][2]
# R = np.array(instance_gt['m2c_R']).reshape(3, 3)
# t = np.array(instance_gt['m2c_T']).reshape(3, 1)
# pose = np.concatenate([R, t], axis=1)  # 合成3x4位姿矩阵

# pose = np.array([[-0.9873088  ,-0.1580538 , -0.01550228  ,0.0429814 ],
#  [-0.13783599 , 0.90129637, -0.41068978, -0.09390491],
#  [ 0.07888322, -0.40334086, -0.91164334,  0.44795966]])
pose = np.load("m2c.npy")
print(pose)


init2()
im2 = creatImg("obj7",pose)
cv2.imwrite("test_render.png",im2)
render_img = cv2.imread("/home/lqz/liquanzhi/Edge_Pose/utils/test_render.png")
edge_img = cv2.Canny(render_img, 150, 200, 5, L2gradient=True)
element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
edge_img = cv2.dilate(edge_img, element)
edge_final = edge_img
mask_inv = cv2.bitwise_not(edge_final)
edge_final = cv2.cvtColor(edge_final, cv2.COLOR_GRAY2BGR)
edge_final[:, :, 0] = 0
edge_final[edge_final > 0] = 150
edge_final[:, :, 2] = 0
background_img = cv2.bitwise_and(img, img, mask=mask_inv)
last_img = cv2.add(edge_final, background_img)
cv2.imwrite("test_render2.png",last_img)




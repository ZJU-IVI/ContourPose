import threading

import pygame
from pygame.locals import *
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os
from ruamel import yaml
from stl.stl_model import stl_model


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


#def init(width, height):import ctypes

PyGILState_Ensure = ctypes.PyDLL(None).PyGILState_Ensure
PyGILState_Ensure.restype = ctypes.c_size_t
PyGILState_Ensure.argtypes = []

PyGILState_Release = ctypes.PyDLL(None).PyGILState_Release
PyGILState_Release.restype = None
PyGILState_Release.argtypes = [ctypes.c_size_t]

def do_something_without_gil(func):
    gstate = PyGILState_Ensure()
    try:
        func()
    finally:
        PyGILState_Release()



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
    im2 = np.array(np.transpose(im,(1,0,2)))
    return im2


def creatBottom(width, height):  # 生成要求大小的bottom图片
    img = np.zeros((height, width), dtype=np.uint8)
    bottom = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(3):
        bottom[:, :, i] = 0
    return bottom



'------------------------pygame初始化------------------------------'
width = 2448
height = 2048
pygame.init()
display = (width, height)
window = pygame.display.set_mode(display, DOUBLEBUF | OPENGLBLIT | OPENGL)

'-----------------------pygame初始化结束------------------------------'
# Note that this stl model represents each face of the part,
# rendering different faces into different colors,
# which can be generated using CAD software such as solidworks.
# Of course, you can also input a complete stl model, which will render in the normal direction.
stlPath = os.path.join(os.path.dirname(os.getcwd()), "stl/face-stl")
intrinsic = np.array([[2337.1354059537,0,1231.74452654278],[0,2336.97909090459,1048.86628248792],[0,0,1]])


def visualizeById(pose, obj_id_list, real_edge):
    render_img_path = "mutil_render.png"
    for obj_id in obj_id_list:
        img = cv2.imread("mutil_render.png")
        init2()
        obj = "obj{}".format(obj_id)
        im2 = creatImg(obj, pose[obj])
        cv2.imwrite("test_render.png", im2)
        render_img = cv2.imread("test_render.png")
        edge_img = cv2.Canny(render_img, 150, 200, 5, L2gradient=True)
        # 求置信度
        intersection = cv2.bitwise_and(edge_img, real_edge)
        if np.sum(edge_img) == 0:
            confidence = 0
        else:
            confidence = np.sum(intersection) / np.sum(edge_img)
        # 放入结果
        pose[obj] = (pose[obj], confidence)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edge_img = cv2.dilate(edge_img, element)
        edge_final = edge_img
        mask_inv = cv2.bitwise_not(edge_final)
        edge_final = cv2.cvtColor(edge_final, cv2.COLOR_GRAY2BGR)

        edge_final[:, :, 0] = 0
        edge_final[edge_final > 0] = 150
        edge_final[:, :, 2] = 0
        background_img = cv2.bitwise_and(img, img, mask=mask_inv)
        last_img = cv2.add(edge_final, background_img)
        cv2.imwrite(render_img_path, last_img)
    return render_img_path



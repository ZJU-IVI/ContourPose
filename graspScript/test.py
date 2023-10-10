import queue
import socket
import os
import cv2
import numpy as np
from eval import evaluator
import torch
from visualize_pose import visualize, visualizeById
import time

K = np.array([[2337.1354059537, 0, 1231.74452654278], [0, 2336.97909090459, 1048.86628248792], [0, 0, 1]])
dist = np.array([-0.0860476, 0.13340928, 0, 0, 0])


# 这个函数用来处理图片得到物体在相机坐标系下的位姿并返回
def pose_predict(img):
    # 格式转换 2448 * 2048 -> 640 * 480, 内参也要跟着变，还要做图像归一化
    resize_img = resize(img)
    resize_img = resize_img / 255.0
    resize_img -= [0.419, 0.427, 0.424]
    resize_img /= [0.184, 0.206, 0.197]
    resize_img = torch.tensor(resize_img, dtype=torch.float32).permute((2, 0, 1))
    pose = process_image_sync(resize_img)
    cv2.imwrite("mutil_render.png", img)
    # img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    real_edge = cv2.Canny(img, 80, 200, 5, L2gradient=True)
    render_img_path = visualizeById(pose, [7, 13, 18], real_edge)
    # print(pose)
    return pose, render_img_path


def resize(img):
    # 新的宽和高
    width = 1650
    height = 2200
    ratio = width / 640
    # 开始裁剪的区域
    start_x = 120
    start_y = 90

    # 图片裁剪+缩放
    img = img[start_y:start_y + width, start_x:start_x + height]
    img = cv2.resize(img, dsize=(640, 480))

    # 重算K
    cx = K[0][2]
    cy = K[1][2]
    new_cx = -start_x + cx
    new_cy = -start_y + cy
    new_K = np.array(K)
    new_K[0][2] = new_cx / ratio
    new_K[1][2] = new_cy / ratio
    new_K[1][1] /= ratio
    new_K[0][0] /= ratio
    # print(new_K)
    return img


def process_image(image_path):
    target_img = cv2.imread(image_path)

    # cv2.namedWindow("target_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("target_img", target_img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    m2c, render_img_path = pose_predict(target_img)
    npy_save_path = "m2c.npy"
    np.save(npy_save_path, m2c)
    return npy_save_path, render_img_path


# 调用这个函数，开启服务端，阻塞等待客户端发送图片
def server_get_img_and_process():
    server_ip = '0.0.0.0'
    server_port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_socket.close()
    server_socket.bind((server_ip, server_port))
    server_socket.listen(5)

    print("Waiting for connection...")
    client_socket, client_address = server_socket.accept()
    print(f"Connected with {client_address}")

    # 接收图片文件
    with open("received_image.jpg", "wb") as f:
        while True:
            data = client_socket.recv(1024)
            if "END_OF_FILE" in data.decode('utf-8', 'ignore'):
                f.write(data.replace(b"END_OF_FILE", b""))
                break
            f.write(data)

    print("Image received. Processing...")
    processed_file, render_img_path = process_image("received_image.jpg")

    # 发送处理后的npy文件
    with open(processed_file, "rb") as f:
        client_socket.sendall(f.read())

    print("Processed npy file sent.")

    # 发送两个文件有一定间隔，防止发送时客户端没有处于接收状态
    time.sleep(0.1)

    # 发送处理后的render图
    with open(render_img_path, "rb") as f:
        client_socket.sendall(f.read())

    print("Processed render img sent.")

    client_socket.close()
    server_socket.close()


# 调用这个函数，开启服务端，阻塞等待客户端发送图片
def server_get_imgs_and_process():
    server_ip = '0.0.0.0'
    server_port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_socket.close()
    server_socket.bind((server_ip, server_port))
    server_socket.listen(5)

    while True:
        try:
            print("Waiting for connection...")
            client_socket, client_address = server_socket.accept()
            print(f"Connected with {client_address}")

            # 接收图片文件
            with open("received_image.jpg", "wb") as f:
                while True:
                    data = client_socket.recv(10240)
                    if "END_OF_FILE" in data.decode('utf-8', 'ignore'):
                        f.write(data.replace(b"END_OF_FILE", b""))
                        break
                    f.write(data)

            print("Image received. Processing...")
            processed_file, render_img_path = process_image("received_image.jpg")

            # 发送处理后的npy文件
            with open(processed_file, "rb") as f:
                client_socket.sendall(f.read())
            client_socket.sendall(b"END_OF_FILE")
            print("Processed npy file sent.")
            time.sleep(1)
            # 发送处理后的npy文件
            with open(render_img_path, "rb") as f:
                client_socket.sendall(f.read())
            client_socket.sendall(b"END_OF_FILE")
            print("Processed render img sent.")

            client_socket.close()
        except Exception as e:
            print("Error occured!")
            continue
    server_socket.close()


def process_image_sync(image):
    result = {}
    img_queue_7.put(image)
    img_queue_13.put(image)
    img_queue_18.put(image)

    result["obj7"] = res_queue_7.get()
    result["obj13"] = res_queue_13.get()
    result["obj18"] = res_queue_18.get()

    return result


img_queue_7 = queue.Queue()
res_queue_7 = queue.Queue()

img_queue_13 = queue.Queue()
res_queue_13 = queue.Queue()

img_queue_18 = queue.Queue()
res_queue_18 = queue.Queue()

# start 3 threads
processor_7 = evaluator(img_queue_7, res_queue_7, 7)
processor_13 = evaluator(img_queue_13, res_queue_13, 13)
processor_18 = evaluator(img_queue_18, res_queue_18, 18)
# concurrent start
processor_7.start()
processor_13.start()
processor_18.start()

# server_get_imgs_and_process()

img = cv2.imread("/home/lqz/liquanzhi/PoseDataset(1)/rgb/27.jpg")
pose_predict(img)
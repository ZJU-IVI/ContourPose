import queue
import socket
import os
import cv2
import numpy as np
from eval import evaluator
import torch
from visualize_pose import visualize
K = np.array([[2337.1354059537, 0, 1231.74452654278], [0, 2336.97909090459, 1048.86628248792], [0, 0, 1]])
dist = np.array([-0.0860476, 0.13340928, 0 ,0 ,0])

# 这个函数用来处理图片得到物体在相机坐标系下的位姿并返回
def pose_predict(img):
    # 格式转换 2448 * 2048 -> 640 * 480, 内参也要跟着变，还要做图像归一化
    resize_img = resize(img)
    resize_img = resize_img / 255.0
    resize_img -= [0.419, 0.427, 0.424]
    resize_img /= [0.184, 0.206, 0.197]
    resize_img = torch.tensor(resize_img, dtype=torch.float32).permute((2, 0, 1))
    pose = process_image_sync(resize_img)
    # process_render_sync(img, pose)
    return pose



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
    new_K[0][2] = new_cx/ratio
    new_K[1][2] = new_cy/ratio
    new_K[1][1]/=ratio
    new_K[0][0]/=ratio
    # print(new_K)
    return img

def process_image(image_path):
    target_img = cv2.imread(image_path)

    # cv2.namedWindow("target_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("target_img", target_img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    m2c = pose_predict(target_img)
    npy_save_path = "m2c.npy"
    np.save(npy_save_path, m2c)
    return npy_save_path


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
            data = client_socket.recv(10240)
            if "END_OF_FILE" in data.decode('utf-8', 'ignore'):
                f.write(data.replace(b"END_OF_FILE", b""))
                break
            f.write(data)

    print("Image received. Processing...")
    processed_file = process_image("received_image.jpg")

    # 发送处理后的npy文件
    with open(processed_file, "rb") as f:
        client_socket.sendall(f.read())

    print("Processed file sent.")
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
        processed_file = process_image("received_image.jpg")

        # 发送处理后的npy文件
        with open(processed_file, "rb") as f:
            client_socket.sendall(f.read())

        print("Processed file sent.")
        client_socket.close()

    server_socket.close()


def process_image_sync(image):
    img_queue.put(image)
    result = res_queue.get()
    return result

def process_render_sync(img, pose):
    input_queue.put((img, pose))
    render_img = output_queue.get()
    return render_img

img_queue = queue.Queue()
res_queue = queue.Queue()

input_queue = queue.Queue()
output_queue = queue.Queue()

# start a thread
processor = evaluator(img_queue, res_queue)
# visualize_processor = visualize(input_queue, output_queue)
processor.start()
# visualize_processor.start()


server_get_imgs_and_process()



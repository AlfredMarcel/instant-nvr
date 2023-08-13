from PIL import Image
import cv2
import os
import numpy as np

# 文件夹路径
folder_path = "tmp"

# 获取文件夹中的所有图片文件
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

# 获取帧数和视图数
frame_count = len(image_files) // 22
view_count = 22

# 每个视图的宽度和高度
view_width = 512  # 根据实际情况调整
view_height = 512  # 根据实际情况调整

# 每行每列的视图数
rows = 4
cols = 6

# 创建一个空白的大图
big_image_width = cols * view_width
big_image_height = rows * view_height


# 创建一个OpenCV视频编写器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(f"out.mp4", fourcc, 30.0, (big_image_width, big_image_height))

# 拼接每一帧的22个视图
for i in range(frame_count):
    # 创建一个新的大图
    big_image = Image.new("RGB", (big_image_width, big_image_height))
    
    for j in range(view_count+1):
        if j == 4: continue
        # 读取图片
        image_path = os.path.join(folder_path, f"frame{i:04d}_view{j:04d}.png")
        image = Image.open(image_path)
        
        # 计算视图在大图上的位置
        row = j // cols
        col = j % cols
        x = col * view_width
        y = row * view_height
        
        # 将图片粘贴到大图上
        big_image.paste(image, (x, y))

    # 保存拼接后的大图
    big_image.save(f"frame{i:04d}.png")

    # 将大图逐帧写入视频文件
    frame_image_np = np.array(big_image)
    frame_image_bgr = cv2.cvtColor(frame_image_np, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_image_bgr)

# 释放视频编写器
video_writer.release()

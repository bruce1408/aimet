import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
# print(train_df.info())
# print(val_df.info())

def create_label_dirs(image_dir, data_df, mode="train"):
    # 遍历 DataFrame 中的每一行
    for index, row in tqdm(data_df.iterrows()):
        filename = row['filename']
        label = row['label']
        
        # 创建目标目录路径
        label_dir = os.path.join(os.path.join(image_dir, mode), label)
        
        # 如果目录不存在，则创建目录
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # 源文件路径
        src_file = os.path.join(image_dir, "images", filename)  # 假设图片在train目录下
        
        # 目标文件路径
        dest_file = os.path.join(label_dir, filename)
        
        # 复制文件到目标目录
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
        else:
            print(f"文件 {src_file} 不存在")


if __name__ == "__main__":
    # 数据集路径
    image_dir = "/home/bruce_ultra/workspace/data_sets/mini-imagenet"
    train_path = os.path.join(image_dir, "train.csv")
    val_path = os.path.join(image_dir, "val.csv")


    # 读取 CSV 文件
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    # create_label_dirs(image_dir, train_df, mode="train")
    create_label_dirs(image_dir, val_df, mode="val")
    
    
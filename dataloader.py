import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd

# --- 第一步：定义作者映射字典 ---
def create_writer_mapping(csv_path):
    df = pd.read_csv(csv_path)
    # 获取所有唯一的作者ID并排序
    unique_writers = sorted(df['writer_id'].unique())
    # 创建字典：{'W01': 0, 'W02': 1, ...}
    writer_to_idx = {name: i for i, name in enumerate(unique_writers)}
    return writer_to_idx

# --- 第二步：定义 Dataset 类 ---
class CircleBinaryDataset(Dataset):
    def __init__(self, csv_path, npy_dir, writer_mapping):
        """
        npy_dir: 存放 .npy 文件的 processed 文件夹路径
        writer_mapping: 上一步创建的作者映射字典
        """
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.writer_mapping = writer_mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 获取 image_id 并构造文件名 (补齐5位数字)
        image_id = self.df.iloc[idx]['image_id']
        filename = f"{int(image_id):05d}.npy"
        file_path = os.path.join(self.npy_dir, filename)

        # 2. 加载数据
        # 如果文件不存在，这里会报错，建议确保 processed 文件夹已生成
        data = np.load(file_path)

        # 3. 转换为 Tensor (C, H, W) -> (1, 224, 224)
        image_tensor = torch.from_numpy(data).unsqueeze(0).float()

        # 4. 获取标签
        writer_id = self.df.iloc[idx]['writer_id']
        label = self.writer_mapping[writer_id]

        return image_tensor, label

# --- 第三步：主程序入口 ---
if __name__ == "__main__":
    # 配置路径
    CSV_PATH = "train.csv"     # 你的CSV文件
    NPY_DIR = "processed"      # 之前保存 .npy 的文件夹

    # 1. 定义 writer_to_idx
    if os.path.exists(CSV_PATH):
        writer_to_idx = create_writer_mapping(CSV_PATH)
        print(f"成功定义 writer_to_idx，作者总数: {len(writer_to_idx)}")
    else:
        print(f"错误：找不到文件 {CSV_PATH}")
        # 这里手动造一个示例防止报错，实际使用请确保CSV存在
        writer_to_idx = {"W01": 0} 

    # 2. 实例化 Dataset
    dataset = CircleBinaryDataset(
        csv_path=CSV_PATH, 
        npy_dir=NPY_DIR, 
        writer_mapping=writer_to_idx
    )

    # 3. 实例化 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0 # Windows下建议先设为0调试，稳定后再改大
    )

    # 4. 测试一个批次
    try:
        images, labels = next(iter(dataloader))
        print(f"数据加载成功！Batch形状: {images.shape}")
    except Exception as e:
        print(f"加载失败: {e}")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import numpy as np
import pandas as pd
import os
import json
# ==========================================
# 1. 数据集定义与工具函数
# ==========================================

import pandas as pd
import json
import os

def create_writer_mapping(csv_path, save_path="writer_mapping.json"):
    """
    根据 CSV 统计作者，建立映射并保存。
    csv_path: 你的 train.csv 路径
    save_path: 想要保存的 json 文件名
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")
        
    df = pd.read_csv(csv_path)
    # 获取唯一的作者并排序，确保每次运行生成的 ID 顺序一致
    unique_writers = sorted(df['writer_id'].unique())
    
    # 在这里定义 writer_to_idx
    writer_to_idx = {name: i for i, name in enumerate(unique_writers)}
    
    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(writer_to_idx, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 映射字典已保存至: {save_path}")
    print(f"✅ 成功定义 writer_to_idx，作者总数: {len(writer_to_idx)}")
    
    return writer_to_idx

class CircleBinaryDataset(Dataset):
    """自定义数据集：读取处理好的 .npy 浮点文件"""
    def __init__(self, csv_path, npy_dir, writer_mapping):
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.writer_mapping = writer_mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 根据 image_id 拼凑文件名，如 01004.npy
        image_id = self.df.iloc[idx]['image_id']
        filename = f"{int(image_id):05d}.npy"
        file_path = os.path.join(self.npy_dir, filename)

        # 加载 二进制数据 (数据范围已处理为 -1 到 1)
        data = np.load(file_path)
        # 转为 Tensor，并增加通道维，形状为 [1, 224, 224]
        image_tensor = torch.from_numpy(data).unsqueeze(0).float()

        # 标签转换
        writer_id = self.df.iloc[idx]['writer_id']
        label = self.writer_mapping[writer_id]
        return image_tensor, label

# ==========================================
# 2. ResNet18 模型定义 (针对 1 通道修改)
# ==========================================

class CircleResNet(nn.Module):
    def __init__(self, num_classes=44):
        super(CircleResNet, self).__init__()
        # 加载标准 ResNet18
        # weights=models.ResNet18_Weights.DEFAULT 表示使用迁移学习（预训练权重）
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 【重点修改 1】：原版 ResNet18 输入是 RGB 3通道
        # 我们的 .npy 是灰度 1 通道，所以要重新定义第一层卷积
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 【重点修改 2】：原版输出是 1000 类
        # 我们的任务只有 44 个作者，所以修改全连接层（fc）
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. 训练与验证主程序
# ==========================================

def main():
    # --- 参数配置 ---
    CSV_PATH = "train.csv"
    NPY_DIR = "processed"
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备字典与数据
    writer_to_idx = create_writer_mapping(CSV_PATH)
    full_dataset = CircleBinaryDataset(CSV_PATH, NPY_DIR, writer_to_idx)

    # 2. 9:1 划分训练集和验证集
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"数据划分完成：训练集 {train_size} 张，验证集 {val_size} 张")

    # 3. 实例化模型、损失函数、优化器
    model = CircleResNet(num_classes=len(writer_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss() # 分类任务标配：交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam 优化器收敛快

    # --- 训练循环 ---
    best_acc = 0.0
    for epoch in range(EPOCHS):
        # A. 训练阶段
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()           # 梯度清零
            outputs = model(images)         # 前向传播
            loss = criterion(outputs, labels) # 计算误差
            loss.backward()                 # 反向传播
            optimizer.step()                # 更新权重
            train_loss += loss.item()

        # B. 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad(): # 验证时不计算梯度，省内存
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1) # 取概率最大的那个分类
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {acc:.2f}%")

        # 保存表现最好的模型权重
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_circle_model.pth")
            print("  --> 发现更好的模型，已保存！")

    print(f"训练结束！最佳验证集准确率: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
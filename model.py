import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import numpy as np
import pandas as pd
import os
import torch_directml
import json

# ==========================================
# 1. 映射管理逻辑
# ==========================================

def get_or_create_mapping(csv_path, save_path="writer_mapping.json"):
    """建立统一的作者-索引映射表并存盘"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到训练集CSV: {csv_path}")
        
    df = pd.read_csv(csv_path)
    # 对所有唯一作者进行排序，确保 W01 始终对应 0，W05 始终对应 1...
    unique_writers = sorted(df['writer_id'].unique())
    writer_to_idx = {name: i for i, name in enumerate(unique_writers)}
    
    # 保存到本地 JSON，供后续预测脚本使用
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(writer_to_idx, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 映射字典已保存至: {save_path} (共 {len(writer_to_idx)} 位作者)")
    return writer_to_idx

# ==========================================
# 2. 增强型数据集定义
# ==========================================

class CircleBinaryDataset(Dataset):
    def __init__(self, csv_path, npy_dir, writer_mapping, model_type='C'):
        """
        model_type:
            'A': 识别编号 0-21 的作者，其他标记为 22
            'B': 识别编号 22-43 的作者，映射到 0-21，其他标记为 22
            'C': 识别全 44 位作者，标签为 0-43
        """
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.writer_mapping = writer_mapping
        self.model_type = model_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 读取图片数据
        image_id = self.df.iloc[idx]['image_id']
        filename = f"{int(image_id):05d}.npy"
        file_path = os.path.join(self.npy_dir, filename)
        
        data = np.load(file_path)
        image_tensor = torch.from_numpy(data).unsqueeze(0).float()

        # 获取原始作者 ID 字符串
        writer_str = self.df.iloc[idx]['writer_id']
        # 转换为 0-43 的全局标准索引
        raw_idx = self.writer_mapping[writer_str]

        # 根据模型类型重新定义标签
        if self.model_type == 'A':
            # 模型 A 只认前 22 个人
            if raw_idx < 22:
                label = raw_idx
            else:
                label = 22  # “陌生人”类别
        elif self.model_type == 'B':
            # 模型 B 只认后 22 个人
            if raw_idx >= 22:
                label = raw_idx - 22  # 重新映射到 0-21 方便训练
            else:
                label = 22  # “陌生人”类别
        else:
            # 模型 C 全认
            label = raw_idx

        return image_tensor, label

# ==========================================
# 3. ResNet18 模型架构
# ==========================================

class CircleResNet(nn.Module):
    def __init__(self, num_classes):
        super(CircleResNet, self).__init__()
        # 加载预训练 ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 适配 1 通道灰度输入
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改输出层
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 4. 训练核心函数
# ==========================================

def run_training(model_type='C'):
    # 配置
    CSV_PATH = "train.csv"
    NPY_DIR = "processed"
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 0.0001 # 较小的学习率适合微调
    device_dml = torch_directml.device()
    DEVICE = device_dml 
    print(f"✅ 正在使用加速设备: {torch_directml.device_name(0)}")

    # 1. 初始化映射
    writer_to_idx = get_or_create_mapping(CSV_PATH)
    
    # 2. 准备数据
    # A 和 B 模型有 22(成员) + 1(其他) = 23 个输出
    num_classes = 23 if model_type in ['A', 'B'] else 44
    full_dataset = CircleBinaryDataset(CSV_PATH, NPY_DIR, writer_to_idx, model_type=model_type)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 建模
    model = CircleResNet(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n--- 开始训练模型 {model_type} ---")
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out = model(imgs)
                _, pred = torch.max(out, 1)
                total += lbls.size(0)
                correct += (pred == lbls).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"best_model_{model_type}.pth")
            print(f"  ⭐ 模型 {model_type} 已保存！")

if __name__ == "__main__":
    # 你可以依次训练三个模型，或者注释掉不需要的
    run_training('A')
    run_training('B')
    run_training('C')
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
import os

# ==========================================
# 1. 配置与模型定义 (必须与训练时完全一致)
# ==========================================

class CircleResNet(nn.Module):
    def __init__(self, num_classes=44):
        super(CircleResNet, self).__init__()
        self.model = models.resnet18(weights=None) # 预测时不需要加载ImageNet权重
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_inverse_mapping(train_csv_path):
    """读取训练集CSV，重建 索引 -> 作者ID 的逆向字典"""
    df = pd.read_csv(train_csv_path)
    unique_writers = sorted(df['writer_id'].unique())
    # 建立 索引:作者名 字典，例如 {0: 'W01', 1: 'W02'}
    idx_to_writer = {i: name for i, name in enumerate(unique_writers)}
    return idx_to_writer

# ==========================================
# 2. 推理主程序
# ==========================================

def main():
    # --- 路径配置 ---
    TRAIN_CSV = "train.csv"          # 用于重建字典
    TEST_CSV = "test.csv"            # 测试集索引
    MODEL_PATH = "best_circle_model.pth" 
    NPY_DIR = "processed"       # 假设你处理好的测试集.npy存放在这里
    OUTPUT_FILE = "submission.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备逆向映射表
    print("正在构建作者映射表...")
    idx_to_writer = get_inverse_mapping(TRAIN_CSV)
    num_classes = len(idx_to_writer)

    # 2. 加载模型
    print(f"正在从 {MODEL_PATH} 加载模型...")
    model = CircleResNet(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # 切换到评估模式

    # 3. 读取测试集列表
    test_df = pd.read_csv(TEST_CSV)
    results = []

    print(f"开始预测，共 {len(test_df)} 张图片...")

    with torch.no_grad(): # 推理阶段不计算梯度
        for index, row in test_df.iterrows():
            img_id = row['image_id']
            # 构造 .npy 文件名，例如 10924 -> 10924.npy
            # 注意：如果你的文件名补齐了5位，请改为 f"{int(img_id):05d}.npy"
            npy_file = os.path.join(NPY_DIR, f"{int(img_id):05d}.npy")
            
            if not os.path.exists(npy_file):
                print(f"跳过：未找到文件 {npy_file}")
                continue

            # 加载并预处理数据
            data = np.load(npy_file)
            input_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

            # 模型预测
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            
            # 逆向转换回作者ID (Wxx)
            writer_id = idx_to_writer[predicted_idx.item()]
            
            results.append({
                "image_id": img_id,
                "writer_id": writer_id
            })

            if (index + 1) % 500 == 0:
                print(f"已处理 {index + 1} 张图片...")

    # 4. 保存结果
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 预测完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
import os

# ==========================================
# 1. 配置与模型定义
# ==========================================

class CircleResNet(nn.Module):
    def __init__(self, num_classes=44):
        super(CircleResNet, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_inverse_mapping(train_csv_path):
    df = pd.read_csv(train_csv_path)
    unique_writers = sorted(df['writer_id'].unique())
    idx_to_writer = {i: name for i, name in enumerate(unique_writers)}
    return idx_to_writer

# ==========================================
# 2. 推理主程序
# ==========================================

def main():
    # --- 路径与超参数配置 ---
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    MODEL_PATH = "best_circle_model.pth" 
    NPY_DIR = "processed"
    OUTPUT_FILE = "submission.csv"
    
    # 【核心配置】置信度阈值
    # 如果模型预测的最大概率低于这个值，则判定为未知作者 -1
    # 建议值：0.5 ~ 0.8，可以根据验证集表现调整
    THRESHOLD = 0.95
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_to_writer = get_inverse_mapping(TRAIN_CSV)
    num_classes = len(idx_to_writer)

    model = CircleResNet(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_df = pd.read_csv(TEST_CSV)
    results = []

    print(f"开始预测（阈值设定为: {THRESHOLD}），共 {len(test_df)} 张图片...")

    with torch.no_grad():
        for index, row in test_df.iterrows():
            img_id = row['image_id']
            npy_file = os.path.join(NPY_DIR, f"{int(img_id):05d}.npy")
            
            if not os.path.exists(npy_file):
                # 如果没找到处理好的文件，默认填 -1
                results.append({"image_id": img_id, "writer_id": "-1"})
                continue

            data = np.load(npy_file)
            input_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

            # 1. 获取模型输出 (Logits)
            output = model(input_tensor)
            
            # 2. 将输出转换为概率分布 (0.0 ~ 1.0)
            probs = torch.softmax(output, dim=1)
            
            # 3. 找到概率最大的类别及其概率值
            max_prob, predicted_idx = torch.max(probs, 1)
            
            # 4. 判定逻辑：是否足够可信？
            if max_prob.item() < THRESHOLD:
                final_writer = "-1"  # 概率太低，归为未知作者
            else:
                final_writer = idx_to_writer[predicted_idx.item()]
            
            results.append({
                "image_id": img_id,
                "writer_id": final_writer
            })

            if (index + 1) % 500 == 0:
                print(f"进度: {index + 1}/{len(test_df)}，当前置信度平均: {max_prob.item():.4f}")

    # 5. 保存结果
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 预测完成！结果已保存至: {OUTPUT_FILE}")
    print(f"统计：共有 {sum(1 for r in results if r['writer_id'] == '-1')} 个样本被判定为未知作者。")

if __name__ == "__main__":
    main()
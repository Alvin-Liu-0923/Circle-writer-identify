import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
import os
import json

# ==========================================
# 1. 模型架构定义 (保持一致)
# ==========================================

class CircleResNet(nn.Module):
    def __init__(self, num_classes):
        super(CircleResNet, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. 工具函数：加载模型与字典
# ==========================================

def load_trained_model(path, num_classes, device):
    model = CircleResNet(num_classes=num_classes).to(device)
    weights_only=False
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    return model

def get_inverse_mapping(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        writer_to_idx = json.load(f)
    return {i: name for name, i in writer_to_idx.items()}

# ==========================================
# 3. 核心联合预测逻辑
# ==========================================

def main():
    # --- 配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_CSV = "test.csv"
    MAP_JSON = "writer_mapping.json"
    NPY_DIR = "processed"
    OUTPUT_FILE = "submission_ensemble.csv"

    # 加载逆向字典
    idx_to_writer = get_inverse_mapping(MAP_JSON)

    # 1. 加载三个模型
    print("正在加载模型 A, B, C...")
    model_A = load_trained_model("best_model_A.pth", num_classes=23, device=DEVICE)
    model_B = load_trained_model("best_model_B.pth", num_classes=23, device=DEVICE)
    model_C = load_trained_model("best_model_C.pth", num_classes=44, device=DEVICE)

    test_df = pd.read_csv(TEST_CSV)
    results = []

    print(f"开始联合预测，共 {len(test_df)} 张图片...")

    with torch.no_grad():
        for index, row in test_df.iterrows():
            img_id = row['image_id']
            npy_file = os.path.join(NPY_DIR, f"{int(img_id):05d}.npy")
            
            if not os.path.exists(npy_file):
                results.append({"image_id": img_id, "writer_id": "-1"})
                continue

            # 读取数据并转为 Tensor
            data = np.load(npy_file)
            input_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

            # --- 分别获取三个模型的预测值和置信度 ---
            
            # 模型 A
            out_A = model_A(input_tensor)
            prob_A = torch.softmax(out_A, dim=1)
            conf_A, pred_A = torch.max(prob_A, 1)
            pred_A = pred_A.item()
            global_A = pred_A if pred_A < 22 else -1

            # 模型 B
            out_B = model_B(input_tensor)
            prob_B = torch.softmax(out_B, dim=1)
            conf_B, pred_B = torch.max(prob_B, 1)
            pred_B = pred_B.item()
            global_B = (pred_B + 22) if pred_B < 22 else -1

            # 模型 C
            out_C = model_C(input_tensor)
            prob_C = torch.softmax(out_C, dim=1)
            conf_C, pred_C = torch.max(prob_C, 1)
            global_C = pred_C.item()
            c_score = conf_C.item()

            # --- 应用判定规则 ---
            final_writer_id = "-1"

            # 规则 1: C与A相同，B输出-1
            if global_C == global_A and global_B == -1:
                final_writer_id = idx_to_writer[global_C]
            
            # 规则 2: C与B相同，A输出-1
            elif global_C == global_B and global_A == -1:
                final_writer_id = idx_to_writer[global_C]
            
            # 规则 3: A,B都不是-1，C与其中一方相同，且C置信度 > 0.9
            elif global_A != -1 and global_B != -1:
                if (global_C == global_A or global_C == global_B) and c_score > 0.9:
                    final_writer_id = idx_to_writer[global_C]
                else:
                    final_writer_id = "-1"
            
            # 其他情况全部输出 -1
            else:
                final_writer_id = "-1"

            results.append({
                "image_id": img_id,
                "writer_id": final_writer_id
            })

            if (index + 1) % 1000 == 0:
                print(f"已处理 {index + 1} 张图片...")

    # 保存结果
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 联合预测完成！结果已保存至: {OUTPUT_FILE}")
    
    # 打印简要统计
    neg_count = sum(1 for r in results if r['writer_id'] == "-1")
    print(f"统计：识别出有效作者: {len(results)-neg_count} 个，判定为未知(-1): {neg_count} 个。")

if __name__ == "__main__":
    main()
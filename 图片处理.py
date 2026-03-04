import cv2
import numpy as np
import os

def preprocess_single_image(img_path, target_size=224):
    """
    逻辑：裁剪正方形、10%边距、反色处理、Z-score归一化至 [-1, 1]、保存为浮点矩阵
    """
    # 1. 读入并寻找边界
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 高阈值捕捉浅色线
    _, thresh = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        return None
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # 2. 裁剪并做成正方形（10% Margin，居中补齐）
    side = max(w, h)
    margin = int(side * 0.1) 
    side_with_margin = side + 2 * margin
    
    # 初始背景为纯白 (255)
    square_img = np.full((side_with_margin, side_with_margin), 255, dtype=np.uint8)
    
    offset_x = (side_with_margin - w) // 2
    offset_y = (side_with_margin - h) // 2
    square_img[offset_y:offset_y+h, offset_x:offset_x+w] = img[y:y+h, x:x+w]
    
    # 3. 墨迹处理：Z-score 归一化映射至 [-1, 1]
    # 反色：背景变为 0，墨迹变为亮色高值
    square_img = 255 - square_img 
    
    # 创建浮点型输出矩阵，背景默认为 -1 (或者 0，取决于你模型的偏好，通常背景为 -1 较多)
    # 这里我们统一将背景设为 -1，墨迹在 -1 到 1 之间波动
    output_data = np.full(square_img.shape, -1.0, dtype=np.float32)
    
    mask = square_img > 5 
    if mask.any():
        square_img = square_img.astype(np.float32)
        ink_pixels = square_img[mask]
        
        mean = ink_pixels.mean()
        std = ink_pixels.std()
        
        if std > 1e-5:
            # 执行 Z-score 并映射
            # 通过 np.tanh 可以平滑地将所有值压缩到 [-1, 1] 之间
            # 或者简单的线性：(x - mean) / (3 * std) 锁定在 [-1, 1]
            z_score = (square_img[mask] - mean) / std
            output_data[mask] = np.clip(z_score / 3.0, -1.0, 1.0) 
        else:
            output_data[mask] = 0.0 # 若方差为0，设为中值
    
    # 4. 最终缩放
    # 插值使用 INTER_AREA 保持线条连续性
    final_base = cv2.resize(output_data, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return final_base

def batch_process(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建目录: {output_folder}")

    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    files.sort()

    print(f"开始处理，共发现 {len(files)} 张图片...")

    success_count = 0
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        processed_data = preprocess_single_image(input_path)
        
        if processed_data is not None:
            # 关键：保存为 .npy 格式，不再是 .png
            output_path = os.path.join(output_folder, filename.replace('.png', '.npy'))
            np.save(output_path, processed_data)
            
            success_count += 1
            if success_count % 100 == 0:
                print(f"进度: 已完成 {success_count}/{len(files)}")
        else:
            print(f"跳过无效图片: {filename}")

    print(f"处理完成！数据已保存至 '{output_folder}' 为 .npy 文件。")

if __name__ == "__main__":
    DATA_DIR = os.path.join("data", "images")
    SAVE_DIR = "processed"
    
    if os.path.exists(DATA_DIR):
        batch_process(DATA_DIR, SAVE_DIR)
    else:
        print(f"错误: 找不到输入文件夹 '{DATA_DIR}'。")
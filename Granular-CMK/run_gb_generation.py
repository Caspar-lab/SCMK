import os
import sys
import argparse
import numpy as np
import scipy.io as sio

# 自动将粒球核心代码文件夹加入系统路径，防止 Agent 运行时报 Import 错误
current_dir = os.path.dirname(os.path.abspath(__file__))

# 尝试导入
try:
    from GB import getGranularBall
    print("✅ 成功导入 GB 模块")
except ImportError as e:
    print(f"❌ 依然失败: {e}")
    # 如果还是失败，列出该目录下到底有什么
    print(f"该目录下的文件列表: {os.listdir(current_dir)}")

# 导入你提供的粒球核心计算函数
try:
    from GB import getGranularBall
except ImportError:
    print("❌ 错误：未能找到 GB.py 文件，请检查目录结构！")


def load_dataset(file_path):
    """
    自适应加载数据：支持 .mat (ODDS常见格式) 和 .npy / .csv 格式
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.mat':
        data = sio.loadmat(file_path)['trandata']
        # ODDS数据格式通常包含键值 'X' (特征) 和 'y' (标签)
        X = data[:,:-1]
        y = data[:,-1]
    elif ext == '.npy':
        matrix = np.load(file_path)
        X = matrix[:, :-1]
        y = matrix[:, -1]
    elif ext == '.csv':
        matrix = np.loadtxt(file_path, delimiter=',')
        X = matrix[:, :-1]
        y = matrix[:, -1]
    else:
        raise ValueError(f"不支持的数据格式: {ext}")
    return X.astype(np.float64), y.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="GCMK-AD 一阶段：表格数据正常模式粒球自适应生成")
    parser.add_argument("--data_dir", type=str, default="/Dataset/Belloney", help="文件目录")
    parser.add_argument("--data_name", type=str, default="wine.mat", help="数据集名称")
    parser.add_argument("--save_dir", type=str, default="GB_Results", help="粒球计算结果保存目录")
    args = parser.parse_args()

    # 使用绝对路径，这样无论你在哪个目录运行脚本都能找到
    data_path = "/Dataset/Belloney/wine.mat"

    if not os.path.exists(data_path):
        print(f"错误：找不到文件 {data_path}")

    # 1. 路径检查
    file_path = os.path.join(args.data_dir, args.data_name)
    if not os.path.exists(file_path):
        print(f"❌ 找不到指定的输入数据文件: {file_path}")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    dataset_base_name = os.path.splitext(args.data_name)[0]

    # 2. 数据加载与异常检测专属预处理
    print(f"🎬 开始处理数据集: {args.data_name}")
    X, y = load_dataset(file_path)
    print(f"📊 原始数据形态: 样本数={X.shape[0]}, 特征维度={X.shape[1]}")
    
    # 🌟 核心设计：遵循无监督/一分类异常检测范式，仅提取正常样本 (标签通常为0) 来构建基准正常粒球
    normal_idx = (y == 0)
    X_normal = X[normal_idx]
    print(f"🛡️ 过滤后的纯正常样本数: {X_normal.shape[0]} (已剔除异常噪点)")

    if X_normal.shape[0] < 10:
        print("⚠️ 正常样本数量过少，无法有效构建粒球表示！")
        return

    # 3. 运行你的自适应粒球覆盖算法
    print("🔮 正在启动多粒度空间自适应划分算法...")
    centers, sample_num, gb_index = getGranularBall(X_normal)

    print(f"✅ 粒球生成完毕！共生成 {len(centers)} 个自适应粒球。")

    # 5. 序列化保存结果，方便中端对比核网络 (Stage 2) 直接一键加载
    output_path = os.path.join(args.save_dir, f"{dataset_base_name}_gb_data.npz")
    np.savez(
        output_path,
        centers=centers,       # 粒球中心矩阵 [M, D]
        weights=sample_num,    # 粒球权重(含有点的数量) [M]
        X_normal=X_normal,     # 留存的原始正常样本，方便后续切分视图
        X_full=X,              # 完整的测试特征矩阵
        y_full=y               # 完整的测试标签
    )
    print(f"💾 粒球中间特征成功导出至: {output_path}\n" + "-"*50)


if __name__ == "__main__":
    main()
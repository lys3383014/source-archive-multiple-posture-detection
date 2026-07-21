"""
realtime_inference.py

端到端雷达动作识别全流程入口 (TorchScript 部署版)：
1. 调用 get_datamatrix.py 从硬件实时获取二维采样矩阵 [T, R]
2. 执行 TR 图像预处理管线
3. 加载 TorchScript (.pt) 模型，无需导入模型类定义
4. 输入模型进行推理，终端实时输出动作识别结果

运行方式:
python realtime_inference.py --checkpoint ./radar_nn1_4class_v2_scripted.pt
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.signal import butter, lfilter

# [模块导入 1] 导入底层硬件数据采集接口
from get_datamatrix import stream_datamatrix

# [已移除] 使用 TorchScript (.pt) 后不再需要导入模型类定义
# 原代码: from radar_nn1 import radar_nn1

#实时显示距离-时间图像
import cv2                      # [新增] 用于创建文件夹和路径拼接
from datetime import datetime       # [新增] 用于获取精确时间
import multiprocessing
import time

# =====================================================
# 1. 核心参数配置区 (预处理与雷达参数)
# =====================================================
# [雷达硬件采集参数]
RADAR_IP = "192.168.1.100"
SCAN_START_PS = 5000       # 扫描起始时间(皮秒)
SCAN_STOP_PS = 50000        # 扫描结束时间(皮秒)
INTERVAL_US = 20000      # 扫描间隔 4878us = 205Hz 采样率
WINDOW_SCANS = 200          # 每次采集的帧数(例如50Hz下, 96帧约等于1.92秒的数据)

# [距离选择与校正参数]
DISTANCE_MIN_M = 1.0          # 截取有效距离的下限(米)
DISTANCE_MAX_M = 7.0          # 截取有效距离的上限(米)
CABLE_COMPENSATION_M = 0.20   # 硬件电缆延迟距离补偿(米)

# [信号滤波参数]
DEFAULT_MTI_ALPHA = 0.95      # MTI (动目标指示) 滤波器的遗忘因子，越高背景更新越慢
TR1_LPF_ORDER = 3             # 低通滤波器阶数
TR1_LPF_CUTOFF_HZ = 3.2e9     # 低通滤波器截止频率
P440_FAST_TIME_FS_HZ = 16.387e9 # P440 快时间维度的等效采样率

# [模型输入与视觉化参数]
DEFAULT_VIS_SIZE = 280        # 视觉化特征图大小 (仅用于展示时)
DEFAULT_MODEL_SIZE = 224      # 神经网络实际需要的输入尺寸 (224x224)
FALL_ACTIONS = ['fall']

_LPF_B, _LPF_A = butter(N=TR1_LPF_ORDER, Wn=TR1_LPF_CUTOFF_HZ, btype='low', analog=False, output='ba',fs=P440_FAST_TIME_FS_HZ)

# =====================================================
# 2. 数据预处理管线 (矩阵 -> TR图)
# =====================================================
def ps_to_distance_m(time_ps: np.ndarray) -> np.ndarray:
    """皮秒时间转单程物理距离"""
    c = 299_792_458.0
    return 0.5 * c * (time_ps * 1e-12)


def build_range_axis_m(num_bins: int, scan_start_ps: float, scan_stop_ps: float) -> np.ndarray:
    """构建未补偿的原始距离轴"""
    time_axis_ps = np.linspace(scan_start_ps, scan_stop_ps, num_bins, dtype=np.float64)
    return ps_to_distance_m(time_axis_ps).astype(np.float32)


def build_tr_from_scan_matrix(scan_matrix: np.ndarray):
    """
    预处理全流程：将雷达原始二维矩阵 [T, R] 转化为：
      1) 模型所需的归一化灰度矩阵 tr_gray [224, 224] (经 viridis→灰度 映射，与训练数据一致)
      2) 用于实时窗口显示的 viridis 彩色矩阵 tr_color [224, 224, 3]
    返回: (tr_gray, tr_color)
    """
    if scan_matrix.ndim != 2 or scan_matrix.shape[0] < 2:
        raise ValueError("输入矩阵非法，至少需要2帧数据才能进行 MTI 滤波。")

    # [步骤 1] 距离轴补偿与截取
    raw_range_m = build_range_axis_m(scan_matrix.shape[1], SCAN_START_PS, SCAN_STOP_PS)
    corrected_range_m = raw_range_m - CABLE_COMPENSATION_M
    keep = (corrected_range_m >= DISTANCE_MIN_M) & (corrected_range_m <= DISTANCE_MAX_M)
    selected = scan_matrix[:, keep].astype(np.float32)

    # [步骤 2] 去直流偏移和准静态杂波 (沿慢时间维度减去均值)
    dc_removed = selected - np.mean(selected, axis=0, keepdims=True)
    #dc_removed = selected

    # [步骤 3] MTI 动目标显示滤波 (Alpha滤波提取动态前景)

    from scipy.signal import lfilter as lfilter_1d
    bg = lfilter_1d([1.0 - DEFAULT_MTI_ALPHA], [1.0, -DEFAULT_MTI_ALPHA], dc_removed, axis=0)
    mti = dc_removed - bg

    # [步骤 4] 取绝对值，并沿快时间维度(距离维)做三阶 IIR 低通滤波包络提取
    mti_abs = np.abs(mti)
    filtered = lfilter(_LPF_B, _LPF_A, mti_abs, axis=1)
    tr1 = np.maximum(filtered, 0.0)

    # [步骤 5] 线性归一化到 [0, 1] 区间
    x_min, x_max = float(np.min(tr1)), float(np.max(tr1))
    if x_max - x_min > 1e-8:
        norm = (tr1 - x_min) / (x_max - x_min)
    else:
        norm = np.zeros_like(tr1)

    # [步骤 6] 矩阵转置(x轴慢时间, y轴距离) 并利用 PIL 的双线性插值缩放到模型所需的 224x224
    norm = np.clip(norm.T, 0.0, 1.0)
    arr = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((DEFAULT_MODEL_SIZE, DEFAULT_MODEL_SIZE), resample = Image.Resampling.BILINEAR)

    tr_raw_gray = (np.asarray(pil, dtype=np.float32) / 255.0)
    tr_raw_gray = tr_raw_gray.T

    # [步骤 7] viridis 着色 → 生成彩色显示图 + 与训练数据一致的灰度图
    #viridis_cmap = cm.get_cmap('viridis')
    #viridis_cmap = cm.colormaps['viridis']
    viridis_cmap = plt.get_cmap('viridis')
    rgba = viridis_cmap(tr_raw_gray)                                  # [H, W, 4]
    tr_color = (rgba[..., :3] * 255).astype(np.uint8)               # [H, W, 3] 用于显示
    pil_rgb = Image.fromarray(tr_color)
    pil_gray = pil_rgb.convert('L')                                 # 模拟训练时 .convert('L')
    tr_gray = np.asarray(pil_gray, dtype=np.float32) / 255.0        # [H, W] 用于模型推理
    return tr_gray, tr_color


# =====================================================
# 3. 硬编码元数据 (从训练产出的 .pth 中提取，与训练时完全一致)
# =====================================================
# [硬编码] 索引→动作名映射表
#   含义：模型输出层有 4 个神经元，该字典将输出下标映射为动作名称。
#   来源：训练脚本 radar_nn1_4class.py，由数据集文件夹名按字母排序自动生成，
#         存储在 radar_nn1_4class_best.pth 的 idx_to_class 字段中。
#   ⚠️ 重新训练后若动作类别有增减或排序变化，必须同步更新此表！
IDX_TO_CLASS = {
    0: 'stand',
    1: 'walk',
    2: 'bend',
    3: 'fall',
}

# [硬编码] 图像标准化参数
#   含义：推理前对输入图像做 x = (x - MEAN) / STD，将像素值从 [0,1] 映射到 [-1,1]。
#         必须与训练时 radar_nn1.py 中 MEAN=0.5, STD=0.5 保持一致，否则模型预测会出错。
IMAGE_MEAN = 0.5
IMAGE_STD  = 0.5

# =====================================================
# 3b. 模型加载与推理调用 (TorchScript 版)
# =====================================================
def load_trained_model(checkpoint_path: str, device: torch.device):
    """
    [已改为 TorchScript 加载]
    使用 torch.jit.load 直接加载 .pt 文件，无需实例化模型类。
    元数据（类别映射、均值、标准差）从上方硬编码常量获取。
    """
    print(f"[*] 正在加载 TorchScript 模型: {checkpoint_path}...")

    # [改动核心] 原来: torch.load → radar_nn1() → load_state_dict
    #            现在: torch.jit.load 一步完成，模型已内含结构+权重+eval状态
    model = torch.jit.load(checkpoint_path, map_location=device)
    model.eval()

    num_classes = len(IDX_TO_CLASS)
    print(f"[*] 模型加载成功! 支持识别 {num_classes} 个动作类别。")
    return model, IDX_TO_CLASS, IMAGE_MEAN, IMAGE_STD


@torch.no_grad()
def infer_action(model, tr_gray: np.ndarray, device: torch.device, mean: float, std: float) -> np.ndarray:
    """
    前向推理函数
    """
    # 1. 扩充维度为 PyTorch 所需的格式: [Batch=1, Channel=1, H=224, W=224]
    x = torch.from_numpy(tr_gray).unsqueeze(0).unsqueeze(0)

    # 2. 图像标准化 (使用训练时的均值和方差)
    x = (x - mean) / std
    x = x.to(device)

    # 3. 执行模型前向传播
    # [已改动] TorchScript 模型的 forward 直接返回 logits (不再是元组)
    #   原代码: logits, _ = model(x, need_aux=False)
    #   原因: 导出 .pt 时用 Wrapped 类包装过，forward 只返回 logits
    logits = model(x)

    # 4. Softmax 转化为概率分布
    prob = F.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
    return prob


def collector_process(data_queue, ip, start_ps, stop_ps, interval_us, chunk_size, program_start_time):
    """独立进程：持续从雷达采集数据，放入队列"""
    for scan_matrix in stream_datamatrix(
            ip=ip,
            start_ps=start_ps,
            stop_ps=stop_ps,
            interval_us=interval_us,
            chunk_size=chunk_size,
            program_start_time=program_start_time
    ):
        if data_queue.full():
            try:
                data_queue.get_nowait()
            except Exception:
                pass
        data_queue.put(scan_matrix)

# =====================================================
# 4. 主干流水线 (实时采集 -> 预处理 -> 推理输出)
# =====================================================
def main():
    program_start_time = time.time()
    parser = argparse.ArgumentParser(description="雷达实时采集与动作识别端到端系统")
    parser.add_argument("--checkpoint", type=str, required=True, help="TorchScript 模型文件路径 (.pt)")
    parser.add_argument("--ip", type=str, default=RADAR_IP, help="雷达 IP 地址")
    args = parser.parse_args()

    # 初始化加速设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用计算设备: {device}")

    # 加载已有的模型
    model, idx_to_class, mean, std = load_trained_model(args.checkpoint, device)

    print("\n" + "="*55)
    print("🚀 P440 雷达动作识别系统实时监测已启动！")
    print(f"配置: 每次拦截 {WINDOW_SCANS} 帧数据 ({INTERVAL_US}us 间隔) 进行一次预判...")
    print("="*55 + "\n")

    # ================= [新增代码：初始化画图窗口] =================
    cv2.namedWindow('Real-time Radar Monitor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Real-time Radar Monitor', 600, 600)
    # ==============================================================

    # ===== 方案二改进：多进程采集（绕开GIL） =====
    data_queue = multiprocessing.Queue(maxsize=2)

    p = multiprocessing.Process(
        target=collector_process,
        args=(data_queue, args.ip, SCAN_START_PS, SCAN_STOP_PS, INTERVAL_US, WINDOW_SCANS, program_start_time),
        daemon=True
    )
    p.start()

    try:
        while True:
            scan_matrix = data_queue.get()  # 阻塞等待新数据
            t_preprocess_start = time.perf_counter()
            tr_gray, tr_color = build_tr_from_scan_matrix(scan_matrix)
            t_preprocess_end = time.perf_counter()
            print(f"[计时] 原始数据 → 灰度图预处理: {t_preprocess_end - t_preprocess_start:.3f}s")

            # 计时：推理
            t_infer_start = time.perf_counter()
            prob = infer_action(model, tr_gray, device, mean, std)
            t_infer_end = time.perf_counter()
            print(f"[计时] 模型推理: {t_infer_end - t_infer_start:.3f}s")

            # # 步骤 B：执行 TR 信号预处理
            # tr_gray, tr_color = build_tr_from_scan_matrix(scan_matrix)
            #
            # # 步骤 C：送入模型推理预测
            # prob = infer_action(model, tr_gray, device, mean, std)

            # 步骤 D：结果解析与终端输出
            pred_idx = int(np.argmax(prob))
            pred_class = idx_to_class[pred_idx]
            conf = float(prob[pred_idx])

            # ================= [核心修复：生成全局统一时间戳] =================
            current_time = datetime.now()
            # 终端和图表上显示的易读时间 (精确到秒)
            display_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            # ==================================================================


            top4_idx = np.argsort(-prob)[:4].tolist()
            top4_text = ', '.join(f"{idx_to_class[i]}: {prob[i]:.2%}" for i in top4_idx)

            # [修改] 终端输出中加入时间戳前缀，方便查阅
            print(f"[{display_time_str}] ✅ [识别结果] 动作: 【{pred_class}】 | 置信度: {conf:.2%}")
            print(f"   [各种动作置信度] {top4_text}")
            print("-" * 55)

            # ================= [新增代码：刷新图像与文字标注及警报] =================
            # 将 tr_color [224,224,3] 放大到显示尺寸
            display_img = cv2.resize(tr_color, (600, 600), interpolation=cv2.INTER_NEAREST)

            # OpenCV 使用 BGR 格式，需要从 RGB 转换
            display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

            # 摔倒警报：叠加红色半透明蒙层
            # if pred_class in FALL_ACTIONS:
            #      red_overlay = display_img.copy()
            #      red_overlay[:] = (0, 0, 255)  # BGR 红色
            #      display_img = cv2.addWeighted(display_img, 0.6, red_overlay, 0.4, 0)
            #      cv2.putText(display_img, "WARNING (FALL)",
            #                  (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # 左上角叠加识别信息文字
            info_color = (0, 0, 255) if pred_class in FALL_ACTIONS else (0, 200, 0)
            cv2.putText(display_img, f"Action: {pred_class}  Conf: {conf:.2%}",
                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
            cv2.putText(display_img, f"Time: {display_time_str}",
                         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 添加坐标轴标签（直接用文字标注）
            cv2.putText(display_img, "distance", (250, 590),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_img, "time", (5, 15),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # 显示并检测退出键
            cv2.imshow('Real-time Radar Monitor', display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



            # ====================================================================


    except KeyboardInterrupt:
        print("\n[!] 接收到退出信号(Ctrl+C)，程序安全终止。")
    except Exception as e:
        print(f"\n[!] 发生运行时异常: {e}")
    finally:
        # [新增] 退出时关闭交互模式，并保持窗口打开
        cv2.destroyAllWindows()
        print("\n[*] 监测已停止。")

if __name__ == "__main__":
    main()
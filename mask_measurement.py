# -*- coding: utf-8 -*-
"""
4f 光学频域掩膜数据定量评估系统 (终极严谨计量版)
核心规范化升级：
1. 规范直流抑制率 (DC Suppression)：分子与分母严格在相同几何平坦区 (¬M_edge) 统计，消除口径不一致。
2. 规范化命名：sparsity_ratio 正式重命名为 active_pixel_ratio (有效亮像素比)。
3. 暴露关键中间物理量：同时输出 n_edge, n_flat, curr_dc, base_dc，提供完整的可核验追溯链路。
4. 增强工程健壮性：自动清洗 PowerShell 拖入路径包含的 '&' 与引号，支持中文路径。
"""

import cv2
import numpy as np
import os
import sys

def imread_unicode(img_path):
    """支持中文与特殊字符的安全图片读取"""
    try:
        img_array = np.fromfile(img_path, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None

def clean_input_path(raw_path):
    """清洗 Windows / PowerShell 拖拽文件路径的特殊前缀与引号"""
    path = raw_path.strip()
    if path.startswith("&"):
        path = path[1:].strip()
    return path.strip("'").strip('"')

def calculate_metrics_rigorous(img, base_img=None):
    img_f = img.astype(np.float64)
    h, w = img.shape
    total_pixels = h * w

    # 1. 梯度幅值计算 (3x3 Sobel)
    sobelx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # 2. 自适应高梯度边缘像素集合 (M_edge)
    # 取全图梯度 P92 分位与保底阈值 20.0 的较大值
    edge_threshold = max(20.0, np.percentile(grad_mag, 92))
    edge_mask = grad_mag > edge_threshold
    n_edge = int(np.sum(edge_mask))

    if n_edge > 0:
        # 高梯度区梯度平方均值 (Tenengrad 能量)
        edge_tenengrad = float(np.mean(grad_mag[edge_mask]**2))
        # 边缘区 99% 分位峰值梯度
        edge_peak = float(np.percentile(grad_mag[edge_mask], 99))
    else:
        edge_tenengrad = 0.0
        edge_peak = 0.0

    # 3. 规范化有效亮像素比 (Active Pixel Ratio)
    # 灰度大于暗电平底噪阈值 20 的像素占比
    active_pixels = int(np.sum(img > 20))
    active_pixel_ratio = (active_pixels / total_pixels) * 100.0

    # 4. 严谨同口径直流抑制率 (DC Suppression)
    # 统计平坦区 (¬M_edge) 的背景均值衰减
    flat_mask = ~edge_mask
    n_flat = int(np.sum(flat_mask))
    curr_dc = float(np.mean(img_f[flat_mask])) if n_flat > 0 else float(np.mean(img_f))

    if base_img is not None:
        base_f = base_img.astype(np.float64)
        # 同步口径：分母严格取基准图在对应平坦区 (flat_mask) 的平均灰度
        if base_f.shape == img_f.shape and n_flat > 0:
            base_dc = float(np.mean(base_f[flat_mask]))
        else:
            base_dc = float(np.mean(base_f))
        
        dc_suppression = (1.0 - (curr_dc / (base_dc + 1e-6))) * 100.0
    else:
        base_dc = 255.0
        dc_suppression = (1.0 - (curr_dc / 255.0)) * 100.0

    return {
        "edge_tenengrad": edge_tenengrad,
        "edge_peak": edge_peak,
        "active_pixel_ratio": active_pixel_ratio,
        "dc_suppression": dc_suppression,
        "n_edge": n_edge,
        "n_flat": n_flat,
        "curr_dc": curr_dc,
        "base_dc": base_dc
    }

def main():
    print("=" * 115)
    print(" 📊 4f 光频域掩膜定量标定系统 (计量规范版：同几何口径抑制率 + 关键中间量溯源)")
    print("=" * 115)

    base_path_raw = input("👉 请输入【直拍/无掩膜基准图】路径 (如无则回车跳过): ")
    base_path = clean_input_path(base_path_raw)
    base_img = imread_unicode(base_path) if (base_path and os.path.exists(base_path)) else None

    if base_img is not None:
        print(f"✅ 已加载基准参考图: {os.path.basename(base_path)} | 分辨率: {base_img.shape[1]}x{base_img.shape[0]}")
    else:
        print("⚠️ 未载入基准图，直流抑制率将以理论满量程 255 归一化计算。")

    results = []
    print("\n👉 请依次拖入各掩膜测试图像路径 (输入 Q 完成退出)：")
    while True:
        raw_in = input("📷 请拖入图像: ")
        path = clean_input_path(raw_in)
        if path.lower() in ['q', 'exit', '']: 
            break
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            continue

        img = imread_unicode(path)
        if img is None:
            print(f"❌ 图像读取失败，请检查格式: {path}")
            continue

        res = calculate_metrics_rigorous(img, base_img)
        res['name'] = os.path.basename(path)
        results.append(res)
        print(f"   └─ 已解析: {res['name']:<18} | 边缘梯度: {res['edge_tenengrad']:<9.1f} | 抑制率: {res['dc_suppression']:<5.1f}% | 边缘像元数: {res['n_edge']}")

    if not results:
        print("未录入有效数据，程序退出。")
        return

    print("\n" + "=" * 115)
    print("📋 【4f 频域掩膜标定综合报告表】")
    print("-" * 115)
    print(f"{'图像名称':<20} | {'边缘能量(G_edge)':<14} | {'峰值陡峭(S_edge)':<14} | {'直流抑制率(%)':<12} | {'亮像素比(%)':<11} | {'n_edge':<8} | {'平坦均值(curr/base)'}")
    print("-" * 115)
    for r in results:
        dc_str = f"{r['curr_dc']:.1f} / {r['base_dc']:.1f}"
        print(f"{r['name']:<20} | {r['edge_tenengrad']:<16.1f} | {r['edge_peak']:<16.1f} | {r['dc_suppression']:<14.2f} | {r['active_pixel_ratio']:<13.2f} | {r['n_edge']:<8} | {dc_str}")
    print("=" * 115)
    print("📌 计量口径说明：")
    print(" 1. G_edge: 梯度前约 8% (P92，max(20)保底) 区域的梯度平方均值，避免大面积背景稀释。")
    print(" 2. S_edge: 高梯度区内部梯度幅值的 P99 分位数。")
    print(" 3. 直流抑制率: 分子与分母均严格统计在平坦非边缘区 (¬M_edge)，口径完全一致。")
    print(" 4. 亮像素比: 灰度 I > 20 的有效像素占画幅百分比。")
    print("=" * 115)

if __name__ == "__main__":
    main()

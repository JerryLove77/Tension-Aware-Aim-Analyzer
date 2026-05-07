# filepath: analyzer/math_utils.py

# ============================================================
# analyzer/math_utils.py
# 数学工具层 — 信号平滑与数值导数
# 纯函数，无副作用，零 UI 依赖
# ============================================================

import numpy as np
from scipy.signal import savgol_filter


def apply_smoothing(
    series: np.ndarray,
    window_length: int,
    polyorder: int = 3,
) -> np.ndarray:
    """对 1-D 序列应用 Savitzky-Golay 平滑滤波器。

    自动处理边缘情况：
      - 数据长度不足 → 直接返回原序列
      - ``window_length <= polyorder`` → 直接返回原序列
      - ``window_length`` 为偶数 → 自动 +1 变为奇数

    Args:
        series:   输入信号 (1-D numpy array)。
        window_length: 滤波窗口长度 (奇数)。
        polyorder: 多项式拟合阶数。

    Returns:
        平滑后的序列 (与输入形状相同)。
    """
    n = len(series)
    if n < window_length or window_length <= polyorder:
        return series

    if window_length % 2 == 0:
        window_length += 1

    if window_length > n:
        return series

    return savgol_filter(series, window_length, polyorder)


def calc_derivative(series: np.ndarray, fps: float) -> np.ndarray:
    """计算 1-D 序列对时间的导数（差分法）。

    使用 ``np.diff`` 计算差分，乘以 ``fps`` 转换为时间导数。
    为保持输出长度与输入一致，复制最后一个差分值并追加到末尾。

    Args:
        series: 输入序列（例如位置序列）。
        fps:    视频帧率 (frames / second)。

    Returns:
        导数序列 (长度与输入相同)。
    """
    diff = np.diff(series) * fps
    return np.append(diff, diff[-1])

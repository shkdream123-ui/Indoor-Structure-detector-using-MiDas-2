# Python 3.12
import socket
import struct
import numpy as np
import cv2
from collections import deque
import threading
import math
import time
import random
import matplotlib.pyplot as plt

HOST = '0.0.0.0'
PORT = 5000

EXPECTED_W = 256
EXPECTED_H = 256

# ------------------- 큐 -------------------
raw_queue = deque(maxlen=1)
processed_queue = deque(maxlen=1)
gyro_queue = deque(maxlen=1000)# 추후 프로젝트로 인하여 추가된 부분, 여기서 안쓰인다.
accel_queue = deque(maxlen=1000)# 추후 프로젝트로 인하여 추가된 부분, 여기서 안쓰인다.

# ------------------- Depth 노멀라이즈 -------------------
def normalize_depth(depth_frame):
    depth_frame = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = np.min(depth_frame)
    max_val = np.max(depth_frame)
    depth_norm = (depth_frame - min_val) / (max_val - min_val + 1e-6)
    return (depth_norm * 255).astype(np.uint8)

# ------------------- 가중치 계산 -------------------
def compute_depth_confidence(depth_small):
    depth_f = depth_small.astype(np.float32)

    # 1. gradient 계산
    grad_x = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # 2. 블록 단위로 downsample → 공간적 분포 파악 목적
    # 16x16 정도면 충분 (전체가 256x256이라면 16배 축소)
    block = cv2.resize(grad_mag, (16, 16), interpolation=cv2.INTER_AREA)

    # 3. 블록들의 분산 계산
    spatial_var = block.var()  # 하나의 수

    # 4. sigmoid로 0~1 스케일로 압축 (필요하면)
    # C 값은 조정 가능. variance가 0.0~0.01 정도 나온다면 C=0.005 정도
    C = 0.005
    confidence = 1.0 - math.exp(-spatial_var / C)

    return float(np.clip(confidence, 0.0, 1.0))


def compute_flow_weight(flow_mag_map, k=10.0, x0=0.5):
    """
    flow_mag_map : optical flow magnitude map
    k : sigmoid 기울기, 작게 하면 완만
    x0 : sigmoid 중앙값, W=0.5가 되는 std 기준
    """
    std = float(flow_mag_map.std())
    W = 1.0 / (1.0 + np.exp(-k * (std - x0)))
    return np.clip(W, 0.0, 1.0)

def compute_static_weight(gray_frame, NX, NY):
    gray_umat = cv2.UMat(gray_frame)
    small = cv2.resize(gray_umat, (NX//4, NY//4), interpolation=cv2.INTER_AREA).get()
    std = float(small.std())
    C = 50.0
    W = 1.0 - math.exp(-std / C)
    return np.clip(W, 0.0, 1.0)

#-------------------- 필터링 행렬 -----------------
def compute_depth_matrix(depth_uint8, threshold=0.7, k=20.0):
    # 1) 0~1 정규화
    depth_norm = depth_uint8.astype(np.float32) / 255.0

    # 2) 시그모이드 함수 적용
    #    threshold에서 0.5가 되도록 중심을 이동
    #    k는 기울기(가파름)
    W_depth = 1.0 / (1.0 + np.exp(-k * (depth_norm - threshold)))

    return W_depth

def compute_flow_matrix(flow_norm, f_wflow, k=20.0):
    """
    flow_norm: 이미 0~1로 정규화된 optical flow magnitude 값
    f_wflow : 유효값(= sigmoid의 중앙값, 출력이 0.5가 되는 지점)
    k : 기울기 조절 상수 (20~30 권장)
    """
    flow = np.clip(flow_norm - f_wflow,0.0,1.0)
    
    W_flow = 1.0 / (1.0 + np.exp(-k * (flow-0.5)))
    
    return W_flow

# ------------------edge 연결 함수 ----------------
def fill_edges(mask):
    # mask: numpy uint8 binary (0/255)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(mask_filled, contours, -1, 255, thickness=cv2.FILLED)
    return mask_filled

# ------------------- TCP 수신 -------------------
def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def receive_thread(sock, raw_queue, gyro_queue, accel_queue):
    """
    Packet format
    ------------------------------------------------
    [1B] packet_type
      0x01 : frame packet
        [4B] jpeg_len
        [4B] depth_len
        [jpeg bytes]
        [depth bytes]

      0x02 : gyro packet
        [8B] timestamp (ns)
        [4B] gyro_z (rad/s)

      0x03 : accel packet
        [8B] timestamp (ns)
        [4B] accel_x (m/s^2)
        [4B] accel_y (m/s^2)
        [4B] accel_z (m/s^2)

    ------------------------------------------------
    """

    while True:
        # -----------------------------
        # 1) packet type
        # -----------------------------
        pkt_type_raw = recv_all(sock, 1)
        if not pkt_type_raw:
            print("[RECV] connection closed")
            break

        pkt_type = pkt_type_raw[0]

        # -----------------------------
        # 2) FRAME PACKET
        # -----------------------------
        if pkt_type == 0x01:
            header = recv_all(sock, 8)
            if not header:
                break

            jpeg_len, depth_len = struct.unpack('!II', header)

            jpeg_bytes = recv_all(sock, jpeg_len)
            depth_bytes = recv_all(sock, depth_len)

            if jpeg_bytes is None or depth_bytes is None:
                print("[RECV] frame incomplete")
                break

            raw_queue.append(
                (jpeg_bytes, depth_bytes)
            )

        # -----------------------------
        # 3) GYRO PACKET  (추후 프로젝트로 인하여 추가된 부분, 여기서 안쓰인다.)
        # -----------------------------
        elif pkt_type == 0x02:
            payload = recv_all(sock, 12)
            if not payload:
                break

            timestamp, gyro_z = struct.unpack('!qf', payload)

            #print(f"Received gyro: timestamp={timestamp}, gyroZ={gyro_z}")

            gyro_queue.append(
                (timestamp, gyro_z)
            )

        # ------------------------------
        # 4) ACCEL PACKET
        # ------------------------------
        elif pkt_type == 0x03:
            payload = recv_all(sock, 20)
            if not payload:
                break

            timestamp, ax, ay, az = struct.unpack('!qfff', payload)

            pc_ts = time.monotonic_ns()

            #print(f"Received accel: timestamp={timestamp}, ax={ax}, ay={ay}, az={az}")

            accel_queue.append(
                (timestamp, pc_ts, ax, ay, az)
            )


        else:
            print(f"[RECV] unknown packet type: {pkt_type}")
            break

# -----------------------------------------------------------------------------------
# class
class OrientationSample:
    def __init__(self, theta, weight):
        self.theta = theta      # [0, π)
        self.weight = weight    # 중요도

class OrientationDistribution:
    def __init__(self, num_bins=180):
        self.num_bins = num_bins
        self.hist = np.zeros(num_bins, dtype=np.float32)

    def normalize(self):
        s = self.hist.sum()
        if s > 0:
            self.hist /= s

    def log(self):
        return np.log(self.hist + 1e-8)

class VonMisesComponent:
    def __init__(self, mu, kappa, weight):
        self.mu = mu
        self.kappa = kappa
        self.weight = weight

# --------------------------------------------------------------------------------------------
#공통 함수
def angle_diff(a, b):
    d = abs(a - b)
    return min(d, np.pi - d)

def normalize_angle(theta):
    return theta % np.pi

def angle_to_bin(theta, num_bins):
    return int(theta / np.pi * num_bins)

def circular_mean(angles):

    # 🔥 방향성 없는 각도 처리 (π periodic)
    angles2 = 2 * angles

    x = np.cos(angles2)
    y = np.sin(angles2)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    mean_angle = 0.5 * np.arctan2(mean_y, mean_x)

    return np.mod(mean_angle, np.pi)

def von_mises_kernel(theta, mu, kappa):
    return np.exp(kappa * np.cos(theta - mu))

def von_mises_pdf(theta, mu, kappa):
    return np.exp(kappa * np.cos(theta - mu))

def log_von_mises_pdf(theta, mu, kappa):
    return kappa * np.cos(theta - mu)

def build_von_mises_distribution(samples, num_bins=180, base_kappa=2.0):
    dist = OrientationDistribution(num_bins)
    thetas = np.linspace(0, np.pi, num_bins, endpoint=False)

    for sample in samples:
        dist.hist += sample.weight * np.exp(
            np.clip(base_kappa * np.cos(thetas - sample.theta), -50, 50)
        )

    dist.normalize()
    return dist

def fit_von_mises_mixture(samples, K=5, max_iter=20,
                         min_weight=0.05,
                         merge_thresh=np.deg2rad(10)):
    """
    samples: (theta, weight)
    """

    thetas = np.array([s.theta for s in samples])
    weights = np.array([s.weight for s in samples])

    N = len(samples)

    # 🔥 초기화 (균등하게)
    mus = np.linspace(0, np.pi, K, endpoint=False)
    kappas = np.ones(K) * 4.0
    pis = np.ones(K) / K

    for _ in range(max_iter):

        # =========================
        # E-step
        # =========================
        log_resp = np.zeros((N, K))

        for k in range(K):
            log_resp[:, k] = np.log(pis[k] + 1e-12) + log_von_mises_pdf(thetas, mus[k], kappas[k])

        # 🔥 log-sum-exp trick
        max_log = np.max(log_resp, axis=1, keepdims=True)
        log_resp = log_resp - max_log

        resp = np.exp(log_resp)
        resp_sum = resp.sum(axis=1, keepdims=True) + 1e-12
        resp /= resp_sum

        # =========================
        # M-step
        # =========================
        Nk = (resp * weights[:, None]).sum(axis=0)

        for k in range(K):
            if Nk[k] < 1e-6:
                continue

            # 방향 평균 (circular mean)
            sin_sum = np.sum(resp[:, k] * weights * np.sin(thetas))
            cos_sum = np.sum(resp[:, k] * weights * np.cos(thetas))

            mu = np.arctan2(sin_sum, cos_sum)
            if mu < 0:
                mu += np.pi

            mus[k] = mu

            R = np.sqrt(sin_sum**2 + cos_sum**2) / (Nk[k] + 1e-8)

            # kappa 근사
            kappas[k] = min(50.0, max(1e-3, 2 * R / (1 - R + 1e-8)))

            pis[k] = max(1e-6, Nk[k] / (weights.sum() + 1e-8))

    # =========================
    # 🔥 pruning
    # =========================

    components = []

    for k in range(K):
        if pis[k] < min_weight:
            continue

        components.append(VonMisesComponent(mus[k], kappas[k], pis[k]))

    # =========================
    # 🔥 merge 가까운 방향
    # =========================
    merged = []

    for comp in components:
        found = False
        for m in merged:
            d = np.abs(comp.mu - m.mu)
            d = min(d, np.pi - d)

            if d < merge_thresh:
                # merge
                m.mu = (m.mu + comp.mu) / 2
                m.weight += comp.weight
                found = True
                break

        if not found:
            merged.append(comp)

    return merged

def fuse_distributions(dist1, dist2):

    eps = 1e-6

    alpha = compute_confidence(dist1.hist)
    beta  = compute_confidence(dist2.hist)

    total = alpha + beta + eps
    alpha /= total
    beta  /= total

    # 1️⃣ log fusion
    log_p = alpha * dist1.log() + beta * dist2.log()

    fused = OrientationDistribution(dist1.num_bins)
    fused.hist = np.exp(log_p)

    # 2️⃣ normalize
    fused.normalize()

    confidence = compute_confidence(fused.hist)

    if confidence < 0.6:
        fused.hist = fused.hist ** 1.5
        fused.normalize()

    return fused, alpha, beta

def find_peaks(dist, threshold_ratio=0.2):
    hist = dist.hist
    peaks = []
    max_val = np.max(hist)

    for i in range(len(hist)):
        left = hist[i-1]
        center = hist[i]
        right = hist[(i+1) % len(hist)]

        if center > left and center > right and center > threshold_ratio * max_val:
            peaks.append(i)

    return peaks

def compute_confidence(dist):
    peak = np.max(dist.hist)
    total = np.sum(dist.hist)
    return peak / (total + 1e-8)

def compute_depth_confidence(components_depth):
    if len(components_depth) == 0:
        return 0

    weights = [c.weight for c in components_depth]
    return max(weights)

def compute_fused_confidence(fused_components):
    if len(fused_components) == 0:
        return 0

    scores = [c["score"] for c in fused_components]

    # 강한 방향이 몇 개냐 + 얼마나 강하냐
    return np.mean(scores)

def compute_corner_confidence(orientations):
    if len(orientations) < 2:
        return 0.0

    best_score = 0

    for i in range(len(orientations)):
        for j in range(i+1, len(orientations)):

            diff = angle_diff(orientations[i], orientations[j])

            # 90도에 가까울수록 점수 ↑
            score = np.exp(-((diff - np.pi/2)**2) / 0.1)

            best_score = max(best_score, score)

    return best_score

def merge_similar_components(components, thresh=np.deg2rad(10)):
    merged = []

    for comp in components:
        keep = True
        for m in merged:
            if angle_diff(comp["theta"], m["theta"]) < thresh:
                keep = False
                break
        if keep:
            merged.append(comp)

    return merged

# -----------------------------
# edge sample extraction
# -----------------------------
def extract_edge_samples(gray_img):
    """
    input:
        gray_img: (H, W)

    output:
        List[OrientationSample]
    """

    # 1️⃣ saliency (간단히 gradient magnitude 사용)
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    saliency = np.sqrt(gx**2 + gy**2)

    # normalize
    saliency = saliency / (saliency.max() + 1e-6)

    # 2️⃣ canny edge
    edges = cv2.Canny(gray_img, 50, 150)

    # 3️⃣ hough line (probabilistic)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=5
    )

    samples = []

    if lines is None:
        return samples

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1

        length = np.sqrt(dx*dx + dy*dy)
        if length < 15:
            continue

        # orientation
        theta = np.arctan2(dy, dx)
        theta = normalize_angle(theta)

        # -----------------------------
        # 🔥 saliency along line
        # -----------------------------
        num_points = int(length)
        xs = np.linspace(x1, x2, num_points).astype(np.int32)
        ys = np.linspace(y1, y2, num_points).astype(np.int32)

        xs = np.clip(xs, 0, gray_img.shape[1]-1)
        ys = np.clip(ys, 0, gray_img.shape[0]-1)

        sal_vals = saliency[ys, xs]
        sal_mean = np.mean(sal_vals)

        # -----------------------------
        # 🔥 weight 설계 (핵심)
        # -----------------------------
        # length + saliency 기반 soft weighting
        weight = np.log(1 + length)
        weight = min(weight, 5.0)

        samples.append(OrientationSample(theta, weight))

    return samples

def extract_depth_samples(depth_map):
    samples = []

    depth_blur = cv2.GaussianBlur(depth_map, (3,3), 0)

    gx = cv2.Sobel(depth_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_blur, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)

    theta_map = np.arctan2(gy, gx)
    theta_map = np.mod(theta_map, np.pi)

    h, w = depth_map.shape

    for y in range(0, h, 2):
        for x in range(0, w, 2):

            m = mag[y, x]
            if m < 0.01:
                continue

            theta = theta_map[y, x]
            weight = m * np.exp(-m / 0.1)

            samples.append(OrientationSample(theta, weight))

    return samples, theta_map

#시각화
def visualize_edge_samples(gray_img, samples, window_name="edge_samples"):
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    if len(samples) == 0:
        cv2.imshow(window_name, vis)
        return

    # weight 정규화 (색상용)
    weights = np.array([s.weight for s in samples])
    w_min, w_max = weights.min(), weights.max()

    for s in samples:
        theta = s.theta
        weight = s.weight

        # weight normalize → color intensity
        w_norm = (weight - w_min) / (w_max - w_min + 1e-6)

        # 중심 기준 선 그리기
        h, w = gray_img.shape
        cx, cy = w // 2, h // 2

        length = int(40 * w_norm + 10)

        dx = int(np.cos(theta) * length)
        dy = int(np.sin(theta) * length)

        x1, y1 = cx - dx, cy - dy
        x2, y2 = cx + dx, cy + dy

        color = (0, int(255 * w_norm), int(255 * (1 - w_norm)))  
        # 녹색(강함) ↔ 빨강(약함)

        cv2.line(vis, (x1, y1), (x2, y2), color, 1)
        vis_up = cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, vis_up)

def visualize_von_mises_mixture(components, 
                               num_bins=180,
                               width=400, height=200,
                               window_name="von_mises"):

    thetas = np.linspace(0, np.pi, num_bins)
    dist = np.zeros_like(thetas)

    for c in components:
        dist += c.weight * np.exp(
            np.clip(c.kappa * np.cos(thetas - c.mu), -50, 50)
        )

    dist /= (dist.max() + 1e-8)

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    for i in range(num_bins - 1):
        x1 = int(i / num_bins * width)
        x2 = int((i + 1) / num_bins * width)

        y1 = int(height - dist[i] * height)
        y2 = int(height - dist[i + 1] * height)

        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

    for c in components:
        x = int(c.mu / np.pi * width)
        cv2.line(canvas, (x, 0), (x, height), (0, 0, 255), 1)

    cv2.putText(canvas, window_name, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow(window_name, canvas)

def visualize_polar_distribution(dist, window_name="distribution",
                                 width=400, height=200):

    thetas = np.linspace(0, np.pi, len(dist.hist))

    values = dist.hist.copy()
    values /= (values.max() + 1e-8)

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    for i in range(len(values) - 1):
        x1 = int(i / len(values) * width)
        x2 = int((i + 1) / len(values) * width)

        y1 = int(height - values[i] * height)
        y2 = int(height - values[i + 1] * height)

        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

    cv2.imshow(window_name, canvas)

def visualize_depth_gradient(depth_map):

    # =========================
    # 1️⃣ gradient 계산 (동일하게)
    # =========================
    depth_blur = cv2.GaussianBlur(depth_map, (3,3), 0)

    gx = cv2.Sobel(depth_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_blur, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    theta = np.arctan2(gy, gx)
    theta = np.mod(theta, np.pi)

    # =========================
    # 2️⃣ normalize
    # =========================
    mag_norm = mag / (mag.max() + 1e-8)

    # =========================
    # 3️⃣ HSV 변환
    # =========================
    hsv = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

    # Hue: 방향 (0~180)
    hsv[..., 0] = (theta / np.pi * 180).astype(np.uint8)

    # Saturation: 고정
    hsv[..., 1] = 255

    # Value: magnitude
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)

    # =========================
    # 4️⃣ BGR 변환
    # =========================
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr_up = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

    # =========================
    # 5️⃣ 출력
    # =========================
    cv2.imshow("depth_gradient", bgr_up)

def visualize_fused_components(fused_components,
                              num_bins=180,
                              width=400, height=200,
                              window_name="fused_mixture"):

    thetas = np.linspace(0, np.pi, num_bins)
    dist = np.zeros_like(thetas)

    # 🔥 각 fused component를 gaussian처럼 뿌림
    for comp in fused_components:
        mu = comp["theta"]
        score = comp["score"]

        dist += score * np.exp(5 * np.cos(thetas - mu))  # kappa=5 고정

    # normalize
    if dist.max() > 0:
        dist /= dist.max()

    # canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # graph
    for i in range(num_bins - 1):
        x1 = int(i / num_bins * width)
        x2 = int((i + 1) / num_bins * width)

        y1 = int(height - dist[i] * height)
        y2 = int(height - dist[i + 1] * height)

        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

    # peak 표시
    for comp in fused_components:
        x = int(comp["theta"] / np.pi * width)
        cv2.line(canvas, (x, 0), (x, height), (0, 0, 255), 1)

    cv2.putText(canvas, window_name, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow(window_name, canvas)
    
# =========================
# 🔥 Manhattan estimation
# =========================

def find_peaks_simple(p, min_ratio=0.5):
    peaks = []
    max_val = np.max(p)

    for i in range(1, len(p)-1):
        if p[i] > p[i-1] and p[i] > p[i+1]:
            if p[i] > max_val * min_ratio:
                theta = (i / len(p)) * np.pi
                peaks.append((theta, p[i]))

    return peaks


def manhattan_score(p, theta, num_bins, window=2):

    idx = int(theta / np.pi * num_bins)
    idx_ortho = (idx + num_bins // 2) % num_bins

    score = 0.0

    # 🔥 주변까지 포함 (robust)
    for offset in range(-window, window+1):
        score += p[(idx + offset) % num_bins]
        score += p[(idx_ortho + offset) % num_bins]

    return score

#-------------------------------------------------------------------------------------
#segmentation
def segment_by_gradient(depth_map, best_theta):

    num_labels = 0
    labels = None

    h, w = depth_map.shape

    # 1️⃣ gradient
    depth_blur = cv2.GaussianBlur(depth_map, (3,3), 0)

    gx = cv2.Sobel(depth_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_blur, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    theta = np.arctan2(gy, gx)
    theta = np.mod(theta, np.pi)

    # 2️⃣ Manhattan 방향
    theta_main = best_theta
    theta_ortho = (best_theta + np.pi/2) % np.pi

    # 결과 맵
    seg = np.zeros((h, w), dtype=np.int32)

    # label
    LEFT, RIGHT, FLOOR, CEILING = 0, 1, 2, 3

    th = np.percentile(mag, 70)

    for y in range(h):
        for x in range(w):

            weight = mag[y, x] / (th + 1e-6)   # 🔥 soft weight

            if weight < 0.3:
                continue

            t = theta[y, x]

            # 🔥 gradient → surface 방향
            surface_dir = (t + np.pi/2) % np.pi

            # 3️⃣ 방향 차이
            d_main = min(abs(surface_dir - theta_main),
                         np.pi - abs(surface_dir - theta_main))

            d_ortho = min(abs(surface_dir - theta_ortho),
                          np.pi - abs(surface_dir - theta_ortho))

            # 4️⃣ 방향 기반 분류
            score_main = np.exp(-d_main**2 / 0.1)
            score_ortho = np.exp(-d_ortho**2 / 0.1)

            seg_score = np.zeros((h, w, 4), dtype=np.float32)

            # inside loop
            if score_main > score_ortho:
                if x < w // 2:
                    seg_score[y, x, LEFT] += weight * score_main
                else:
                    seg_score[y, x, RIGHT] += weight * score_main
            else:
                if y < h // 2:
                    seg_score[y, x, CEILING] += weight * score_ortho
                else:
                    seg_score[y, x, FLOOR] += weight * score_ortho

            seg = np.argmax(seg_score, axis=2)
            confidence = np.max(seg_score, axis=2)

            min_area = 50

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]

                if area < min_area:
                    mask = (labels == i)

                # 🔥 주변 label 찾기
                dilated = cv2.dilate(mask.astype(np.uint8), np.ones((3,3), np.uint8))
                neighbors = seg[dilated.astype(bool)]

                if len(neighbors) == 0:
                    continue

                # 🔥 가장 많이 등장한 label로 merge
                new_label = np.bincount(neighbors).argmax()

                seg[mask] = new_label

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg.astype(np.uint8))

                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < min_area:
                        seg[labels == i] = -1   # noise

            if num_labels <= 1:
                return np.zeros_like(depth_map, dtype=np.uint8)

    return seg

def smooth_segmentation(seg, ksize=5):

    h, w = seg.shape
    out = seg.copy()

    pad = ksize // 2

    for y in range(pad, h-pad):
        for x in range(pad, w-pad):

            patch = seg[y-pad:y+pad+1, x-pad:x+pad+1]

            values, counts = np.unique(patch, return_counts=True)
            out[y, x] = values[np.argmax(counts)]

    return out

def visualize_depth_segmentation(seg, smoothed):
    h, w = seg.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    labels = np.unique(seg)

    cluster_stats = []

    # 🔥 cluster 통계
    for i in labels:
        mask = (seg == i)

        if np.any(mask):
            vals = smoothed[mask]
            mean = vals.mean()
            var = vals.var()
            size = mask.sum()
        else:
            mean, var, size = np.inf, 0, 0

        cluster_stats.append({
            "label": i,
            "mean": mean,
            "var": var,
            "size": size
        })

    # 🔥 ✅ 핵심 변경: "가장 가까운 cluster = object"
    object_cluster = max(cluster_stats, key=lambda x: x["mean"])["label"]

    # 🔥 structure 분리
    structure_clusters = [c for c in cluster_stats if c["label"] != object_cluster]
    structure_clusters.sort(key=lambda x: x["mean"])

    # 🔥 label_map 생성
    label_map = {}
    label_map[object_cluster] = "OBJECT"

    names = ["NEAR", "MID", "FAR", "VERY_FAR", "EXTREME"]

    for i, c in enumerate(structure_clusters):
        label_map[c["label"]] = names[i] if i < len(names) else "FAR"

    # 🔥 색상
    colors = {
        "OBJECT": (0, 255, 255),
        "NEAR": (0, 0, 255),
        "MID": (0, 255, 0),
        "FAR": (255, 0, 0),
        "VERY_FAR": (255, 255, 0),
        "EXTREME": (255, 0, 255),
        "UNKNOWN": (128, 128, 128)
    }

    # 🔥 시각화
    for label in labels:
        semantic_label = label_map.get(label, "UNKNOWN")
        vis[seg == label] = colors.get(semantic_label, (128,128,128))

    vis_up = cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("depth segmentation", vis_up)

    return object_cluster

def segmentation_vs_global(seg, theta_map, fused_components):

    def angle_diff(a, b):
        d = abs(a - b)
        return min(d, np.pi - d)

    score = 0
    total = 0

    sigma = np.deg2rad(10)

    for label in np.unique(seg):
        mask = (seg == label)

        if np.sum(mask) < 50:
            continue

        orientations = theta_map[mask]

        pixel_scores = np.zeros_like(orientations)

        for comp in fused_components:
            diff = np.abs(orientations - comp["theta"])
            diff = np.minimum(diff, np.pi - diff)

            alignment = np.exp(-(diff**2) / (2 * sigma**2))

            pixel_scores = np.maximum(pixel_scores, alignment * comp["score"])

        region_score = np.mean(pixel_scores)

        score += region_score * np.sum(mask)
        total += np.sum(mask)

    return score / (total + 1e-6)

def segment_by_depth_adaptive(depth_map, k_min=3, k_max=5,
                              spatial_weight=0.2, grad_weight=0.3):

    h, w = depth_map.shape

    # normalize
    depth_norm = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    xs = xs.astype(np.float32) / w
    ys = ys.astype(np.float32) / h

    gx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)

    features = np.stack([
        depth_norm,
        xs * spatial_weight,
        ys * spatial_weight,
        grad_mag * grad_weight
    ], axis=-1)

    Z = features.reshape(-1, 4).astype(np.float32)

    best_score = -np.inf
    best_seg = None
    best_k = k_min

    for k in range(k_min, k_max + 1):

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _, labels, centers = cv2.kmeans(
            Z, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )

        seg = labels.reshape(h, w)

        # 🔥 평가 점수 계산]
        cluster_means = []
        cluster_vars = []

        for i in range(k):
            mask = (seg == i)
            if np.any(mask):
                vals = depth_map[mask]
                cluster_means.append(vals.mean())
                cluster_vars.append(vals.var())
            else:
                cluster_means.append(np.inf)
                cluster_vars.append(0)

        # 기본 score
        separation = np.var(cluster_means)
        compactness = np.mean(cluster_vars)

        # 🔥 object separation
        object_mean = min(cluster_means)
        others = [m for m in cluster_means if m != object_mean]

        if len(others) > 0:
            separation_obj = np.mean(others) - object_mean
        else:
            separation_obj = 0

        # 🔥 최종 score
        score = separation - compactness + 0.5 * separation_obj

        if score > best_score:
            best_score = score
            best_seg = seg
            best_k = k

    return best_seg, best_k

def visualize_structure_depth(structure_depth):
    # 0값(마스킹된 부분) 포함해서 normalize
    vis = cv2.normalize(structure_depth, None, 0, 255, cv2.NORM_MINMAX)
    vis = vis.astype(np.uint8)

    # 확대
    vis_up = cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("structure_depth", vis_up)

def visualize_structure_with_mask(smoothed, obj_mask):
    # depth normalize
    depth_vis = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)

    # 컬러로 변환
    vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

    # 🔥 object 영역 빨간색 표시
    vis[obj_mask == 1] = (0, 0, 255)

    vis_up = cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("structure + object mask", vis_up)
    
#--------------------------------------------------------------------------------
#classify structure
def classify_structure(fused_components, is_manhattan,
                       ratio_thresh=0.6,
                       min_score=0.05):

    if len(fused_components) == 0:
        return "UNKNOWN"

    strong = [c for c in fused_components if c["score"] > min_score]

    if len(strong) == 1:
        return "WALL"

    if len(strong) >= 2:

        s1 = strong[0]["score"]
        s2 = strong[1]["score"]

        if s2 / (s1 + 1e-6) > ratio_thresh:

            if is_manhattan:
                return "CORNER"
            else:
                return "MULTI_DIRECTION"

    return "WALL"

def classify_from_segmentation(seg, theta_map):
    dirs = []

    for label in np.unique(seg):
        mask = (seg == label)

        if np.sum(mask) < 50:
            continue

        mean_dir = circular_mean(theta_map[mask])
        dirs.append(mean_dir)

    # 방향 clustering
    if len(dirs) < 2:
        return "WALL"

    diffs = []
    for i in range(len(dirs)):
        for j in range(i+1, len(dirs)):
            d = abs(dirs[i] - dirs[j])
            d = min(d, np.pi - d)
            diffs.append(d)

    if any(abs(d - np.pi/2) < np.deg2rad(15) for d in diffs):
        return "CORNER"

    return "WALL"

def estimate_orientation(depth, mask):
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

    gx_vals = gx[mask == 1]
    gy_vals = gy[mask == 1]

    if len(gx_vals) < 10:
        return None

    angles = np.arctan2(gy_vals, gx_vals)

    # 대표 방향 (circular mean)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))

    theta = np.arctan2(sin_mean, cos_mean)

    return theta

def select_dominant_clusters(cluster_stats, top_k=2):
    # size 기준으로 상위 2개
    sorted_clusters = sorted(cluster_stats, key=lambda x: x["size"], reverse=True)
    return sorted_clusters[:top_k]

def compute_cluster_stats(seg, smoothed):
    h, w = seg.shape
    labels = np.unique(seg)

    cluster_stats = []

    for i in labels:
        mask = (seg == i)

        if np.any(mask):
            vals = smoothed[mask]
            mean = vals.mean()
            var = vals.var()
            size = mask.sum()
        else:
            mean, var, size = np.inf, 0, 0

        cluster_stats.append({
            "label": i,
            "mean": mean,
            "var": var,
            "size": size
        })

    return cluster_stats

#-------------------------------------------------------------------------------------
#2dpontcloud 추출
def extract_wall_lines_ransac(seg, smoothed, dominant_clusters):
    lines = []

    for c in dominant_clusters:
        label = c["label"]

        mask = (seg == label)

        if np.sum(mask) < 100:
            continue

        ys, xs = np.where(mask)
        pts = np.stack([xs, ys], axis=1)

        if len(pts) < 50:
            continue

        line = ransac_line_fit(pts)

        if line is not None:
            lines.append(line)

    return lines

def fit_line_ransac(points, n_iter=100, thresh=0.02):
    best_inliers = []
    best_model = None

    N = len(points)
    if N < 2:
        return None, None

    for _ in range(n_iter):
        i1, i2 = np.random.choice(N, 2, replace=False)
        p1, p2 = points[i1], points[i2]

        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]

        norm = np.sqrt(dx*dx + dz*dz) + 1e-6
        a = dz / norm
        b = -dx / norm
        c = -(a * p1[0] + b * p1[1])

        # distance
        dists = np.abs(a * points[:,0] + b * points[:,1] + c)

        inliers = points[dists < thresh]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (a, b, c)

    return best_model, best_inliers

def align_to_manhattan(lines, best_theta):
    aligned = []

    for (a, b, c) in lines:
        theta = np.arctan2(-a, b)

        d1 = angle_diff(theta, best_theta)
        d2 = angle_diff(theta, best_theta + np.pi/2)

        if d1 < d2:
            theta_new = best_theta
        else:
            theta_new = best_theta + np.pi/2

        # 기존 line에서 한 점 추출
        if abs(b) > 1e-6:
            x0 = 0
            y0 = -(c + a*x0)/b
        else:
            y0 = 0
            x0 = -(c + b*y0)/a

        a_new = np.sin(theta_new)
        b_new = -np.cos(theta_new)
        c_new = -(a_new * x0 + b_new * y0)

        aligned.append((a_new, b_new, c_new))

    return aligned

def compute_best_intersection(lines):
    best_pair = None
    best_score = 0

    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            l1 = lines[i]["line"]
            l2 = lines[j]["line"]

            a1, b1, _ = l1
            a2, b2, _ = l2

            angle = abs(angle_diff(
                np.arctan2(-a1, b1),
                np.arctan2(-a2, b2)
            ))

            score = np.exp(-((angle - np.pi/2)**2) / 0.1)

            if score > best_score:
                best_score = score
                best_pair = (l1, l2)

    if best_pair is None:
        return None

    return compute_intersection(best_pair)

def compute_intersection(line_pair):
    (a1, b1, c1), (a2, b2, c2) = line_pair

    A = np.array([
        [a1, b1],
        [a2, b2]
    ])

    B = np.array([
        -c1,
        -c2
    ])

    det = np.linalg.det(A)

    if abs(det) < 1e-6:
        # 평행 또는 거의 평행
        return None

    x, y = np.linalg.solve(A, B)
    return np.array([x, y])

def compute_segment_slope(mask, gx):

    values = gx[mask]

    if len(values) < 50:
        return None, None

    mean_slope = np.mean(values)
    var_slope = np.var(values)

    return mean_slope, var_slope

def build_wall_pointcloud_from_slope(x0, z0, slope, length=5):

    pts = []

    for t in np.linspace(-length, length, 200):
        X = x0 + t
        Z = z0 + slope * t

        pts.append([X, Z])

    return np.array(pts)

def build_corner_pointcloud_from_slope(x0, z0, slope, length=5, num_points=100):

    pts = []

    # 🔥 방향 벡터 1 (벽 방향)
    dir1 = np.array([1.0, slope])
    dir1 = dir1 / (np.linalg.norm(dir1) + 1e-6)

    # 🔥 방향 벡터 2 (직각 방향)
    dir2 = np.array([-slope, 1.0])
    dir2 = dir2 / (np.linalg.norm(dir2) + 1e-6)

    # 🔥 두 방향으로 코너 생성
    for t in np.linspace(0, length, num_points):
        p1 = np.array([x0, z0]) + dir1 * t
        p2 = np.array([x0, z0]) + dir2 * t

        pts.append(p1)
        pts.append(p2)

    return np.array(pts)

def build_pointcloud_from_lines(lines, num_points=200):
    pts = []

    for (a, b, c) in lines:
        for t in np.linspace(-100, 100, num_points):
            if abs(b) > 1e-6:
                x = t
                y = -(a*x + c)/b
            else:
                y = t
                x = -(b*y + c)/a

            pts.append([x, y])

    return np.array(pts)

def build_corner_pointcloud(lines, corner_pt, length=100):
    pts = []

    if corner_pt is None:
        return None

    cx, cy = corner_pt

    for (a, b, c) in lines[:2]:
        # 방향 벡터
        dir_vec = np.array([b, -a])
        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)

        for t in np.linspace(0, length, 100):
            p = np.array([cx, cy]) + dir_vec * t
            pts.append(p)

    return np.array(pts)

def build_corner_pointcloud_from_single_line(corner_pt, dir1, length=100):
    pts = []

    cx, cy = corner_pt

    dir2 = np.array([-dir1[1], dir1[0]])

    for t in np.linspace(0, length, 100):
        p1 = np.array([cx, cy]) + dir1 * t
        p2 = np.array([cx, cy]) + dir2 * t

        pts.append(p1)
        pts.append(p2)

    return np.array(pts)

def visualize_pointcloud_cv2(pc, size=600, title="PointCloud BEV", structure=None):
    if pc is None or len(pc) == 0:
        print("Empty pointcloud")
        return

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    xs = pc[:, 0]
    ys = pc[:, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    scale = max(x_range, y_range)

    for x, y in pc:
        px = int((x - x_min) / scale * (size - 1))
        py = int((y - y_min) / scale * (size - 1))
        py = size - 1 - py
        cv2.circle(canvas, (px, py), 1, (0, 255, 0), -1)

    # 🔴 카메라 위치 표시
    cv2.circle(canvas, (size // 2, size - 10), 5, (0, 0, 255), -1)

    # 🔥 structure 텍스트 표시 (왼쪽 상단)
    if structure is not None:
        text = f"Structure: {structure}"
        cv2.putText(
            canvas,
            text,
            (10, 30),                      # 위치 (왼쪽 상단)
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,                           # 글자 크기
            (255, 255, 255),               # 색 (흰색)
            2,                             # 두께
            cv2.LINE_AA
        )

    cv2.imshow(title, canvas)

def select_best_wall_line(lines):
    best_score = -1e9
    best_line = None

    for l in lines:
        a, b, c = l["line"]
        theta = np.arctan2(-a, b)

        # 🔥 scoring
        score_inliers = l["inliers"]
        score_depth = l["mean_y"]   # 아래쪽일수록 큼
        angle_error = abs(angle_diff(theta, l["theta"]))

        score = (
            1.0 * score_inliers +
            0.5 * score_depth -
            2.0 * angle_error
        )

        if score > best_score:
            best_score = score
            best_line = l["line"]

    return best_line

def extract_depth_cross_section(depth_map):

    h, w = depth_map.shape
    y = h // 2   # 가운데 행

    xs = np.arange(w)
    zs = depth_map[y, :]

    pts = np.stack([xs, zs], axis=1)

    return pts

def pixel_to_camera_coords(xs, zs, width, fov_deg=89):

    cx = width / 2

    # 🔥 focal length 계산
    fx = (width / 2) / np.tan(np.deg2rad(fov_deg / 2))

    Xs = (xs - cx) / fx * zs

    return Xs, zs

def convert_pc_to_metric(pc, width):

    xs = pc[:, 0]
    zs = pc[:, 1]

    Xs, Zs = pixel_to_camera_coords(xs, zs, width)

    return np.stack([Xs, Zs], axis=1)

def compute_metric_slope(mask, depth_map, width):

    ys, xs = np.where(mask)
    zs = depth_map[ys, xs]

    if len(zs) < 50:
        return None, None

    # 🔥 camera 좌표 변환
    Xs, Zs = pixel_to_camera_coords(xs, zs, width)

    # 🔥 선형 회귀 (Z = aX + b)
    A = np.vstack([Xs, np.ones_like(Xs)]).T
    a, b = np.linalg.lstsq(A, Zs, rcond=None)[0]

    # 🔥 분산 (평면성)
    Z_pred = a * Xs + b
    var = np.var(Zs - Z_pred)

    return a, var

# ------------------- Depth + Saliency 통합 스레드 (최신 frame만) -------------------
def depth_saliency_thread():
    prev_depth = None
    depth_alpha = 0.6
    smoothing_skip = 2
    frame_counter = 0

    edge_block = True
    depth_block = True
    segmentation_block = True
    pointcloud_block = True

    depth_vis = False
    edge_vis = False
    fused_vis = True
    segmentation_vis = True
    pointcloud_vis =True

    while True:
        if not raw_queue:
            time.sleep(0.001)
            continue

        # 항상 최신 frame만
        jpeg_bytes, depth_bytes = raw_queue.pop()
        raw_queue.clear()

        depth_array = np.frombuffer(depth_bytes, dtype='<f4').copy()
        if depth_array.size != EXPECTED_W * EXPECTED_H:
            continue
        depth_frame = depth_array.reshape((EXPECTED_H, EXPECTED_W))

        # Depth downsample + smoothing
        depth_small = cv2.resize(depth_frame, (64, 64), interpolation=cv2.INTER_AREA)
        frame_counter += 1
        if prev_depth is None:
            smoothed = depth_small.copy()
        else:
            if frame_counter % smoothing_skip == 0:
                smoothed = cv2.addWeighted(prev_depth.astype(np.float32),
                                           depth_alpha,
                                           depth_small.astype(np.float32),
                                           1 - depth_alpha, 0.0)
            else:
                smoothed = depth_small.copy()
        prev_depth = smoothed.copy()

        # jpeg decode
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        h, w = frame.shape

        scale = 128 / max(h, w)

        new_w = round(w * scale)
        new_h = round(h * scale)

        gray_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if edge_block:

            samples_edge = extract_edge_samples(gray_small)

            components_edge = fit_von_mises_mixture(
                samples_edge,
                K=6,  # 여유있게
                min_weight=0.05,
                merge_thresh=np.deg2rad(10)
            )
            peaks_edge = [c.mu for c in components_edge]
            confidences_edge = [c.weight * c.kappa for c in components_edge]

            # debug
            if edge_vis:
                visualize_edge_samples(gray_small, samples_edge)
                visualize_von_mises_mixture(components_edge, window_name="edge_mixture")

                print("edge peaks:", peaks_edge)
                print("edge confidence:", confidences_edge)

        if depth_block:

            samples_depth,theta_map = extract_depth_samples(smoothed)

            components_depth = fit_von_mises_mixture(
                samples_depth,
                K=7,
                min_weight=0.1,
                merge_thresh=np.deg2rad(20)
            )

            peaks_depth = [c.mu for c in components_depth]
            confidences_depth = [c.weight * c.kappa for c in components_depth]

            if depth_vis:
                visualize_depth_gradient(smoothed)
                visualize_von_mises_mixture(components_depth, window_name="depth_mixture")

                print("depth peaks:", peaks_depth)
                print("depth confidence:", confidences_depth)

        if edge_block and depth_block:

            fused_components = []

            depth_conf = compute_depth_confidence(components_depth)

            for e in components_edge:

                support = 0

                sigma = np.deg2rad(10)

                for d in components_depth:
                    diff = angle_diff(e.mu, d.mu)
                    alignment = np.exp(-(diff**2) / (2 * sigma**2))
                    support += d.weight * alignment

                # 🔥 depth가 확실할 때만 반영
                if depth_conf > 0.1:
                    score = e.weight * (1 + support)
                else:
                    score = e.weight   # edge만 사용

                fused_components.append({
                    "theta": e.mu,
                    "score": score
                })

            fused_components = merge_similar_components(fused_components)

            fused_components = [
                c for c in fused_components if c["score"] > 0.05
            ] 

            fused_components = sorted(
                fused_components,
                key=lambda x: x["score"],
                reverse=True
            )

            fused_components = fused_components[:3]   # 또는 2~3

            # =========================
            # 🔥 best 방향 선택
            # =========================
            best_theta = None
            best_score = -1

            for comp in fused_components:
                if comp["score"] > best_score:
                    best_score = comp["score"]
                    best_theta = comp["theta"]

            # =========================
            # 🔥 Manhattan 판단
            # =========================
            is_manhattan = False

            def is_orthogonal(a, b, tol=np.deg2rad(15)):
                diff = abs(a - b)
                diff = min(diff, np.pi - diff)
                return abs(diff - np.pi/2) < tol

            found = False

            for i in range(len(fused_components)):
                for j in range(i+1, len(fused_components)):

                    if is_orthogonal(fused_components[i]["theta"],
                                     fused_components[j]["theta"]):
                        is_manhattan = True
                        found = True
                        break

                if found:
                    break

            # =========================
            # 🔥 debug
            # =========================
            if fused_vis:
                print("edge peaks:", [np.degrees(c.mu) for c in components_edge])
                print("depth peaks:", [np.degrees(c.mu) for c in components_depth])

                print("fused peaks:", [np.degrees(c["theta"]) for c in fused_components])
                print("best theta:", np.degrees(best_theta) if best_theta else None)
                print("num fused:", len(fused_components))
                print("is manhattan:", is_manhattan)
                
                visualize_edge_samples(gray_small, samples_edge)
                visualize_fused_components(fused_components, window_name="fused_mixture")
                #visualize_von_mises_mixture(components_edge, window_name="edge_mixture")
                #visualize_von_mises_mixture(components_depth, window_name="depth_mixture")
            
        if segmentation_block and edge_block and depth_block:

            # 🔥 adaptive segmentation (k 자동 결정)
            seg_depth, k = segment_by_depth_adaptive(
                smoothed,
                k_min=3,
                k_max=5,
                spatial_weight=0.2,
                grad_weight=0.3
            )

            # 🔥 object cluster 추출 + 시각화
            if segmentation_vis:
                obj_cluster = visualize_depth_segmentation(seg_depth, smoothed)
            else:
                # 시각화 안 해도 object는 필요함
                obj_cluster = extract_object_cluster(seg_depth, smoothed)

            # 🔥 object mask 생성
            obj_mask = (seg_depth == obj_cluster).astype(np.uint8)

            # 🔥 구조 영역만 남기기
            structure_depth = smoothed * (1 - obj_mask)

            labels = np.unique(seg_depth)
            structure_labels = [l for l in labels if l != obj_cluster]

            cluster_stats = compute_cluster_stats(seg_depth, smoothed)

            dominant = select_dominant_clusters(cluster_stats, top_k=2)

            orientations = []

            for c in dominant:
                label = c["label"]
                mask = (seg_depth == label).astype(np.uint8)

                theta = estimate_orientation(structure_depth, mask)

                if theta is not None:
                    orientations.append(theta)

            corner_conf = compute_corner_confidence(orientations)

            visualize_structure_depth(structure_depth)
            visualize_structure_with_mask(smoothed, obj_mask)

            if corner_conf > 0.7:
                structure = "CORNER"

            elif is_manhattan and len(fused_components) >= 2:
                structure = "LIKELY_CORNER"

            elif len(fused_components) >= 1:
                structure = "WALL"

            else:
                structure = "UNKNOWN"


            # 🔥 ambiguity 보정
            if corner_conf < 0.3 and not is_manhattan:
                if len(fused_components) >= 2:
                    structure = "UNKNOWN"

            print("final decision:", structure)

        if pointcloud_block:
            
            depth_corrected = 1.0 / (smoothed + 1e-6)
            smoothed = depth_corrected

            cross_section = extract_depth_cross_section(smoothed)
            metric_smoothed = convert_pc_to_metric(cross_section, width=smoothed.shape[1])
            points = metric_smoothed

            if structure != "UNKNOWN":

                candidates = []

                for c in dominant:

                    label = c["label"]
                    mask = (seg_depth == label)

                    if np.sum(mask) < 100:
                        continue

                    slope, var = compute_metric_slope(mask, smoothed, smoothed.shape[1])
                    if slope is None:
                        continue

                    ys, xs = np.where(mask)
                    zs = smoothed[ys, xs]

                    # 🔥 metric 변환
                    Xs, Zs = pixel_to_camera_coords(xs, zs, smoothed.shape[1])

                    mean_z = np.median(Zs)

                    # 🔥 scoring (핵심)
                    score = (
                        -abs(slope) * 2.0      # 기울기 너무 크면 불안정
                        -var * 5.0             # 평면성
                        -mean_z * 1.0          # 가까운 벽
                    )

                    candidates.append({
                        "slope": slope,
                        "var": var,
                        "mean_z": mean_z,
                        "score": score,
                        "mask": mask
                    })

                if len(candidates) == 0:
                    pointcloud = None

                else:
                    best = max(candidates, key=lambda x: x["score"])

                    slope = best["slope"]

                    # 🔥 대표 점 (anchor)
                    ys, xs = np.where(best["mask"])
                    zs = smoothed[ys, xs]

                    Xs, Zs = pixel_to_camera_coords(xs, zs, smoothed.shape[1])

                    x0 = np.mean(Xs) 
                    z0 = np.mean(Zs)

                    if structure == "WALL":

                        model, inliers = fit_line_ransac(points)

                        if model is not None and len(inliers) > 10:
                            x0 = np.mean(inliers[:,0])
                            z0 = np.mean(inliers[:,1])

                            dx = -model[1]
                            dz = model[0]

                            slope = dz / (dx + 1e-6)

                            pointcloud = build_wall_pointcloud_from_slope(x0, z0, slope)
                        else:
                            pointcloud = None

                    elif structure in ["CORNER", "LIKELY_CORNER"]:

                        model1, inliers1 = fit_line_ransac(points)

                        if model1 is None:
                            pointcloud = None

                        else:
                            # 🔥 첫 번째 라인 제거
                            mask = np.ones(len(points), dtype=bool)
                            for p in inliers1:
                                mask &= ~np.all(points == p, axis=1)

                            remaining = points[mask]

                            if len(remaining) < 10:
                                pointcloud = None

                            else:
                                model2, inliers2 = fit_line_ransac(remaining)

                                if model2 is None:
                                    pointcloud = None
                                else:
                                    # 🔥 직교 조건 확인
                                    n1 = np.array([model1[0], model1[1]])
                                    n2 = np.array([model2[0], model2[1]])

                                    cos_angle = np.abs(np.dot(n1, n2))

                                    if cos_angle < 0.3:  # ≈ 90도
                                        # 🔥 교점 계산
                                        A = np.array([[model1[0], model1[1]],
                                                      [model2[0], model2[1]]])
                                        B = np.array([-model1[2], -model2[2]])

                                        try:
                                            intersection = np.linalg.solve(A, B)
                                            x0, z0 = intersection

                                            # 방향은 첫 번째 라인 기준
                                            dx = -model1[1]
                                            dz = model1[0]
                                            slope = dz / (dx + 1e-6)

                                            pointcloud = build_corner_pointcloud_from_slope(x0, z0, slope)

                                        except:
                                            pointcloud = None
                                    else:
                                        pointcloud = None

                    else:
                        pointcloud = None

                if pointcloud_vis and pointcloud is not None:
                    metric_pc = pointcloud
                    visualize_pointcloud_cv2(metric_pc, title="Metric PointCloud", structure=structure)
                    visualize_pointcloud_cv2(metric_smoothed, title="Depth Cross Section", structure=structure)
            
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
        


# ------------------- Main -------------------
def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST,PORT))
        s.listen(1)
        print(f"Listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        print(f"Client connected: {addr}")

        # PC에서 보여줄 화면 크기
        target_H, target_W = 480, 640

        threading.Thread(target=receive_thread, args=(conn, raw_queue, gyro_queue, accel_queue), daemon=True).start()
        threading.Thread(target=depth_saliency_thread,daemon=True).start()

        # keep main alive
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down...")

if __name__=="__main__":
    main()

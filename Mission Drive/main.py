#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import serial
import time
import math
import threading
from ultralytics import YOLO
from rplidar import RPLidar
import torch

# =================================================================================
# [설정] 포트 및 파라미터
# =================================================================================
SERIAL_PORT = 'COM5'
LIDAR_PORT_NAME = 'COM7'
BAUD_RATE = 115200

# 카메라 인덱스 (USB 2대)
CAM_LANE_INDEX = 1
CAM_TRAFFIC_INDEX = 2

# 모델 경로
LANE_MODEL_PATH = 'mission2/weights/best.pt'
TRAFFIC_MODEL_PATH = 'train2/weights/best.pt'

WIDTH, HEIGHT = 640, 480
OFFSET = 300
PROCESS_WIDTH = WIDTH + (OFFSET * 2)

# ===================== [라이다 감지 파라미터] =====================***************************************
SAFE_DISTANCE = 1100      # 전방 1100mm(1.1m)
SCAN_ANGLE = 30           # deg (전방 +-30도)

# 회피 횟수 *******************************************************************
MAX_OBSTACLES = 1

# ===================== [회피/직진 타이밍] ===================== *************************************
AVOID_TIME = 3.0          # 회피 동작 유지 시간(초) (기존 3.0 유지)
STRAIGHT_TIME = 0.0       # ✅ 회피 1회 후 직진 유지 시간(초) <-- 이 값만 조정하면 됨

# ===================== [최신 차선 제어 파라미터] =====================
BASE_SPEED = -70
STEER_GAIN = -0.7
ALPHA = 0.05
DEADZONE = 0.02

# 슬라이딩 윈도우 (최신 값)
N_WINDOWS = 10
WINDOW_MARGIN = 50
MIN_PIX = 15
LANE_WIDTH = 340

# ===================== [신호등 판정] =====================
STOP_FLAG = 0
TRAFFIC_PIXEL_THRESH = 500

# ===================== [공유 변수] =====================
current_lane = 2               # 2차선 시작
obstacle_count = 0
avoid_end_time = 0.0
is_obstacle_detected = False
lidar_running = True

# ✅ 회피 후 직진 모드 + 종료 시간
force_straight_mode = False
straight_end_time = 0.0

# ===================== [BEV 좌표] =====================
TX, TY = 185, 268
BX, BY = 266, 392
SRC_POINTS = np.float32([
    [320 - TX, TY], [320 + TX, TY],
    [320 - BX, BY], [320 + BX, BY]
])
DST_POINTS = np.float32([
    [150 + OFFSET, 0], [490 + OFFSET, 0],
    [150 + OFFSET, HEIGHT], [490 + OFFSET, HEIGHT]
])


# =================================================================================
# [라이다 전용 쓰레드]*********************************** 여기서 이제 라이다가 계속 보면서 뒷감지 하고있음
# =================================================================================
def lidar_thread_func():
    global is_obstacle_detected, lidar_running, obstacle_count

    lidar = None
    try:
        lidar = RPLidar(LIDAR_PORT_NAME, timeout=3)
        print("✅ LiDAR Thread Started")
    except:
        print("❌ LiDAR Connection Failed! Check COM Port.")
        return

    try:
        lidar.start_motor()
        lidar.clean_input()
    except:
        pass

    while lidar_running:
        # ✅ 회피 1회 완료하면 라이다 감지 무력화 (이후 트리거 절대 안 걸림)
        if obstacle_count >= MAX_OBSTACLES:
            time.sleep(0.2)
            continue

        try:
            for scan in lidar.iter_scans():
                if not lidar_running:
                    break

                min_dist_in_scan = 9999
                detected_in_scan = False

                for point in scan:
                    if len(point) == 3:
                        quality, angle, distance = point
                    elif len(point) == 4:
                        _, quality, angle, distance = point
                    else:
                        continue

                    if distance == 0 or distance > 2000:
                        continue

                    # 전방만 체크********************
                    if (angle < SCAN_ANGLE) or (angle > (360 - SCAN_ANGLE)):
                        if distance < min_dist_in_scan:
                            min_dist_in_scan = distance
                        if distance < SAFE_DISTANCE:
                            detected_in_scan = True

                is_obstacle_detected = detected_in_scan

                if min_dist_in_scan < 1500:
                    print(f"[LiDAR] Dist: {int(min_dist_in_scan)}mm | Detect: {detected_in_scan}")

        except:
            try:
                if lidar:
                    lidar.clean_input()
            except:
                pass

    if lidar:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
    print("🛑 LiDAR Thread Stopped")


# =================================================================================
# [차선/신호등 처리 함수들]
# =================================================================================
def warp_image(img):
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    return cv2.warpPerspective(img, M, (PROCESS_WIDTH, HEIGHT))


def send_command(ser, speed, steer):
    if ser and ser.is_open:
        speed = int(max(min(speed, 255), -255))
        steer = int(max(min(steer, 100), -100))
        ser.write(f"S {speed} {steer}\n".encode())


def get_lane_data(frame, model):
    results = model.predict(frame, device=0, conf=0.4, verbose=False)[0]
    annotated = results.plot()
    left_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    right_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    end_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    if results.masks is not None:
        for mask, cls in zip(results.masks.data, results.boxes.cls):
            m = cv2.resize(mask.cpu().numpy(), (WIDTH, HEIGHT))
            m = (m > 0.5).astype(np.uint8) * 255
            c = int(cls)
            if c == 1:
                left_mask = cv2.bitwise_or(left_mask, m)
            elif c == 0:
                right_mask = cv2.bitwise_or(right_mask, m)
            elif c == 2:
                end_mask = cv2.bitwise_or(end_mask, m)

    return left_mask, right_mask, end_mask, annotated


def get_sliding_window_center(warped_left, warped_right):
    """
    최신 코드 방식: 좌/우 마스크 독립 추적 + 한쪽 없으면 차선폭으로 보정
    """
    combined_bin = cv2.bitwise_or(warped_left, warped_right)
    out_img = np.dstack((combined_bin, combined_bin, combined_bin))

    left_hist = np.sum(warped_left[HEIGHT // 2:, :], axis=0)
    right_hist = np.sum(warped_right[HEIGHT // 2:, :], axis=0)

    leftx_current = np.argmax(left_hist) if np.max(left_hist) > MIN_PIX else (PROCESS_WIDTH // 2 - LANE_WIDTH // 2)
    rightx_current = np.argmax(right_hist) if np.max(right_hist) > MIN_PIX else (PROCESS_WIDTH // 2 + LANE_WIDTH // 2)

    window_height = int(HEIGHT // N_WINDOWS)
    nz_left = warped_left.nonzero()
    nz_right = warped_right.nonzero()

    total_left_x, total_right_x = [], []

    for window in range(N_WINDOWS):
        win_y_low = HEIGHT - (window + 1) * window_height
        win_y_high = HEIGHT - window * window_height

        # left window
        win_xleft_low = leftx_current - WINDOW_MARGIN
        win_xleft_high = leftx_current + WINDOW_MARGIN
        good_left = ((nz_left[0] >= win_y_low) & (nz_left[0] < win_y_high) &
                     (nz_left[1] >= win_xleft_low) & (nz_left[1] < win_xleft_high)).nonzero()[0]
        if len(good_left) > MIN_PIX:
            leftx_current = int(np.mean(nz_left[1][good_left]))

        # right window
        win_xright_low = rightx_current - WINDOW_MARGIN
        win_xright_high = rightx_current + WINDOW_MARGIN
        good_right = ((nz_right[0] >= win_y_low) & (nz_right[0] < win_y_high) &
                      (nz_right[1] >= win_xright_low) & (nz_right[1] < win_xright_high)).nonzero()[0]
        if len(good_right) > MIN_PIX:
            rightx_current = int(np.mean(nz_right[1][good_right]))

        # mutual correction
        if len(good_left) > MIN_PIX and len(good_right) <= MIN_PIX:
            rightx_current = leftx_current + LANE_WIDTH
        elif len(good_left) <= MIN_PIX and len(good_right) > MIN_PIX:
            leftx_current = rightx_current - LANE_WIDTH

        # visualize
        cv2.rectangle(out_img, (leftx_current - WINDOW_MARGIN, win_y_low),
                      (leftx_current + WINDOW_MARGIN, win_y_high), (0, 255, 0), 1)
        cv2.rectangle(out_img, (rightx_current - WINDOW_MARGIN, win_y_low),
                      (rightx_current + WINDOW_MARGIN, win_y_high), (0, 255, 0), 1)

        total_left_x.append(leftx_current)
        total_right_x.append(rightx_current)

    lane_center = (np.mean(total_left_x) + np.mean(total_right_x)) / 2.0
    return lane_center, out_img


def get_traffic_masks(frame, traffic_model, device):
    """
    기존 신호등 방식 유지 (red=0, green=1 마스크 분리)
    """
    results = traffic_model.predict(frame, device=device, conf=0.4, verbose=False)[0]
    red_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    green_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    if results.masks is not None:
        for mask, cls in zip(results.masks.data, results.boxes.cls):
            m = mask.cpu().numpy()
            m = cv2.resize(m, (WIDTH, HEIGHT))
            m = (m > 0.5).astype(np.uint8) * 255
            c = int(cls)
            if c == 0:
                red_mask = cv2.bitwise_or(red_mask, m)
            elif c == 1:
                green_mask = cv2.bitwise_or(green_mask, m)

    annotated = results.plot()
    return red_mask, green_mask, annotated


# =================================================================================
# [메인]
# =================================================================================
def main():
    global STOP_FLAG, current_lane, obstacle_count, avoid_end_time
    global lidar_running, is_obstacle_detected, force_straight_mode, straight_end_time

    # 디바이스 자동 선택
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🧠 Torch CUDA available = {torch.cuda.is_available()} | device = {device}")

    print("⏳ Loading Models...")
    lane_model = YOLO(LANE_MODEL_PATH)
    traffic_model = YOLO(TRAFFIC_MODEL_PATH)

    # 시리얼
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        if ser:
            ser.write(b'F\n')
            print("✅ Arduino Connected")
    except:
        ser = None
        print("⚠️ Arduino Not Connected!")

    # 라이다 스레드 시작
    t = threading.Thread(target=lidar_thread_func, daemon=True)
    t.start()

    # 카메라 연결
    cap1 = cv2.VideoCapture(CAM_LANE_INDEX, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(CAM_TRAFFIC_INDEX, cv2.CAP_DSHOW)

    # 카메라 해상도 고정
    for cap in (cap1, cap2):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

    smoothed_steer = 0.0
    current_error = 0.0

    print("🚀 System Start")

    FPS_LIMIT = 30
    FRAME_TIME = 1.0 / FPS_LIMIT

    while True:
        loop_start = time.time()

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("❌ Camera read failed")
            break

        frame1 = cv2.resize(frame1, (WIDTH, HEIGHT))
        frame2 = cv2.resize(frame2, (WIDTH, HEIGHT))

        now = time.time()

        # ---------------------------------------------------------
        # [A] 회피 기동 중: 차선/신호등 완전 무시
        # ---------------------------------------------------------
        if now < avoid_end_time:
            remaining = avoid_end_time - now
            cv2.putText(frame1, f"AVOIDING... {remaining:.1f}s", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Lane Cam", frame1)
            cv2.imshow("Traffic Cam", frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ---------------------------------------------------------
        # [B] 회피 1회 끝난 뒤: 직진 하드코딩 모드 (시간 조절 가능)
        #     - straight_end_time까지 조향 0 직진
        #     - 끝나면 즉시 차선+신호등 모드로 복귀
        # ---------------------------------------------------------
        if force_straight_mode:
            if now < straight_end_time:
                remaining = straight_end_time - now
                send_command(ser, BASE_SPEED, 0)

                cv2.putText(frame1, f"FORCE STRAIGHT... {remaining:.1f}s", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 50), 2)

                cv2.imshow("Lane Cam", frame1)
                cv2.imshow("Traffic Cam", frame2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            else:
                # 직진 종료 → 정상주행 복귀
                force_straight_mode = False

        # ---------------------------------------------------------
        # [C] 라이다 장애물 트리거 (✅ 1번만)
        #     - 회피 1회 후엔 obstacle_count == MAX_OBSTACLES가 되어
        #       라이다 스레드 자체가 감지 무력화됨
        # ---------------------------------------------------------
        if obstacle_count < MAX_OBSTACLES and is_obstacle_detected:
            print("⚠️ TRIGGERED! Obstacle Found!")

            # 2차선이면 L로 1차선 이동, 1차선이면 R로 2차선 이동
            if current_lane == 2:
                cmd, current_lane = 'L', 1
            else:
                cmd, current_lane = 'R', 2

            if ser:
                ser.write(f"{cmd}\n".encode())
                print(f"📡 Serial Sent: {cmd}")

            # 회피 동작 시간
            avoid_end_time = now + AVOID_TIME

            obstacle_count += 1
            is_obstacle_detected = False

            # ✅ 회피가 끝난 뒤 STRAIGHT_TIME 동안 직진
            force_straight_mode = True
            straight_end_time = avoid_end_time + STRAIGHT_TIME

            continue

        # ---------------------------------------------------------
        # [D] 정상 차선 + 신호등 주행
        # ---------------------------------------------------------
        l_mask, r_mask, e_mask, lane_img = get_lane_data(frame1, lane_model)

        # BEV
        w_left = warp_image(l_mask)
        w_right = warp_image(r_mask)

        # 슬라이딩 윈도우 중심
        lane_center, win_img = get_sliding_window_center(w_left, w_right)

        # 최신 조향 계산
        current_error = lane_center - (PROCESS_WIDTH / 2)
        raw_steer_angle = math.degrees(math.atan2(current_error, HEIGHT / 3))

        if abs(raw_steer_angle) < DEADZONE:
            raw_steer_angle = 0.0

        smoothed_steer = (ALPHA * raw_steer_angle) + ((1.0 - ALPHA) * smoothed_steer)
        steer_cmd = int(smoothed_steer * STEER_GAIN)

        # 신호등
        red_m, green_m, traffic_annot = get_traffic_masks(frame2, traffic_model, device=device)

        end_pixel_count = np.sum(e_mask > 0)
        red_pixel_count = np.sum(red_m > 0)
        green_pixel_count = np.sum(green_m > 0)

        if STOP_FLAG == 0:
            target_speed = BASE_SPEED
        elif STOP_FLAG == 1:
            target_speed = 0
            print("STOP")
            if green_pixel_count > TRAFFIC_PIXEL_THRESH:
                STOP_FLAG = 0
                print("START!!")

        if end_pixel_count > TRAFFIC_PIXEL_THRESH:
            if red_pixel_count > TRAFFIC_PIXEL_THRESH:
                STOP_FLAG = 1

        send_command(ser, target_speed, steer_cmd)

        # 시각화
        cv2.imshow("Lane Cam", lane_img)
        cv2.imshow("Warped Windows", win_img)
        cv2.imshow("Traffic Cam", traffic_annot)

        # FPS sync
        process_time = time.time() - loop_start
        wait_ms = int((FRAME_TIME - process_time) * 1000)
        if wait_ms < 1:
            wait_ms = 1

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    # 종료 처리
    lidar_running = False
    try:
        if ser:
            send_command(ser, 0, 0)
            ser.close()
    except:
        pass

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

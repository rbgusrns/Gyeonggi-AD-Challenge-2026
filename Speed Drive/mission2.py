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

# =================================================================================
# [설정] 포트 및 파라미터
# =================================================================================
SERIAL_PORT = 'COM16'  # 아두이노 포트 (장치 관리자 확인 필수!)
LIDAR_PORT_NAME = 'COM15'  # 라이다 포트 (장치 관리자 확인 필수!)
BAUD_RATE = 115200

LANE_MODEL_PATH = 'runs/segment/mission2/weights/best.pt'
TRAFFIC_MODEL_PATH = 'runs/segment/train2/weights/best.pt'

WIDTH, HEIGHT = 640, 480
OFFSET = 300
PROCESS_WIDTH = WIDTH + (OFFSET * 2)

# 라이다 감지 파라미터
SAFE_DISTANCE = 1100  # 1.1m 이내 감지
SCAN_ANGLE = 30  # 전방 +- 60도 감시
MAX_OBSTACLES = 2  # 최대 2번 회피

# 차선 주행 파라미터
DX_GAIN, DEGREE_GAIN, MAX_DEGREE = 0.5, 1.2, 50
START_BOX, TARGET_BOX = 0, 8
BASE_SPEED = -50  # 평상시 속도
ALPHA, DEADZONE = 0.2, 0.02
STOP_FLAG = 0

# 슬라이딩 윈도우 파라미터
N_WINDOWS = 10
WINDOW_MARGIN = 70
MIN_PIX = 15
LANE_WIDTH = 270

# ===================== [공유 변수] =====================
current_lane = 2
obstacle_count = 0
avoid_end_time = 0.0
is_obstacle_detected = False
lidar_running = True

# 좌표 변환 점
TX, TY = 185, 268
BX, BY = 266, 392
SRC_POINTS = np.float32([[320 - TX, TY], [320 + TX, TY], [320 - BX, BY], [320 + BX, BY]])
DST_POINTS = np.float32([[150 + OFFSET, 0], [490 + OFFSET, 0], [150 + OFFSET, HEIGHT], [490 + OFFSET, HEIGHT]])


# ===================== [라이다 전용 쓰레드 (에러 완벽 수정)] =====================
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
        if obstacle_count >= MAX_OBSTACLES:
            time.sleep(1)
            continue

        try:
            # 에러 없이 무한루프
            for scan in lidar.iter_scans():
                if not lidar_running: break

                min_dist_in_scan = 9999
                detected_in_scan = False

                # ★★★ [수정] 데이터 개수(3개/4개) 상관없이 처리 ★★★
                for point in scan:
                    if len(point) == 3:
                        quality, angle, distance = point
                    elif len(point) == 4:
                        _, quality, angle, distance = point
                    else:
                        continue

                    if distance == 0 or distance > 3000: continue

                    if (angle < SCAN_ANGLE) or (angle > (360 - SCAN_ANGLE)):
                        if distance < min_dist_in_scan:
                            min_dist_in_scan = distance

                        if distance < SAFE_DISTANCE:
                            detected_in_scan = True

                is_obstacle_detected = detected_in_scan

                # [디버깅] 거리 출력 (1.5m 이내일 때만 로그 뜸)
                if min_dist_in_scan < 1500:
                    print(f"[LiDAR] Dist: {int(min_dist_in_scan)}mm | Detect: {detected_in_scan}")

        except Exception as e:
            # print(f"⚠️ LiDAR Retry... {e}")
            try:
                if lidar: lidar.clean_input()
            except:
                pass

    if lidar:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
    print("🛑 LiDAR Thread Stopped")


# ===================== [영상 처리 함수들] =====================
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


def get_traffic_data(frame, model):
    results = model.predict(frame, device=0, conf=0.4, verbose=False)[0]
    annotated = results.plot()
    red_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    green_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    if results.masks is not None:
        for mask, cls in zip(results.masks.data, results.boxes.cls):
            m = cv2.resize(mask.cpu().numpy(), (WIDTH, HEIGHT))
            m = (m > 0.5).astype(np.uint8) * 255
            c = int(cls)
            if c == 0:
                red_mask = cv2.bitwise_or(red_mask, m)
            elif c == 1:
                green_mask = cv2.bitwise_or(green_mask, m)
    return red_mask, green_mask, annotated


def warp_image(img):
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    return cv2.warpPerspective(img, M, (PROCESS_WIDTH, HEIGHT))


def get_sliding_window_data(warped_left, warped_right):
    combined_bin = cv2.bitwise_or(warped_left, warped_right)
    out_img = np.dstack((combined_bin, combined_bin, combined_bin))
    left_hist = np.sum(warped_left[HEIGHT // 2:, :], axis=0)
    right_hist = np.sum(warped_right[HEIGHT // 2:, :], axis=0)

    leftx_current = np.argmax(left_hist) if np.max(left_hist) > MIN_PIX else (PROCESS_WIDTH // 2 - LANE_WIDTH // 2)
    rightx_current = np.argmax(right_hist) if np.max(right_hist) > MIN_PIX else (PROCESS_WIDTH // 2 + LANE_WIDTH // 2)

    window_height = int(HEIGHT // N_WINDOWS)
    nz_left = warped_left.nonzero()
    nz_right = warped_right.nonzero()

    lx, rx, ly, ry = [], [], [], []

    for window in range(N_WINDOWS):
        win_y_low = HEIGHT - (window + 1) * window_height
        win_y_high = HEIGHT - window * window_height

        good_left = ((nz_left[0] >= win_y_low) & (nz_left[0] < win_y_high) &
                     (nz_left[1] >= leftx_current - WINDOW_MARGIN) & (
                             nz_left[1] < leftx_current + WINDOW_MARGIN)).nonzero()[0]
        good_right = ((nz_right[0] >= win_y_low) & (nz_right[0] < win_y_high) &
                      (nz_right[1] >= rightx_current - WINDOW_MARGIN) & (
                              nz_right[1] < rightx_current + WINDOW_MARGIN)).nonzero()[0]

        if len(good_left) > MIN_PIX: leftx_current = int(np.mean(nz_left[1][good_left]))
        if len(good_right) > MIN_PIX: rightx_current = int(np.mean(nz_right[1][good_right]))

        if len(good_left) > MIN_PIX and len(good_right) <= MIN_PIX:
            rightx_current = leftx_current + LANE_WIDTH
        elif len(good_left) <= MIN_PIX and len(good_right) > MIN_PIX:
            leftx_current = rightx_current - LANE_WIDTH

        lx.append(leftx_current);
        rx.append(rightx_current)
        ly.append((win_y_low + win_y_high) / 2);
        ry.append((win_y_low + win_y_high) / 2)

        cv2.rectangle(out_img, (leftx_current - WINDOW_MARGIN, win_y_low), (leftx_current + WINDOW_MARGIN, win_y_high),
                      (200, 50, 255), 1)
        cv2.rectangle(out_img, (rightx_current - WINDOW_MARGIN, win_y_low),
                      (rightx_current + WINDOW_MARGIN, win_y_high), (200, 50, 255), 1)

    return lx, rx, ly, ry, out_img


def calculate_pd_control(lx, rx, ly, ry):
    dx = ((rx[TARGET_BOX] + lx[TARGET_BOX]) / 2) - ((rx[START_BOX] + lx[START_BOX]) / 2)
    dy = ((ly[START_BOX] + ry[START_BOX]) / 2) - ((ly[TARGET_BOX] + ry[TARGET_BOX]) / 2)
    degree = (math.atan2(dx, dy) * 180) / math.pi if dy != 0 else 0
    if abs(degree) > MAX_DEGREE: degree = MAX_DEGREE * (1 if degree > 0 else -1)

    error = (PROCESS_WIDTH / 2) - (rx[TARGET_BOX] + lx[TARGET_BOX] + rx[START_BOX] + lx[START_BOX]) / 4
    left_flag = 1 if dx < 0 else 0
    angle = abs(error * DX_GAIN + degree * DEGREE_GAIN)
    final_steer = -angle if left_flag else angle
    return final_steer


# ===================== [메인 루프] =====================
# ===================== [메인 루프 (명령어 전송 수정됨)] =====================
def main():
    global STOP_FLAG, current_lane, obstacle_count, avoid_end_time, lidar_running, is_obstacle_detected

    print("⏳ Loading Models...")
    lane_model = YOLO(LANE_MODEL_PATH).to('cuda:0')
    traffic_model = YOLO(TRAFFIC_MODEL_PATH).to('cuda:0')

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        if ser:
            ser.write(b'F')
            print("✅ Arduino Connected")
    except:
        ser = None
        print("⚠️ Arduino Not Connected!")

    # 쓰레드 시작
    t = threading.Thread(target=lidar_thread_func)
    t.daemon = True
    t.start()

    cap1 = cv2.VideoCapture("lane.mp4")
    cap2 = cv2.VideoCapture("sd3.mp4")
    smoothed_steer = 0.0

    print("🚀 System Start (Command Fix Applied)")

    # FPS 제한
    FPS_LIMIT = 30
    FRAME_TIME = 1.0 / FPS_LIMIT

    while True:
        loop_start = time.time()

        # [영상 읽기 및 무한 반복]
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = cap1.read()

        if not ret2:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = cap2.read()

        if not ret1 or not ret2: break

        frame1 = cv2.resize(frame1, (WIDTH, HEIGHT))
        frame2 = cv2.resize(frame2, (WIDTH, HEIGHT))

        now = time.time()

        # [1] 회피 기동 중 체크
        if now < avoid_end_time:
            remaining = int(avoid_end_time - now)
            cv2.putText(frame1, f"AVOIDING... {remaining}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Lane Mode (Cam1)", frame1)
            cv2.imshow("Traffic (Cam2)", frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

            # [2] 장애물 감지 체크
        if obstacle_count < MAX_OBSTACLES and is_obstacle_detected:
            print(f"⚠️ TRIGGERED! Obstacle Found!")
            if current_lane == 2:
                cmd, current_lane = 'L', 1
            else:
                cmd, current_lane = 'R', 2

            if ser:
                # ★★★ [핵심 수정] 명령어 뒤에 개행문자(\n) 추가 ★★★
                ser.write(f"{cmd}\n".encode())
                print(f"📡 Serial Sent: {cmd} (with newline)")

            avoid_end_time = time.time() + 11.0
            obstacle_count += 1
            is_obstacle_detected = False
            continue

            # [3] 일반 주행
        l_mask, r_mask, e_mask, lane_img = get_lane_data(frame1, lane_model)
        w_l, w_r = warp_image(l_mask), warp_image(r_mask)
        lx, rx, ly, ry, win_img = get_sliding_window_data(w_l, w_r)

        steer_angle = calculate_pd_control(lx, rx, ly, ry)
        smoothed_steer = (ALPHA * steer_angle) + ((1.0 - ALPHA) * smoothed_steer)

        red_m, green_m, traffic_img = get_traffic_data(frame2, traffic_model)

        end_pixel_count = np.sum(e_mask > 0)
        red_pixel_count = np.sum(red_m > 0)
        green_pixel_count = np.sum(green_m > 0)

        if STOP_FLAG == 0:
            target_speed = BASE_SPEED
        elif STOP_FLAG == 1:
            target_speed = 0
            print("STOP")
            if green_pixel_count > 500:
                STOP_FLAG = 0
                print("START!!")

        if end_pixel_count > 500:  # 500은 환경에 따라 조절!
            if red_pixel_count > 500:  # 500은 환경에 따라 조절!
                STOP_FLAG = 1

        print(f"S Speed:{target_speed}, Steer:{int(-smoothed_steer)}")

        if ser:
            ser.write(f"S {target_speed} {int(-smoothed_steer)}\n".encode())

        cv2.imshow("Lane Mode (Cam1)", lane_img)
        cv2.imshow("Traffic (Cam2)", traffic_img)
        cv2.imshow("Windows", win_img)

        # 속도 조절 (Sync)
        process_time = time.time() - loop_start
        wait_ms = int((FRAME_TIME - process_time) * 1000)
        if wait_ms < 1: wait_ms = 1

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'): break

    lidar_running = False
    if ser: ser.close()
    cap1.release();
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
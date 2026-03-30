#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import serial
import time
import math
from ultralytics import YOLO

# ===================== [1] 설정 및 파라미터 =====================
SERIAL_PORT = 'COM10'
BAUD_RATE = 115200
WIDTH, HEIGHT = 640, 480
OFFSET = 300
PROCESS_WIDTH = WIDTH + (OFFSET * 2)

# PD 제어 파라미터
DX_GAIN = 0.5  # 오차(Error)에 대한 가중치
DEGREE_GAIN = 1.2  # 기울기(Degree)에 대한 가중치
MAX_DEGREE = 50  # 최대 허용 기울기
START_BOX = 0  # 하단 윈도우 인덱스
TARGET_BOX = 8  # 상단 윈도우 인덱스

BASE_SPEED = -50  # 평상시 속도
ALPHA = 0.2  # 조향 부드러움 계수
DEADZONE = 0.02
STOP_FLAG = 0

# 슬라이딩 윈도우 파라미터
N_WINDOWS = 10
WINDOW_MARGIN = 70
MIN_PIX = 15
LANE_WIDTH = 270

# Bird's Eye View 좌표
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

# 모델 경로 (송주 환경에 맞게 수정해!)
LANE_MODEL_PATH = 'runs/segment/mission2/weights/best.pt'
TRAFFIC_MODEL_PATH = 'runs/segment/train2/weights/best.pt'

# 모델 로드
lane_model = YOLO(LANE_MODEL_PATH).to('cuda:0')
traffic_model = YOLO(TRAFFIC_MODEL_PATH).to('cuda:0')


# ===================== [2] 핵심 처리 함수 =====================

def get_lane_data(frame):
    """카메라 1: 차선 마스크 및 YOLO 시각화 영상 반환"""
    results = lane_model.predict(frame, device=0, conf=0.4, verbose=False)[0]
    annotated = results.plot()

    left_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    right_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    end_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    if results.masks is not None:
        for mask, cls in zip(results.masks.data, results.boxes.cls):
            m = cv2.resize(mask.cpu().numpy(), (WIDTH, HEIGHT))
            m = (m > 0.5).astype(np.uint8) * 255
            if int(cls) == 1:
                left_mask = cv2.bitwise_or(left_mask, m)
            elif int(cls) == 0:
                right_mask = cv2.bitwise_or(right_mask, m)
            elif int(cls) == 2:
                end_mask = cv2.bitwise_or(end_mask, m)

    return left_mask, right_mask, end_mask, annotated


def get_traffic_data(frame):
    """카메라 2: 신호등 마스크 및 YOLO 시각화 영상 반환 (0=RED, 1=GREEN)"""
    results = traffic_model.predict(frame, device=0, conf=0.4, verbose=False)[0]
    annotated = results.plot()

    red_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    green_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    if results.masks is not None:
        for mask, cls in zip(results.masks.data, results.boxes.cls):
            m = cv2.resize(mask.cpu().numpy(), (WIDTH, HEIGHT))
            m = (m > 0.5).astype(np.uint8) * 255
            if int(cls) == 0:
                red_mask = cv2.bitwise_or(red_mask, m)
            elif int(cls) == 1:
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

        good_left_inds = ((nz_left[0] >= win_y_low) & (nz_left[0] < win_y_high) &
                          (nz_left[1] >= leftx_current - WINDOW_MARGIN) &
                          (nz_left[1] < leftx_current + WINDOW_MARGIN)).nonzero()[0]
        if len(good_left_inds) > MIN_PIX:
            leftx_current = int(np.mean(nz_left[1][good_left_inds]))

        good_right_inds = ((nz_right[0] >= win_y_low) & (nz_right[0] < win_y_high) &
                           (nz_right[1] >= rightx_current - WINDOW_MARGIN) &
                           (nz_right[1] < rightx_current + WINDOW_MARGIN)).nonzero()[0]
        if len(good_right_inds) > MIN_PIX:
            rightx_current = int(np.mean(nz_right[1][good_right_inds]))

        if len(good_left_inds) > MIN_PIX and len(good_right_inds) <= MIN_PIX:
            rightx_current = leftx_current + LANE_WIDTH
        elif len(good_left_inds) <= MIN_PIX and len(good_right_inds) > MIN_PIX:
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
    if abs(degree) > MAX_DEGREE:
        degree = MAX_DEGREE * (1 if degree > 0 else -1)

    error = (PROCESS_WIDTH / 2) - (rx[TARGET_BOX] + lx[TARGET_BOX] + rx[START_BOX] + lx[START_BOX]) / 4

    left_flag = 1 if dx < 0 else 0
    angle = abs(error * DX_GAIN + degree * DEGREE_GAIN)
    final_steer = -angle if left_flag else angle

    return final_steer


def send_command(ser, speed, steer):
    if ser and ser.is_open:
        speed = int(max(min(speed, 255), -255))
        steer = int(max(min(steer, 100), -100))
        ser.write(f"S {speed} {steer}\n".encode())


# ===================== [3] 메인 루프 =====================

def main():

    global STOP_FLAG

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
    except:
        ser = None

    cap1 = cv2.VideoCapture("video/lane.mp4")  # 차선용
    cap2 = cv2.VideoCapture("video/sd3.mp4")  # 신호등용

    smoothed_steer = 0.0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2: break

        frame1 = cv2.resize(frame1, (WIDTH, HEIGHT))
        frame2 = cv2.resize(frame2, (WIDTH, HEIGHT))

        # 1. 차선 데이터 처리
        l_mask, r_mask, e_mask, lane_yolo_img = get_lane_data(frame1)
        w_left = warp_image(l_mask)
        w_right = warp_image(r_mask)
        lx, rx, ly, ry, visual_img = get_sliding_window_data(w_left, w_right)

        # 2. 신호등 데이터 처리
        red_m, green_m, traffic_yolo_img = get_traffic_data(frame2)

        end_pixel_count = np.sum(e_mask > 0)
        red_pixel_count = np.sum(red_m > 0)
        green_pixel_count = np.sum(green_m > 0)

        if STOP_FLAG == 0:
            current_speed = BASE_SPEED
        elif STOP_FLAG == 1:
            current_speed = 0
            print("STOP")
            if green_pixel_count > 500:
                STOP_FLAG = 0
                print("START!!")

        if end_pixel_count > 500:  # 500은 환경에 따라 조절!
            if red_pixel_count > 500:  # 500은 환경에 따라 조절!
                STOP_FLAG = 1



        # 3. PD 제어 및 조향 계산
        steer_angle = calculate_pd_control(lx, rx, ly, ry)
        smoothed_steer = (ALPHA * steer_angle) + ((1.0 - ALPHA) * smoothed_steer)

        # 명령 전송 (제어 방향에 따라 - 기호 확인)
        send_command(ser, current_speed, int(-smoothed_steer))

        # 4. 시각화 (YOLO 시각화 영상 + 슬라이딩 윈도우)
        cv2.imshow("Lane YOLO (Cam 1)", lane_yolo_img)
        cv2.imshow("Traffic YOLO (Cam 2)", traffic_yolo_img)
        cv2.imshow("Sliding Windows", visual_img)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if ser:
        send_command(ser, 0, 0)
        ser.close()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
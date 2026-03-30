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

# 제어 파라미터`
BASE_SPEED = -80
STEER_GAIN = -0.8
ALPHA = 0.05
DEADZONE = 0.02

# 윈도우 파라미터
N_WINDOWS = 10
WINDOW_MARGIN = 50
MIN_PIX = 15
LANE_WIDTH = 340
MIN_LANE_GAP = LANE_WIDTH * 0.7



# [1] 설정 및 파라미터 섹션 수정
TX, TY = 185, 268  # 트랙바에서 확정한 값 입력
BX, BY = 266, 392

SRC_POINTS = np.float32([
    [320 - TX, TY], [320 + TX, TY],  # Top Left, Top Right
    [320 - BX, BY], [320 + BX, BY]   # Bottom Left, Bottom Right
])


DST_POINTS = np.float32([
    [150 + OFFSET, 0], [490 + OFFSET, 0],
    [150 + OFFSET, HEIGHT], [490 + OFFSET, HEIGHT]
])

# YOLO 모델 로드 (클래스: 0=Right, 1=Left)
MODEL_PATH = 'runs/segment/mission1/weights/best.pt'
model = YOLO(MODEL_PATH).to('cuda:0')

# ===================== [2] 핵심 처리 함수 =====================

def get_yolo_masks(frame):
    """YOLO 추론 후 좌/우 마스크를 각각 분리하여 반환"""
    results = model.predict(frame, device=1, conf=0.4, verbose=False)[0]

    # 각 차선별 독립된 마스크 생성
    left_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    right_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    if results.masks is not None:
        for mask, cls in zip(results.masks.data, results.boxes.cls):
            m = mask.cpu().numpy()
            m = cv2.resize(m, (WIDTH, HEIGHT))
            m = (m > 0.5).astype(np.uint8) * 255

            if int(cls) == 1:  # Left Lane
                left_mask = cv2.bitwise_or(left_mask, m)
            elif int(cls) == 0:  # Right Lane
                right_mask = cv2.bitwise_or(right_mask, m)

    return left_mask, right_mask


def warp_image(img):
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    return cv2.warpPerspective(img, M, (PROCESS_WIDTH, HEIGHT))


def send_command(ser, speed, steer):
    if ser and ser.is_open:
        speed = int(max(min(speed, 255), -255))
        steer = int(max(min(steer, 100), -100))
        ser.write(f"S {speed} {steer}\n".encode())
        #print(f"S {speed} {steer}\n")

def get_sliding_window_center(warped_left, warped_right, prev_error):
    """분리된 마스크를 사용하여 좌우 윈도우를 독립적으로 추적"""
    # 두 마스크를 합쳐서 시각화용 베이스 생성
    combined_bin = cv2.bitwise_or(warped_left, warped_right)
    out_img = np.dstack((combined_bin, combined_bin, combined_bin))

    # 초기 위치 설정 (히스토그램 기반이 아닌 각 마스크의 하단부 중심점 활용)
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

        # 좌측 윈도우 추적
        win_xleft_low, win_xleft_high = leftx_current - WINDOW_MARGIN, leftx_current + WINDOW_MARGIN
        good_left_inds = ((nz_left[0] >= win_y_low) & (nz_left[0] < win_y_high) &
                          (nz_left[1] >= win_xleft_low) & (nz_left[1] < win_xleft_high)).nonzero()[0]

        if len(good_left_inds) > MIN_PIX:
            leftx_current = int(np.mean(nz_left[1][good_left_inds]))

        # 우측 윈도우 추적
        win_xright_low, win_xright_high = rightx_current - WINDOW_MARGIN, rightx_current + WINDOW_MARGIN
        good_right_inds = ((nz_right[0] >= win_y_low) & (nz_right[0] < win_y_high) &
                           (nz_right[1] >= win_xright_low) & (nz_right[1] < win_xright_high)).nonzero()[0]

        if len(good_right_inds) > MIN_PIX:
            rightx_current = int(np.mean(nz_right[1][good_right_inds]))

        # 상호 보정 (한쪽이 없을 때)
        if len(good_left_inds) > MIN_PIX and len(good_right_inds) <= MIN_PIX:
            rightx_current = leftx_current + LANE_WIDTH
        elif len(good_left_inds) <= MIN_PIX and len(good_right_inds) > MIN_PIX:
            leftx_current = rightx_current - LANE_WIDTH

        # 시각화
        cv2.rectangle(out_img, (leftx_current - WINDOW_MARGIN, win_y_low), (leftx_current + WINDOW_MARGIN, win_y_high),
                      (0, 255, 0), 1)
        cv2.rectangle(out_img, (rightx_current - WINDOW_MARGIN, win_y_low),
                      (rightx_current + WINDOW_MARGIN, win_y_high), (0, 255, 0), 1)

        total_left_x.append(leftx_current)
        total_right_x.append(rightx_current)

    current_lane_center = (np.mean(total_left_x) + np.mean(total_right_x)) / 2
    return current_lane_center, out_img


# ===================== [3] 메인 루프 =====================

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
    except:
        ser = None

    cap = cv2.VideoCapture(1)
    print("\n[알림] 프로그램이 준비되었습니다.")
    input()  # 사용자가 엔터키를 누를 때까지 여기서 대기해


    smoothed_steer = 0.0
    current_error = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # 1. YOLO로 좌우 마스크 분리 추출
        l_mask, r_mask = get_yolo_masks(frame)

        # 2. Bird's Eye View 변환
        w_left = warp_image(l_mask)
        w_right = warp_image(r_mask)

        # 3. 슬라이딩 윈도우 (좌우 독립 마스크 입력)
        lane_center, visual_img = get_sliding_window_center(w_left, w_right, current_error)

        # 4. 제어 연산
        current_error = lane_center - (PROCESS_WIDTH / 2)
        raw_steer_angle = math.degrees(math.atan2(current_error, HEIGHT / 2))

        if abs(raw_steer_angle) < DEADZONE: raw_steer_angle = 0.0
        smoothed_steer = (ALPHA * raw_steer_angle) + ((1.0 - ALPHA) * smoothed_steer)
        steer_cmd = int(smoothed_steer * STEER_GAIN)

        send_command(ser, BASE_SPEED, steer_cmd)

        # 시각화 데이터 합성
        cv2.imshow("Original with YOLO", frame)
        cv2.imshow("Warped Segmentation", visual_img)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if ser:
        send_command(ser, 0, 0)
        ser.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
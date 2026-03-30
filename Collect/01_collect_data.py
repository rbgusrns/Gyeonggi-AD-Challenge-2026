import cv2
import pygame
import os
import csv
import pandas as pd
import serial
import time
from datetime import datetime
from utils.config import *
# preprocess는 더 이상 수집 루프에서 사용하지 않으므로 제외하거나 남겨두어도 무방합니다.

# ===================== 시리얼 설정 =====================
SERIAL_PORT = 'COM17'
BAUD_RATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    ser.write(b"MAN\n")
    print(f"아두이노 연결 성공: {SERIAL_PORT}")
except Exception as e:
    print(f"시리얼 연결 실패: {e}")
    ser = None


# ===================== 게임패드 제어 클래스 =====================
class GamepadController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"게임패드 '{self.joystick.get_name()}' 연결 완료")
        else:
            print("경고: 게임패드를 찾을 수 없습니다.")

    def vibrate(self, duration=500, low=0.0, high=0.1):
        if self.joystick:
            self.joystick.rumble(low, high, duration)

    def get_inputs(self):
        if not self.joystick: return 0.0, 0.0, None
        pygame.event.pump()

        # 조향 (왼쪽 스틱 좌우: -1.0 ~ 1.0)
        steering = round(self.joystick.get_axis(0), 2)
        if abs(steering) < 0.05: steering = 0.0

        # 스로틀 (RT - LT: -1.0 ~ 1.0)
        forward = (self.joystick.get_axis(5) + 1.0) / 2.0
        reverse = (self.joystick.get_axis(4) + 1.0) / 2.0
        throttle = round(forward - reverse, 2)

        action = None
        # 버튼 인덱스 (X=2, Y=3, LB=4, RB=5)
        if self.joystick.get_button(2) and self.joystick.get_button(3):
            action = "DELETE_LAST_100"
        elif self.joystick.get_button(5): # RB
            action = "RECORD_START"
        elif self.joystick.get_button(4): # LB
            action = "RECORD_STOP"

        return steering, throttle, action


# ===================== 데이터 삭제 기능 =====================
def delete_last_100(label_file, image_path, controller):
    if not os.path.exists(label_file): return 0
    try:
        if os.path.getsize(label_file) == 0: return 0
        df = pd.read_csv(label_file, header=None, names=['img_name', 'steering'])

        total_len = len(df)
        if total_len == 0: return 0

        num_to_delete = min(total_len, 100)
        to_delete = df.tail(num_to_delete)
        df_remaining = df.head(total_len - num_to_delete)

        for img_name in to_delete['img_name']:
            img_p = os.path.join(image_path, img_name)
            if os.path.exists(img_p):
                os.remove(img_p)

        df_remaining.to_csv(label_file, index=False, header=False)
        controller.vibrate(400, low=1.0, high=1.0)
        print(f"\n[삭제 완료] {num_to_delete}장 삭제 (남은 데이터: {len(df_remaining)}장)")
        return num_to_delete
    except Exception as e:
        print(f"\n[삭제 에러] {e}")
        return 0


# ===================== 메인 루프 =====================
controller = GamepadController()
cap = cv2.VideoCapture(0)
recording, count, last_action = False, 0, None

print("조작 안내: RB(녹화시작), LB(녹화중지), X+Y동시(데이터삭제), Q(종료)")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        steering, throttle, action = controller.get_inputs()

        # 1. 아두이노 실시간 명령 전송
        if ser and ser.is_open:
            arduino_speed = int(throttle * 255)
            arduino_steer = int(steering * -100)
            cmd = f"S {arduino_speed} {arduino_steer}\n"
            ser.write(cmd.encode())

        # 2. 버튼 이벤트 처리
        if action != last_action:
            if action == "RECORD_START":
                recording = True
                print(">> 녹화 시작 (원본 저장 모드)")
                controller.vibrate(duration=50)
            elif action == "RECORD_STOP":
                recording = False
                print(">> 녹화 중지")
                controller.vibrate(duration=30)
            elif action == "DELETE_LAST_100":
                was_rec = recording
                recording = False
                deleted = delete_last_100(LABEL_FILE, IMAGE_PATH, controller)
                count -= deleted
                if count < 0: count = 0
                recording = was_rec
            last_action = action

        # 3. 데이터 저장 (수정 포인트: 전처리 없이 원본 frame 저장)
        if recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_name = f"img_{timestamp}.jpg"

            # 원본 프레임을 그대로 저장 (BGR 형식 유지)
            cv2.imwrite(os.path.join(IMAGE_PATH, img_name), frame)

            with open(LABEL_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([img_name, steering])

            count += 1
            if count % 50 == 0:
                controller.vibrate(duration=10)

        # 4. 화면 출력
        status = "REC" if recording else "IDLE"
        cv2.putText(frame, f"S:{steering} T:{throttle} | C:{count} | {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Data Capture & Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if ser and ser.is_open:
                ser.write(b"STOP\n")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    if ser:
        ser.close()
import cv2
import numpy as np


def nothing(x): pass


cv2.namedWindow("Tuning")
# 초기값은 대략적인 사다리꼴 형태
cv2.createTrackbar("Top_X", "Tuning", 200, 320, nothing)
cv2.createTrackbar("Top_Y", "Tuning", 300, 480, nothing)
cv2.createTrackbar("Bottom_X", "Tuning", 50, 320, nothing)
cv2.createTrackbar("Bottom_Y", "Tuning", 450, 480, nothing)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (640, 480))

    tx, ty = cv2.getTrackbarPos("Top_X", "Tuning"), cv2.getTrackbarPos("Top_Y", "Tuning")
    bx, by = cv2.getTrackbarPos("Bottom_X", "Tuning"), cv2.getTrackbarPos("Bottom_Y", "Tuning")

    src = np.float32([[320 - tx, ty], [320 + tx, ty], [320 - bx, by], [320 + bx, by]])
    dst = np.float32([[150, 0], [490, 0], [150, 480], [490, 480]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (640, 480))

    # 가이드 라인 그리기
    for p in src:
        cv2.circle(frame, tuple(p.astype(int)), 5, (0, 0, 255), -1)

    cv2.imshow("Original (Press Q to Save)", frame)
    cv2.imshow("Bird's Eye View", warped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\n[복사용 좌표]\nSRC_POINTS = np.float32({src.tolist()})")
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import time
from src.model import Model
import threading

play_model = Model(threshold=0.3, batch_size=16)

cap = cv2.VideoCapture(0)

last_capture_time = 0


def display_camera():
    while True:
        # 获取当前画面
        ret, frame = cap.read()
        if ret:
            # 显示画面
            cv2.imshow('frame', frame)

        # 检测是否按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def pipeline_task():
    global last_capture_time
    while True:
        current_time = time.time()
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('./resources/input.jpg', frame)
            answer = play_model.pipeline('./resources/input.jpg')

            # 检查answer内参数数量
            while len(answer) < 1:
                answer = play_model.pipeline('./resources/input.jpg')

            print(answer)

            last_capture_time = current_time


def run_threads():
    # 创建两个线程
    display_thread = threading.Thread(target=display_camera)
    pipeline_thread = threading.Thread(target=pipeline_task)

    # 启动线程
    display_thread.start()
    pipeline_thread.start()

    # 等待线程结束
    display_thread.join()
    pipeline_thread.join()


if __name__ == "__main__":
    run_threads()

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

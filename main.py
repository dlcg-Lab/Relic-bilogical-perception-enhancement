import cv2
import time
from src.model import Model
import threading

play_model = Model(threshold=0.3, batch_size=16)

cap = cv2.VideoCapture(0)

last_capture_time = 0


def display_camera():
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if ret:
            # Display the frame
            cv2.imshow('frame', frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def pipeline_task():
    global last_capture_time
    while True:
        current_time = time.time()
        ret, frame = cap.read()
        if ret:
            # Save the frame as an image
            cv2.imwrite('./resources/input.jpg', frame)
            # Process the image using the model
            answer = play_model.pipeline('./resources/input.jpg')

            while len(answer) < 1:
                # Retry processing if no answer is obtained
                answer = play_model.pipeline('./resources/input.jpg')

            # Print the answer
            print(answer)

            last_capture_time = current_time


def run_threads():
    # Create and start threads for camera display and pipeline tasks
    display_thread = threading.Thread(target=display_camera)
    pipeline_thread = threading.Thread(target=pipeline_task)

    display_thread.start()
    pipeline_thread.start()

    # Wait for both threads to finish
    display_thread.join()
    pipeline_thread.join()


if __name__ == "__main__":
    # Run the threads
    run_threads()

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

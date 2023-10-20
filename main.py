import cv2
import time
from src.viewermodel import ViewerModel
import threading

play_model = ViewerModel(llm="gpt-4", threshold=0.3, batch_size=16, debug=False, decay_rate=0.7, removal_threshold=0.3)

cap = cv2.VideoCapture(0)

last_capture_time = 0

class Relic:
    def __init__(self):
        self.answer = "test"

    def display_camera(self):
        # Create a named window with custom size
        cv2.namedWindow('Relic', cv2.WINDOW_NORMAL)

        while True:
            # Read frame from camera
            ret, frame = cap.read()

            if ret:
                # Split the frame into two parts
                height, width, _ = frame.shape
                split_line = height // 2
                top_frame = frame[:split_line, :]

                # Resize frames to fit the big window
                top_frame = cv2.resize(top_frame, (800, 300))

                # Create a new section with a gray background
                relic_section = 100  # height of the new section

                # Display the frame in the big window
                cv2.imshow('Relic', frame)

            # Exit if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def pipeline_task(self):
        global last_capture_time
        while True:
            current_time = time.time()
            ret, frame = cap.read()
            if ret:
                # Save the frame as an image
                cv2.imwrite('./resources/input.jpg', frame)
                # Process the image using the model
                self.answer = play_model.pipeline_str('./resources/input.jpg')

                # Print the answer
                print(self.answer)

                last_capture_time = current_time

            time.sleep(1)

def run_threads():
    relic = Relic()

    # Create and start threads for camera display and pipeline tasks
    display_thread = threading.Thread(target=relic.display_camera)
    pipeline_thread = threading.Thread(target=relic.pipeline_task)

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

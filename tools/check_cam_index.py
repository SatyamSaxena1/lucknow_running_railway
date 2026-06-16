import cv2
import os
import time

def check_and_preview_video_devices(max_devices=10):
    for i in range(max_devices):
        # dev_path = f"/dev/video{i}"
        dev_path = f"{i}"
        if os.path.exists(dev_path):
            print(f"\n📷 Found device: {dev_path}")
            cap = cv2.VideoCapture(dev_path)
            if cap.isOpened():
                print(f"  ✅ {dev_path} is usable - showing preview...")

                start_time = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("  ❌ Failed to grab frame.")
                        break

                    cv2.imshow(f"Camera {i} - {dev_path}", frame)

                    # Exit after 3 seconds or on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    if time.time() - start_time > 3:
                        break

                cap.release()
                cv2.destroyAllWindows()
            else:
                print(f"  ❌ {dev_path} exists but cannot be opened.")
        else:
            print(f"{dev_path} does not exist.")

if __name__ == "__main__":
    check_and_preview_video_devices()

import cv2

def list_connected_cameras(max_tested=5):
    available_cams = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cams.append(i)
            cap.release()
    return available_cams

# print("Available camera indices:", list_connected_cameras())

import cv2

def identify_cameras(indices):
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Showing camera at index {idx}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read from camera {idx}")
                    break
                cv2.imshow(f"Camera {idx}", frame)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:  # Press 'q' or ESC to close current cam
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"Camera at index {idx} could not be opened.")

indices = [0, 1, 2]
identify_cameras(indices)

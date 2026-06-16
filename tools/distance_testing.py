from ultralytics import YOLO
import cv2
import pyzed.sl as sl
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
import cupy as cp
from cupy.cuda import Device
import time
import logging
logging.basicConfig(level=logging.INFO)
yolo_model = 'yolo11x.pt'
detection_class = 39
model = YOLO(yolo_model)
zed = sl.Camera()
confidence_threshold_depth = 50
texture_confidence_threshold = 50
depth_maximum_distance = 10
depth_minimum_distance = 0.1
depth_stabilization = 90
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.depth_stabilization = depth_stabilization
init_params.coordinate_units = sl.UNIT.METER
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
init_params.depth_maximum_distance = depth_maximum_distance
init_params.depth_minimum_distance = depth_minimum_distance
runtime_params = sl.RuntimeParameters(enable_fill_mode=False, confidence_threshold = confidence_threshold_depth, texture_confidence_threshold = texture_confidence_threshold)
image = sl.Mat()
depth = sl.Mat()
device = Device(0)

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    logging.error("Camera Open Error: %s", repr(status))
    exit()

frame_depths = deque(maxlen=30)

def get_depth_values_in_roi(depth_data, x1, y1, x2, y2):
    with device:
        depth_array = cp.array(depth_data.get_data())
        roi_depth_values = depth_array[y1:y2, x1:x2]
        # print("roi_depth_values:", len(roi_depth_values))
        valid_mask = (roi_depth_values > depth_minimum_distance) & (roi_depth_values < depth_maximum_distance)
        valid_depth_values = roi_depth_values[valid_mask]
        # if len(valid_depth_values):
            # print("Valid Depth Values:", valid_depth_values)
        return valid_depth_values

def process_depth_values(depth_values):
    if len(depth_values) == 0:
        return float('inf')
    return float(cp.median(depth_values))

running_avg_depth = None
try:
    while True:
        current_time = time.time()
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            color_frame_rgb = image.get_data()[:, :, :3]
            color_frame_rgb = np.array(color_frame_rgb)

            frame_height, frame_width, _ = color_frame_rgb.shape
            results = model(color_frame_rgb, verbose=False, conf=0.30)

            for result in results:
                for box in result.boxes:
                    if int(box.cls) == detection_class:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        width = x2 - x1
                        height = y2 - y1
                        width_reduction = width * 0.1  # 20% reduction horizontally
                        height_reduction = height * 0.1  # 10% reduction vertically
                        x1 = int(x1 + width_reduction / 2)
                        x2 = int(x2 - width_reduction / 2)
                        y1 = int(y1 + height_reduction / 2)
                        y2 = int(y2 - height_reduction / 2)
                        depth_values = get_depth_values_in_roi(depth, x1, y1, x2, y2)
                        cv2.rectangle(color_frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        if depth_values.size > 0:
                            current_distance = process_depth_values(depth_values)
                            frame_depths.append(current_distance)
                            running_avg_depth = np.mean(frame_depths)
                            if running_avg_depth:
                                cv2.putText(color_frame_rgb, f"Avg Depth: {running_avg_depth:.2f} METER", (center_x, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.imshow("Color Frame", color_frame_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    zed.close()
    cv2.destroyAllWindows()

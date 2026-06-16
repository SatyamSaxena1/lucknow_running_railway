from ultralytics import YOLO
import cv2
import pyzed.sl as sl
import numpy as np
import cupy as cp
from cupy.cuda import Device
from collections import deque
import time
import logging

# ================= CONFIG =================
logging.basicConfig(level=logging.INFO)

YOLO_MODEL = "yolo11x.pt"
DETECTION_CLASS = 39   # bottle
CONF_THRES = 0.30

DEPTH_MIN = 0.1
DEPTH_MAX = 10.0
TEMPORAL_WINDOW = 15

# ================= INIT YOLO =================
model = YOLO(YOLO_MODEL)

# ================= INIT ZED =================
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.depth_minimum_distance = DEPTH_MIN
init_params.depth_maximum_distance = DEPTH_MAX
init_params.depth_stabilization = 90

runtime_params = sl.RuntimeParameters(
    confidence_threshold=50,
    texture_confidence_threshold=50,
    enable_fill_mode=False
)

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    logging.error("Failed to open ZED camera")
    exit(1)

# ================= CAMERA INTRINSICS =================
cam_info = zed.get_camera_information()
fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
fy = cam_info.camera_configuration.calibration_parameters.left_cam.fy
cx = cam_info.camera_configuration.calibration_parameters.left_cam.cx
cy = cam_info.camera_configuration.calibration_parameters.left_cam.cy

image = sl.Mat()
point_cloud = sl.Mat()

device = Device(0)

# ================= TEMPORAL STORAGE =================
position_buffer = deque(maxlen=TEMPORAL_WINDOW)

# ================= FUNCTIONS =================
def get_xyz_points_in_roi(point_cloud, x1, y1, x2, y2):
    with device:
        pc = cp.array(point_cloud.get_data())  # H x W x 4
        roi = pc[y1:y2, x1:x2, :3]

        valid = cp.isfinite(roi).all(axis=2)
        roi = roi[valid]

        return roi.get()  # return NumPy for downstream ops


def compute_3d_centroid(xyz_points):
    if xyz_points.shape[0] == 0:
        return None
    return np.median(xyz_points, axis=0)


def compute_3d_bbox(xyz_points):
    if xyz_points.shape[0] == 0:
        return None, None
    return np.min(xyz_points, axis=0), np.max(xyz_points, axis=0)


def get_3d_bbox_corners(min_xyz, max_xyz):
    x_min, y_min, z_min = min_xyz
    x_max, y_max, z_max = max_xyz

    return np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ])


def project_3d_to_2d(points_3d):
    pts_2d = []
    for X, Y, Z in points_3d:
        if Z <= 0:
            pts_2d.append(None)
            continue
        u = int((X * fx) / Z + cx)
        v = int((Y * fy) / Z + cy)
        pts_2d.append((u, v))
    return pts_2d


def draw_3d_bbox(frame, corners_2d):
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i, j in edges:
        if corners_2d[i] is not None and corners_2d[j] is not None:
            cv2.line(frame, corners_2d[i], corners_2d[j], (0, 0, 255), 2)

# ================= MAIN LOOP =================
try:
    while True:
        start_time = time.time()

        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

        frame = image.get_data()[:, :, :3].copy()
        results = model(frame, verbose=False, conf=CONF_THRES)

        for r in results:
            for box in r.boxes:
                if int(box.cls) != DETECTION_CLASS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                w = x2 - x1
                h = y2 - y1
                x1 += int(0.1 * w)
                x2 -= int(0.1 * w)
                y1 += int(0.1 * h)
                y2 -= int(0.1 * h)

                xyz_points = get_xyz_points_in_roi(point_cloud, x1, y1, x2, y2)

                centroid = compute_3d_centroid(xyz_points)
                if centroid is not None:
                    position_buffer.append(centroid)
                    avg_pos = np.mean(position_buffer, axis=0)
                    X, Y, Z = avg_pos

                    cv2.putText(
                        frame,
                        f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f} m",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2
                    )

                min_xyz, max_xyz = compute_3d_bbox(xyz_points)
                if min_xyz is not None:
                    corners_3d = get_3d_bbox_corners(min_xyz, max_xyz)
                    corners_2d = project_3d_to_2d(corners_3d)
                    draw_3d_bbox(frame, corners_2d)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("3D Bounding Box (XYZ)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    zed.close()
    cv2.destroyAllWindows()

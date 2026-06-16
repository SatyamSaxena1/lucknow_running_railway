import sys
import time
import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,QTabWidget, QTableWidget, QTableWidgetItem, QSpinBox, QGroupBox,
QLineEdit, QButtonGroup, QHeaderView, QSplitter, QRadioButton, QFileDialog, QComboBox,QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QColor
from old_models_realtimeprocessor import RealtimeProcessor
from zed_implantation import ZEDProcessor
from line_profiler import profile
from ultralytics import YOLO
import logging
from multiprocessing import shared_memory
# import win32event
import subprocess
import os
from mpeg_writer import FFmpegWriter

logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_zed_pixmap_signal = pyqtSignal(np.ndarray)
    update_logs_signal = pyqtSignal(dict)
    update_camera_info_signal = pyqtSignal(str)
    update_gps_info_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    alert_signal = pyqtSignal(bool)
    @profile
    def __init__(self, conf_threshold, train_height, selected_panto_width, camera_offset,roof_to_frame_height,gps_excel, selected_gps_port,selected_cam_index, mast_direction, height_correction, setting_distance_correction, stagger_distance_correction):

        super().__init__()
        self.conf_threshold = conf_threshold
        self.train_height = train_height
        self.pantograph_width = selected_panto_width
        self.gps_excel = gps_excel
        self.com_port = selected_gps_port
        self.selected_cam_index = selected_cam_index
        self.height_correction = height_correction
        self.roof_to_frame_height = roof_to_frame_height
        self.setting_distance_correction = setting_distance_correction
        self.stagger_distance_correction = stagger_distance_correction
        self.mast_direction = mast_direction

        self.webcam_processor = RealtimeProcessor(float(train_height), selected_panto_width, gps_excel, selected_gps_port, float(roof_to_frame_height) ,mast_direction ,float(height_correction),float(stagger_distance_correction))
        self.webcam_processor.start_gps_thread()

        self.zed_processor = ZEDProcessor(self.webcam_processor, float(camera_offset), setting_distance_correction)

        self.running = True
        self.model_path = os.path.join(os.getcwd(),"models","best_81_l.engine")
        self.model = YOLO(self.model_path, task='detect')
        self.fps_history = deque(maxlen=10)  # Store last 10 FPS measurements

        logger.info("VideoThread Init completed")

    def start_ffmpeg_writer(self, output_video, fps, width, height, codec="h264"):
        # Generate unique filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_parts = output_video.rsplit('.', 1)
        unique_output_video = os.path.join(self.webcam_processor.log_folder,f"{filename_parts[0]}_{timestamp}.{filename_parts[1]}")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f", "rawvideo",  # Raw video input
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",  # Input pixel format
            "-s", f"{width}x{height}",  # Input frame size
            "-r", str(fps),  # Frame rate
            "-i", "-",  # Input from stdin
            "-c:v", f"h264_nvenc" if codec == "h264" else "hevc_nvenc",  # Use GPU encoder
            "-preset", "fast",  # Encoding speed preset
            "-pix_fmt", "yuv420p",  # Output pixel format
            unique_output_video,
        ]
        return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


    @profile
    def run(self):
        logger.info("Webcam selected")
        cap = cv2.VideoCapture(self.selected_cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #1280  #1920
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #720 #1080
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #1280  #1920
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #720 #1080
        # cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_FPS,60) # 60

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nominal_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Nominal FPS: {nominal_fps}")
        logger.info(f"Width:{width}")
        logger.info(f"Height:{height}")

        if nominal_fps <= 0 or nominal_fps > 120:
            logger.info("Setting default FPS 30......")
            nominal_fps = 30

        if not cap.isOpened():
            logger.error("Unable to open video source ")
            self.update_camera_info_signal.emit("Error: Unable to open video source")
            return

        # FPS Calculation Improvements
        start_time = time.perf_counter()
        frame_count = 0
        sampling_interval = 1.0  # 1-second interval
        text = "GPS not connected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.3
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)


        self.shm = shared_memory.SharedMemory(name="shared_color")
        self.shm_depth = shared_memory.SharedMemory(name="shared_depth")
        self.frame_buffer = np.ndarray((720, 1280, 3), dtype=np.uint8, buffer=self.shm.buf)
        self.frame_shape_depth = (720,1280)
        self.frame_buffer_depth = np.ndarray(self.frame_shape_depth, dtype=np.float32, buffer=self.shm_depth.buf)


        # self.ffmpeg_process_webcam  = self.start_ffmpeg_writer("output_webcam.mp4", nominal_fps, 640, 360, codec="h264")
        # self.ffmpeg_process_depth_rgb = self.start_ffmpeg_writer("output_depth_rgb.mp4", nominal_fps, 640, 360, codec="h264")

        # new class writer
        self.ffmpeg_process_webcam  = FFmpegWriter("output_webcam.mp4", nominal_fps, 640, 360, self.webcam_processor.log_folder ,codec="h264")
        self.ffmpeg_process_depth_rgb = FFmpegWriter("output_depth_rgb.mp4", nominal_fps, 640, 360, self.webcam_processor.log_folder, codec="h264")

        while self.running:
            frame_start_time = time.perf_counter()
            # Wait for Zed Frames
            if os.path.exists("color.flag") and os.path.exists("depth.flag"):
                try:
                    self.zed_processor.zed_rgb_frame = self.frame_buffer.copy()
                    self.zed_processor.depth = self.frame_buffer_depth.copy()
                    os.remove("color.flag")
                    os.remove("depth.flag")
                except Exception as e:
                    logger.warning(f"Error while reading ZED shared memory: {e}")

            ret, webcam_rgb_frame = cap.read()
            if ret:
                frame_count += 1
                if not self.webcam_processor.gps_data["connected"]:
                    height, width = webcam_rgb_frame.shape[:2]
                    x = int((width - text_width) / 2)
                    y = text_height + 20
                    cv2.putText(webcam_rgb_frame, text, (x, y), font, font_scale, (0, 0, 255), thickness)
                # Inference on Color Frames
                self.detection_results_web = self.model(webcam_rgb_frame, verbose=False, conf=self.conf_threshold,task='detect')
                self.detection_results_depth = self.model(self.zed_processor.zed_rgb_frame, verbose=False, conf=self.conf_threshold,task='detect')

                processed_frame_webcam, logs = self.webcam_processor.process_frame(webcam_rgb_frame, nominal_fps, self.detection_results_web)
                # Write processed frame to video
                small_webcam_frame = cv2.resize(processed_frame_webcam, (640, 360))

                try:
                    self.ffmpeg_process_webcam.write(small_webcam_frame.tobytes())
                except BrokenPipeError:
                     logger.error("FFmpeg process (webcam) crashed or closed unexpectedly.")

                self.change_pixmap_signal.emit(processed_frame_webcam)
                self.update_logs_signal.emit(logs)


                zed_frame, setting_distance, alert_status, mast_data = self.zed_processor.process_frame(self.detection_results_depth)
                if zed_frame is not None:
                    self.change_zed_pixmap_signal.emit(zed_frame)
                    self.alert_signal.emit(alert_status)
                    small_zed_frame = cv2.resize(zed_frame, (640, 360))
                    try:
                        self.ffmpeg_process_depth_rgb.write(small_zed_frame.tobytes())
                        # self.ffmpeg_process_depth_rgb.stdin.write(small_zed_frame.tobytes())
                    except BrokenPipeError:
                        logger.error("FFmpeg process (ZED) crashed or closed unexpectedly.")

                    if setting_distance is not None:
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

                        # Get lat, lon, and mast name from mast_data if available
                        lat = "N/A"
                        lon = "N/A"
                        mast_name = "Unknown"

                        if mast_data and 'mast' in mast_data:
                            _, mast_name, lat, lon = mast_data['mast']

                        logs['setting_distance'] = [
                            current_time,
                            f"{setting_distance:.2f}",
                            "Yes" if alert_status else "No",
                            mast_name,  # Use mast_name instead of current_mast_area
                            lat,        # Use lat from mast_data
                            lon         # Use lon from mast_data
                        ]
                        self.update_logs_signal.emit(logs)


                # Improved FPS Calculation
                current_time = time.perf_counter()
                if current_time - start_time >= sampling_interval:
                    frame_processing_time = current_time - start_time
                    measured_fps = frame_count / max(frame_processing_time, 0.001)

                    # Add to history and calculate moving average
                    self.fps_history.append(measured_fps)
                    avg_fps = sum(self.fps_history) / len(self.fps_history)

                    camera_info = f"ACTUAL FPS: {measured_fps:.2f} fps (Avg: {avg_fps:.2f} fps)"
                    self.update_camera_info_signal.emit(camera_info)

                    # Reset for next interval
                    start_time = current_time
                    frame_count = 0

                # Optional: Control frame rate if processing is too fast
                frame_processing_duration = time.perf_counter() - frame_start_time
                if frame_processing_duration < 1/nominal_fps:
                    time.sleep(max(0, 1/nominal_fps - frame_processing_duration))

                gps_info = f"Speed : {self.webcam_processor.gps_data.get('speed')}"
                self.update_gps_info_signal.emit(gps_info)

        logger.info("Video Thread Stopped.")
        cap.release()
        self.finished_signal.emit()

    def stop(self):
        self.running = False

        try:
            self.webcam_processor.automate_all_log()
            logger.info("Called automate_all_log() successfully.")
        except Exception as e:
            logger.error(f"Error calling automate_all_log(): {e}")

        try:
            open("all_shutdown.flag", "w").close()
            logger.info("Shutdown flag created: all_shutdown.flag")
        except Exception as e:
            logger.warning("Could not create shutdown flag:", e)

        # Cleanup shared memory
        for shd in [self.shm, self.shm_depth]:
            try:
                shm_name = shd.name if hasattr(shd, "name") else str(shd)
                shd.close()
                shd.unlink()
                logger.info(f"Unlinked shared memory: {shm_name}")
            except FileNotFoundError:
                logger.warning(f"Shared memory not found for unlinking: {shm_name}")
            except Exception as e:
                logger.error(f"Error unlinking shared memory {shm_name}: {e}")

        # Cleanup FFmpeg processes and stop threads
        try:
            self.ffmpeg_process_webcam.close()
            # self.ffmpeg_process_webcam.stdin.close()
            self.ffmpeg_process_webcam.wait()
            logger.debug("Webcam FFmpeg process closed and waited.")
        except Exception as e:
            logger.warning(f"Error closing webcam FFmpeg process: {e}")


        for proc in [self.ffmpeg_process_depth_rgb]:
            try:
                proc.close()
                # proc.stdin.close()
                proc.wait()
                logger.debug("Depth FFmpeg process closed and waited.")
            except Exception as e:
                logger.warning(f"Error closing depth FFmpeg process: {e}")

        # Ensure graceful exit of QApplication (if used in your GUI)
        try:
            QApplication.instance().quit()
            logger.info("Qt application instance quit.")
        except Exception as e:
            logger.warning(f"Error quitting QApplication: {e}")

class App(QMainWindow):
    @profile
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Pantograph Alert Generation")
        self.setGeometry(100, 100, 900, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        upper_widget = QWidget()
        upper_layout = QHBoxLayout(upper_widget)

        left_widget = QWidget()
        left_widget.setMinimumWidth(320)
        left_widget.setMaximumWidth(400)
        # left_widget.setFixedWidth(int(self.width() * 0.3))
        left_layout = QVBoxLayout(left_widget)

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)
        # controls_group.setFixedSize(350, 550)

        # confidence - camera_id row layout
        conf_cam_row_layout = QHBoxLayout()
        self.conf_threshold_input = QLineEdit()
        self.conf_threshold_input.setText("0.35")

        self.cam_index_selector = QComboBox()
        cam_indexes = ["Camera 0", "Camera 1", "Camera 2", "Camera 3", "Camera 4"]
        self.cam_index_selector.addItems(cam_indexes)
        self.cam_index_selector.setCurrentIndex(0)

        # train height - panto layout
        train_panto_layout = QHBoxLayout()
        self.train_height_spinbox = QLineEdit()
        self.train_height_spinbox.setText("3500")

        self.pantograph_width_selector = QComboBox()
        panto_widths = ["1800","2030"]
        self.pantograph_width_selector.addItems(panto_widths)
        self.pantograph_width_selector.setCurrentIndex(0)

        #mast offset - gps port layout
        mast_offset_gps_port_layout = QHBoxLayout()
        self.camera_offset_spinbox = QLineEdit()
        self.camera_offset_spinbox.setText("28") # Default value

        self.gps_port_selector = QComboBox()
        gps_ports = [
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
            "COM9", "COM10", "COM11", "COM12", "COM13", "COM14", "COM15", "COM16",
            "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3",
            "/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyS0", "/dev/ttyS1"
        ]
        self.gps_port_selector.addItems(gps_ports)
        self.gps_port_selector.setCurrentIndex(12)

        #mast offset - gps port layout
        roof_to_frame_height_layout = QHBoxLayout()
        self.roof_to_frame_spinbox = QLineEdit()
        self.roof_to_frame_spinbox.setText("900")

        self.gps_excel_layout = QHBoxLayout()
        self.gps_excel_input = QLineEdit()
        self.gps_excel_input.setReadOnly(True)  # Make input read-only
        self.gps_excel_upload_button = QPushButton("Upload GPS Excel")
        self.gps_excel_upload_button.clicked.connect(self.upload_gps_excel)

        #mast direction section
        mast_direction_layout = QHBoxLayout()
        self.mast_direction_layout_radio_left = QRadioButton("Left")
        self.mast_direction_layout_radio_right = QRadioButton("Right")
        self.mast_direction_radio_group = QButtonGroup()
        self.mast_direction_radio_group.addButton(self.mast_direction_layout_radio_left)
        self.mast_direction_radio_group.addButton(self.mast_direction_layout_radio_right)
        self.mast_direction_layout_radio_left.setChecked(True)

        # Height Correction Section
        height_correction_layout = QHBoxLayout()
        self.height_correction_radio_plus = QRadioButton("+")
        self.height_correction_radio_minus = QRadioButton("-")
        self.height_correction_radio_group = QButtonGroup()
        self.height_correction_radio_group.addButton(self.height_correction_radio_plus)
        self.height_correction_radio_group.addButton(self.height_correction_radio_minus)
        self.height_correction_radio_plus.setChecked(True)  # Default to plus

        self.height_correction_input = QSpinBox()
        self.height_correction_input.setRange(0, 1000)  # Adjust range as needed
        self.height_correction_input.setValue(0)

        # Setting Distance Correction Section
        setting_distance_correction_layout = QHBoxLayout()
        self.setting_distance_correction_radio_plus = QRadioButton("+")
        self.setting_distance_correction_radio_minus = QRadioButton("-")
        self.setting_distance_correction_radio_group = QButtonGroup()
        self.setting_distance_correction_radio_group.addButton(self.setting_distance_correction_radio_plus)
        self.setting_distance_correction_radio_group.addButton(self.setting_distance_correction_radio_minus)
        self.setting_distance_correction_radio_plus.setChecked(True)  # Default to plus

        self.setting_distance_correction_input = QSpinBox()
        self.setting_distance_correction_input.setRange(0, 1000)  # Adjust range as needed
        self.setting_distance_correction_input.setValue(0)

        # stagger correction
        stagger_distance_correction_layout = QHBoxLayout()
        self.stagger_distance_correction_radio_plus = QRadioButton("+")
        self.stagger_distance_correction_radio_minus = QRadioButton("-")
        self.stagger_distance_correction_radio_group = QButtonGroup()
        self.stagger_distance_correction_radio_group.addButton(self.stagger_distance_correction_radio_plus)
        self.stagger_distance_correction_radio_group.addButton(self.stagger_distance_correction_radio_minus)
        self.stagger_distance_correction_radio_plus.setChecked(True)  # Default to plus

        self.stagger_distance_correction_input = QSpinBox()
        self.stagger_distance_correction_input.setRange(0, 1000)  # Adjust range as needed
        self.stagger_distance_correction_input.setValue(0)

        conf_cam_row_layout.addWidget(QLabel("Threshold:"))
        conf_cam_row_layout.addWidget(self.conf_threshold_input,stretch=2)
        conf_cam_row_layout.addSpacing(5)  # optional spacing between inputs
        conf_cam_row_layout.addWidget(QLabel("Webcam Index:"))
        conf_cam_row_layout.addWidget(self.cam_index_selector,stretch=1)
        controls_layout.addLayout(conf_cam_row_layout)

        train_panto_layout.addWidget(QLabel("Train Height:"))
        train_panto_layout.addWidget(self.train_height_spinbox,stretch=2)
        train_panto_layout.addSpacing(5)  # optional spacing between inputs
        train_panto_layout.addWidget(QLabel("Pantograph (mm):"))
        train_panto_layout.addWidget(self.pantograph_width_selector,stretch=1)
        controls_layout.addLayout(train_panto_layout)

        mast_offset_gps_port_layout.addWidget(QLabel("Cam Offset:"))
        mast_offset_gps_port_layout.addWidget(self.camera_offset_spinbox,stretch=2)
        mast_offset_gps_port_layout.addSpacing(5)  # optional spacing between inputs
        mast_offset_gps_port_layout.addWidget(QLabel("GPS :"))
        mast_offset_gps_port_layout.addWidget(self.gps_port_selector,stretch=1)
        controls_layout.addLayout(mast_offset_gps_port_layout)

        roof_to_frame_height_layout.addWidget(QLabel("Roof to frame height (mm):"))
        roof_to_frame_height_layout.addWidget(self.roof_to_frame_spinbox,stretch=2)
        controls_layout.addLayout(roof_to_frame_height_layout)

        self.gps_excel_layout.addWidget(QLabel("GPS File:"))
        self.gps_excel_layout.addWidget(self.gps_excel_input)
        self.gps_excel_layout.addWidget(self.gps_excel_upload_button)
        controls_layout.addLayout(self.gps_excel_layout)

        mast_direction_layout.addWidget(QLabel("Mast Direction :"))
        mast_direction_layout.addWidget(self.mast_direction_layout_radio_left)
        mast_direction_layout.addWidget(self.mast_direction_layout_radio_right)
        controls_layout.addLayout(mast_direction_layout)

        height_correction_layout.addWidget(QLabel("Height Correction:"))
        height_correction_layout.addWidget(self.height_correction_radio_plus)
        height_correction_layout.addWidget(self.height_correction_radio_minus)
        height_correction_layout.addWidget(self.height_correction_input)
        controls_layout.addLayout(height_correction_layout)

        setting_distance_correction_layout.addWidget(QLabel("Implantation Correction:"))
        setting_distance_correction_layout.addWidget(self.setting_distance_correction_radio_plus)
        setting_distance_correction_layout.addWidget(self.setting_distance_correction_radio_minus)
        setting_distance_correction_layout.addWidget(self.setting_distance_correction_input)
        controls_layout.addLayout(setting_distance_correction_layout)

        stagger_distance_correction_layout.addWidget(QLabel("Stagger Correction:"))
        stagger_distance_correction_layout.addWidget(self.stagger_distance_correction_radio_plus)
        stagger_distance_correction_layout.addWidget(self.stagger_distance_correction_radio_minus)
        stagger_distance_correction_layout.addWidget(self.stagger_distance_correction_input)
        controls_layout.addLayout(stagger_distance_correction_layout)

        self.source_button_group = QButtonGroup()

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.setStyleSheet("background-color: #00ff2a; color: #121110")
        self.stop_button.setStyleSheet("background-color: #ff0000; color: #121110")
        self.stop_button.setEnabled(False)

        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        self.camera_info_label = QLabel()
        self.camera_info_label.setWordWrap(True)
        self.camera_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        controls_layout.addWidget(self.camera_info_label)

        self.gps_info_label = QLabel()
        self.gps_info_label.setWordWrap(True)
        self.gps_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        controls_layout.addWidget(self.gps_info_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_group)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ccc;
                background-color: #f9f9f9;
            }
        """)
        left_layout.addWidget(scroll, stretch=3)
        # left_layout.addWidget(controls_group)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        video_layout = QHBoxLayout()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.zed_video_label = QLabel()
        self.zed_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.zed_video_label)

        right_layout.addLayout(video_layout)

        self.alert_label = QLabel()
        self.alert_label.setStyleSheet("QLabel { color: black; font-size: 16px; }")
        right_layout.addWidget(self.alert_label)

        upper_layout.addWidget(left_widget)
        upper_layout.addWidget(right_widget)

        lower_widget = QWidget()
        lower_layout = QVBoxLayout(lower_widget)

        self.tabs = QTabWidget()
        self.stagger_table = QTableWidget()
        self.height_table = QTableWidget()
        self.setting_table = QTableWidget()
        self.gradient_table = QTableWidget()
        self.double_contact_table = QTableWidget()

        self.tabs.addTab(self.stagger_table, "Stagger Alerts")
        self.tabs.addTab(self.height_table, "Height Alerts")
        self.tabs.addTab(self.setting_table, "Setting Distance Alerts")
        self.tabs.addTab(self.gradient_table, "Gradient Alerts")
        self.tabs.addTab(self.double_contact_table, "Double Contact Alerts")
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                background: lightgray;
                padding: 10px;
                color: black;
            }
            QTabBar::tab:selected {
                background: lightblue;
                color: black;
            }
            QTabBar::tab:!selected {
                background: lightgray;
                color: black;
            }
            QTabBar::tab:hover {
                background: lightgreen;
                color: black;
            }
        """)

        lower_layout.addWidget(self.tabs)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(upper_widget)
        splitter.addWidget(lower_widget)
        main_layout.addWidget(splitter)

        self.setup_tables()

        # Load embedded image
        logo_layout = QHBoxLayout()
        logo_layout.addStretch()
        self.powered_by_label = QLabel("Researched, Designed, and Developed by : ", self)
        self.powered_by_label.setStyleSheet("color: black; font-size: 14px;")
        logo_layout.addWidget(self.powered_by_label)

        self.logo_label = QLabel(self)
        pixmap = QPixmap()
        pixmap.loadFromData(self.get_embedded_image())
        self.logo_label.setPixmap(pixmap.scaled(100, 100,Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo_layout.addWidget(self.logo_label)
        main_layout.addLayout(logo_layout)

        self.setLayout(main_layout)

        self.start_button.clicked.connect(self.start_video_processing)
        self.stop_button.clicked.connect(self.stop_video_processing)

    def get_gps_port(self):
        return self.gps_port_selector.currentText()

    def get_selected_camera_index(self):
        """Get the selected camera index"""
        return self.cam_index_selector.currentIndex()

    def get_panto_width(self):
        """Get the selected panto width"""
        return self.pantograph_width_selector.currentText()

    def get_embedded_image(self):
        import base64
        image_data = b"""/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCACqAfQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9U6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKTcPWgA3D1ozXh3xk/a28D/CFp7F7ltf16P5TpensGKN6SyH5U9xy3+zXyf4o/wCCgvj/AFe6lFhYabo1gxISGAM0oHo0rE5P+0qj6d6+vy3hTNc0h7WlT5Yd5aJ+nV/JWPncbn+BwMnCcuaXaOv/AAPxP0hZgqksQAOpJrGu/G3h3T5GS61/S7Z16rNeRqR+BavzFuf2qrzXJIRr1lfXYB+eY37TMPcIwA/8eHetzRfiR4d8VMqWd8sNy/S1ux5Uuc491Y98AmvVqcFY7DLmrp28lf8Ar7j4fFccV6bbo4W6XVy/RL9T9G7fx54ZvGAg8RaTMT2jvom/k1bMFxFdRiSGVJY26NGwYH8RX5tXi/eB7HBGORVO11jUtDn8/TNQutPnPSS1neJvzDD/AD0rj/1a517lXXzX/B/Q5KHiE+a1fDaeUv0a/U/TSjcPWvgTw3+1V8QfCsirLqMet2y8eTqUQcn/AIGu189+WI7YNe7/AA6/bG8J+K5IrHxBG3he+fjzZ332jN/10wCnr84AHTJrysVw/jsLHn5eaPeOv4bn2mX8WZXmDUFPkl2lp+O34n0LRUcFxFdQxzQypLDIodJI2DKynoQR1FSV84fYhRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUm4DqRQBBe31tp1nNdXU8dtbQoZJZpXCoigZLMTwAB3NfBn7SP7ZN/4omuvD3gO6m07Qxuin1WPKXF32Pl9DGnvkM3GcDIMP7W/7Rkvj/Urjwn4duyvhi0fbczxHi/lB65zzGp6DoSN3Py4+YZkr9x4T4Sp04xx+YwvJ6xi9l2bXfy6eu35DxDxPKpOWDwMrRWjkuvkvLu+vpvlzJ97PJ9f51SljrUlj61TmXrX7VBn57GRlTR1Tmj9M57Y/StSaPHb2qlNHjIPUda64tPRnZCW1mdZ4P+LOpeHTFaagzalpa4Xy3OZIhjHyMeoxj5W4wMZHWvXrHVrPX9OjvrCdbi2kO0EcYbj5SOx5HHpg96+aZErW8H+LrnwfqvnoWkspjsubfPEqjPPswySD9c8EivmszyOliYuth1afls/+CY4jBwrrmhpI9zu161jXSZyetbS3UGpWMV3ayrNbTL5kci9Mf06EEdc8Ve8GeBdR+I3i7T9B01T9pvHw0jDKxx8lpG9gDn8sckV8J7SOHUnW0Ub3v5Hz9KjVqVI0IK8m7JeZ9AfsN23jG6uNTum1OdPBVuDEtpNh0luTz+7yPk2gksV6krnPb7Grn/BPg3TvAPhfTtB0qHyrOzjCLxy7dWdj3ZiST7mugr8OzTGrMMXPERjZPbS2nd+b6n9Q5PgZZbgqeGnJyaWut9ey8l0CiiivKPZCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvnT9sT4vP4E8Fp4b0uby9Y1tGWR1PzQ2vRyPQvyo9g/cCvolpFjUszBVUZJJwBX5gfGvx3J8S/iRrOuFy9s0phsw3RbdflQDjjIG4jplvxr7jhHKlmOYKpUV4U9X5vov1+R8Pxdmry7AclN2nU0Xp1f6fM82mTkn3zzVGWPtWrImaqSQ5Qs5WOMEZkfoM9Pr06da/pGMktD8KwtGti6qo0YuU3sluZMydq2/BPwv8U/EzVF0/wANaLdanMfvNGuIox6tISFUcjqe9bfwo+H2p/F7xxaeG9AiUSyHzLm+uIy0dtCuN8hQHGBu2jdksSo4yK/Tz4Y/C/RfhP4Yg0bRYG2qAZ7uY7p7qTGDJI3c+g6AcAAAV8VxJxXDJEqNGKlVfR7Jd3+iP1vB8C4ilGMsxnyt/ZW69eiPl74d/wDBOfT4YYbnxt4gnuZ2UF9P0YCKNT12mVgSw9gq/j1r2fSv2N/hBpMKRr4MguivWS7uJpSx9SGfA/AD6V7ZRX4fi+JM3xsm6uJlbsnyr7lZH3WHyfAYVJU6S9Wrv72eH6t+xb8HNYVxJ4LhgZhw9rd3EJU+oCyY/Q/SvHPH/wDwTR8M6hbyTeD/ABJqGkXvUQamq3MDccLuUK6/U7vpX2lkUtLCcR5vgmnRxMvRvmX3O6LrZVgq6tOkvkrP70flVcfBPx78BtVk0XxZpfmaPcEvZaxZfvrQyDOV8w4KFh0VwCSvyj5s19w/svfB3/hX/hY6zqUG3XtWQO4cYaCHgqnsx4Zh64H8Ne1zW8V1C8c0KzROMMkihg3sQamGBXXmvEuJzaiqdWKi38TX2rbadPPXc8bA8NYTA4+WNg7u2ifR9X92g+ik3D1pa+RPrgooooAKKTcPWjcPWgBaKTcPWjNAC0UmR60bh60ALRSZo3D1oAWiiigAopNw9aNw9aAFopNw9RRQAtFFFABRSZHrS0AFFFJQAtFJuHqKMj1oAWikzS0AFFFJuHqKAFopMj1oyKAFopNw9aNw9aAFopNw9aWgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPOf2gfEreFfg54ovIn2Tvam1jYdQ0pEeR7gOT+FfmtKnWvvL9tC8a2+ENvErEfadVhiPuAkr4/wDHP0r4K1i+j0q1M8iCRs7Y42PDt6n2Gc/kMgnNfunAtFU8vnVtrKT/AASX+Z+LcU4fE5xnlHLcKry5Vb1bd3925n6pdRabCJJfmdgfLjHVh6n/AGen1x9SOWvL6a+k3ytx2RfugegH4f480y6upby4eeZzJKx3M38sfkOOwAAqz4f0s65r2m6cH2G8uY7cMO29gv8AWv0/SnBzn0P6N4Y4TwfDeGTilKrb3pPv1Svskfo3+xP8J4vh/wDCa31m5ixrPiLbeSsw5SDnyEHttO/6v7V9FVWs7OHT7WC1t41ht4UWKONRgKqjAA9gBVmv5Kx2Mnj8TUxNR6yd/wDJfJaHg1qsq9SVSW7CiiiuExPlL9sn4zeL/hf4g8NW3hbWm0pbi2mknUW8Uu/5lC58xGx0P514Fb/tlfFqH7/iSGf/AK6adbD/ANBjFd7+39N5nxI8OwZ4TSd//fU0g/8AZa+Xtq+lfbYCnhnhYe1ppvvbzP3/AIdyHB4rKqNWvRjJtN3sr7u2p7xbftwfE+DG640u4/662I/9lYV7d+zV+0V48+L3i82esWWkRaJHGwluba2lSQybSVVWMhX68Ht6g18deC/BF5421dbO1GyFfmnuGHyxKev1PoPp2r7k+APhmw8M6pp+nWMWyC3ic5YkszEYLk+pzz+XSvg+JuIsuy/E4fLMNTTrVZRTt9mN1d/NaI4uIcpyvA4SoqVGKnZu66HrnxQ1a70H4c+I9QsZjb3trZSSwzAAlWC8EA8V8Uf8NAfEL/oabz/vlP8A4mvs341/8kn8W/8AYOl/9Br89bVfMuIVYZBcA1+XcY4vE0MVShQqOKa6Nrr5HFwNg8LiMJWnXpRm1Lqk9LeZ3v8Aw0D8Qv8AoaLz/vmP/wCJpP8AhoD4hf8AQ03n5J/8RX2N/wAKH+H3/QqWH/fJ/wAaT/hQ/wAPv+hV0/8A75P+Nbf6t5z/ANBr/wDApnP/AK1ZF/0AL/wGB87/AAM+MnjPxJ8VND0zU9fuL2wuGkElvIiAMBC7DPy5zkA8EdK9V/am8da94G8PaLNoWoyabLcXTpK8aqSyhM45B/pXoOifCTwf4b1SHUdL8PWdlew58uaNOUyCpxnpwT+deQftp/8AIr+G/wDr9k/9AFelXwuNyvJMQq1Zynumm7pXWl3r3PJw+KwGccQYaVCgoU9nFpWb953stOx5R4L/AGlvGGjeJbG51nV5tV0sPturWREO+NjglSFGGHUeuMH3+3dO1C31Sxt7y1lWa0uI1lilQ5V1YZDD8K/MSvqf9kn4qefC/grUpsyRhp9OZz1X7zxZ9RksP+Bc4ArwOFM8qe3eDxc3JT+Ft3d+133/AD9T6XjHh2n9WWPwVNRcPiUVZW72Xbr5eh9C+KvE1l4P8P3+s6jJ5VlZxmSQ9z6KPUkkAepIr4i1r9o7x5qerXd1b67NYQSyFo7WFE2RLnhRlcnA4yTnjPWuy/au+Kv/AAkOvL4S06bOn6c+booeJbjkbf8AgAyP97d3Ar5/rm4nz6rUxX1bCVHGMN2na766rtt63OrhHhyjTwn1rG01KU9Umr2XTR9/ysfdv7N3izVvGfw1W/1m8e/vReSxec6qCVG0joB6mvHv2ivix4u8KfE2707SdcuNPsY4ISkUQXALKCx6e/vXpn7In/JJP+4hN/JK8F/aq/5LLqOf+faDn/tmD16/56ens5pisRT4ew9WNRqb5bu7u9H13PDyfB4apxPiaMqcXBc9k0mlqumxgf8AC/PiD/0NN5+a/wDxNC/H74gq2R4pvOPUIf5rXqX7Kvw98OeNNB16bW9It9Skhuo0jM4Y7V2k9M+vP1r228/Z8+H99bvC3hq1jVhjdCzow+hDV5WByfOMfhY4qni2lLZOUu9j2MxzzJMtxk8HVwSbi7NqMOyeh8/+AP2t/EOk3kUPilI9Z05iFe4jjWK4j5xkbQFbjsQP96vrfR9XtNd0u11GwnW5srmMSRSp0ZSOD/8AWr8/fi98Px8M/Ht/okMpntFCy28rgb2jdflDYAyeCpPTjOB0r6O/Y68QT6j4F1PSpWLpp12DDkcKkgztA7DcrHv97qa9PhvNsZHGzyzHS5mr2b3TW6v1R5HFOS4GWAhm+XR5Yu10tE1LZ26O55L8T/jd440f4ieI7Cy8Q3NtaWt/NDDDGqYVFYqo5X2/TPWuZ/4aC+If/Q03X/fKf/E1mfGL/kqni7/sKXH/AKMavor9nX4V+EvFXwvstQ1bQrS/vXnnVppQSSFcgDr6CvmsKsyzXMauHo4iUbcz1lLZO3T1PqsZLKsmyujiq+FjLmUVpGO7jfqjwj/hoD4h/wDQ0XX/AHyn/wATS/8ADQXxD/6Gm7/74T/4mvsT/hQ/w/8A+hVsP++T/jR/wof4f/8AQraf/wB8n/Gvpf8AVvOf+g1/+BTPlv8AWrI/+gBf+AwLc2uXv/CpJNYWfGo/2IbsTbR/rfI37sYx15xivi3/AIX58Qf+hqvfyX/Cvt3x3bR2fw38RQQxrFDHpNyiIo4VRCwAA+lfnFXPxficThalCnSqOPuu9m1d6b2OngjC4TF08TUrUoy95Wuk7KzPQP8AhfnxB/6Gq9/Jf8KsWP7RXxDsJ1kTxLNMM52zxRurexyv8sV9ZaP8DfAU2lWMsnheweRoEZmKnJO0Z71x3xg/Zx8K3Hg3UtR0KwXSNUsLd7lDbs2yYIu4qyknqAcEYOSOo4rOrkWeUaTrxxTdleylK/y8zSjxHw/XrKhPBpJu13CFvmT/AAN/aOi+Il4mh65BFp+tsuYJISRDdYBJABJKsAM4yc4JGOlXf2oPGmt+CPBGm3eh376dcS36xSSRqpJTy3OOQccgH8K+LtJ1SfQ9WtNRtH8u5tZo5o2/2lO4H0PPrzX6L654Z0X4gaPax6xp0OoWmVuYo5x91tpwwx3wTXp5LmGLzvLq+GlO1WNkpbb7Xt10ep5OfZZguH80w+LVO9GV24b7b79NU18z4f8A+GgfiF/0NN3/AN8p/wDE0n/DQPxD/wChqvP++U/+Jr7C/wCFA/D8/wDMq2P47v8A4qviD4mabbaN8Q/EljZQrb2ttqE8MUKdFVZCAB+Ar5XNsHm+T041K2KbUnbSUv1sfY5LjsmzypOlRwii4q+sI/pc3f8AhoH4h/8AQ03f/fKf/E10Hw8+OXjrVPH3hqxu/EVzcWt1qVvBPG6Jho2lVSPu+hI/HPWvVf2dfhP4R8WfC+x1HV9Btr++eeZWmm3FiA5A6H0r1TTfgr4I0fULa/s/Ddnb3lvKssMqq2UcHIYc9eBXtZfk2cV40cU8U+SVpW5pbbnz+Z57keHlWwawfvx5o35YWvt6mx438aab4A8OXWtarN5drbjAVeXkY5wijux/xPQV8heNP2qPGPiS8kGlXCeH7DOEhtVVpSpzy0jAnOMfd2/Q13H7aWqXHn+F9NBZbXbNcMo6M+VVc/Qbun9488155+zV8PdI+IPji5i1pPtNpZWxuRabyFmYOqjPOcAtkgEZOM++ueZhjsZmiyrBz5Fpd7XbV9XvZLojLh/LcvwOUvOcdD2j10avZJ22el2+r28jlk+MnjqOQSjxbq+7k83TFeuRwePzH516R4C/a28R6LdRw+JkTXdPPDyrGsdyg9QVAVunQjt94V9K3nwa8C31mbaTwjpKxkY3Q2iRSD6OoDD8DXxz8evhhB8LfG32GyleXTbuEXVt5nLIMkGMnvgr6dCvvXnY3BZvw/FYyGIco3Serf3p6W/rQ9TL8fkfEs3gp4VQk02tF07Nap/1qfc3h7xFp/ivRbTVtMuVurG6TfFKvfsQR2IOQR2IIrXr5i/Yt8QXE1l4j0WV2a3t2iuYVycKWDK+P++U/EH1r6dr9QyvHLMsHTxSVub807P8j8lzjL/7Kx1XB3uovR+TSa/BhRRRXqnjBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQB4P+2Rpst/8JYZY/u2upwzyc4AUpJH/ADcdeOa/NbxBqn9rag7qf3CfJCP9kdz7nr+Nfqz+0P4bk8WfBHxpp0Ks07abLNEiLuLPGPNVQO5JQD8a/JSv3ngCqqmAqU3vCX4Nf53Pa4dyjD/X6uZtXqNKPp/w4VoeHtVOg6/pupIMtZ3UVwBjOSjhsfpWfRX6fOKnFxezP0eS5k0+p+1dpdQ3ttFcQSLJDKgdHU8MpGQasV81fsR/GaHx58NofDF7cKdd8OxrAI2OGltBxE4HcKP3Z9Nqk/eFfSm4etfyPj8FUy/FVMLVWsXb/J/Nan5NiKMsPVlSnuhaKTNG4eorzznPgj9uqTz/AIxaenaLRYV+mZpj/WvEfCvg288VXgjgXyrYHEtwwyqfT1PtX0F+1J4e/wCEm+OVxJJKv2S2s7eI+Ww3bsFivt97r16Y7kZekQW+n28cFtEsUSDCqo4FfMZ1xS8souhhXep+R/VGTYhYfI8LTp/E4L5X1N3wnodl4W06GxsYtka/M0mMs57lj6/54GK9m+CZ87xXJ/s2rt/48o/rXjlpcdOa9e+AJ8zxJfN122bDqe7p/hX4dk0auN4hoYjESvLmbb9NT4fiHmjga05btfmeifGr/kk/iz/sHS/+g1+eUMnlzJJjO1g1foZ8av8Akk/iv/sHS/8AoNfnnDCZpo487SxA3V9txtf63Rt/L+pXh/b6lXvtzfofUv8Aw2xb9f8AhEJf/BiP/jdavhP9rq28UeKNK0Y+GJbQ391Ha+d9tD7C7BQ2PLGcE+orD/4YjY4/4rIZ7/8AEqz/AO1v8+la/g/9kP8A4RXxRpGsHxU10un3Udz9nXT/AC/M2MCBnzSB0x0NenQfFntY+1Xu3V/4e3XY8bELg32M/Yt89nb+Jv030Po6vm/9tP8A5Ffw5/1+Sf8AoFfSFfN/7af/ACLPhv8A6/JP/QK+q4k/5FNe3b9UfIcLf8jnD+v6M8E+C/gu1+IXjZdCuyUS5tbjy5F/5ZSBCUf3wcHHGe9c7eWureAfFEsDGTT9Y0y5xujb5kdT94H04BB/oa9G/ZT/AOSzaZ/173H/AKKNe3fHz9nm6+JmrWes6FLaWeqBfJuvtbMiTIPusCoJ3L09xjnjFfluDyWePylYrCq9WE3to2rL8Vufr+Oz6nlucvCYyX7mcFvqk7vp2a0fyPlTwN4Pv/iN4wstGtMtPeyFpZmGQiDJaRvXHOc9SMdasfFLQbTwv8QNb0mwQxWllcGGPccnAUc59c8/U19d/AL4Hv8ACiwvbnU5ILvXbw7GltyzJHCMYVWYAkk8scdlH8Oa+Vfjp/yV/wAVEdftrY/Ifz6eoPORUZhk7yzKoVa6/ezlr5KzsvXqy8szxZtnFShh3ejCGnm7q7/Rf8E+nv2RP+SS/wDcQm/kleC/tXf8lk1D/r2t/wD0WK97/ZD/AOSR/wDcQn/kleC/tWc/GTUf+ve3/wDRYr6HOP8AkmcM/wDD+TPmsk/5KvF/9v8A/pUTp/2Yvip4W+Huh63B4h1T+z5bi5SSIG3lk3KFIJyinH6dK9kvP2oPhxa27SR65JdOoyIYbKYO3sCyAfmRXyf8Ofgz4i+KFpd3OiJa+XaSLHJ583lkMQTxgd+M9OgrsU/ZD8eMwB/sxB/eN2cfkErmyzMs9o4OnTwmHUoJaOz7/wCJL8DpzbKuHa+PqVcXinCo3qrq2y/ut/icD8VvH8nxM8bX+tvC1rDJiK3t2OSkSqMZ7ZPJPXr7CvqD9kXwpcaH8PbvU7lGibVrnzYlYYJiRQqt+J3Y9Rg9653wB+x3Fp95FeeLNRj1BYyGFhYhhG5HPzyHBI7YAH17V9I29tHZwxwQxLFDGoRI41wqqBgADsPavb4fyXGU8XPMswVpu9l1u93p+CPA4lz7BVMFDKst1hG130stkr792/zufnn8Yv8Akq3i7/sKXH/oxq9F+E/7TEXwz8F2+hP4efUDDLI/npdiPO9iwG3YcdfWvOvjD/yVXxb/ANhS4/8AQ2r0D4U/syn4neDbfXv+Ej/s3zpZI/s/2HzcbWwTu80dTz0/OvhsD/aP9pVv7L+O8r/Dtzf3tD9DzBZX/ZND+1/4do2+Lfl0+HXa53H/AA2zbf8AQoS/+DAf/G6998F+JE8YeFdK1pIDbJfQLOIWYNsyOmcDP1wK+ev+GIT/ANDn/wCUv/7dX0N4J8M/8Ib4T0rRBMboWNusHnlNpfA64ycfma/T8k/tv2s/7U+G2nw7/wDbv6n5FxB/YCpQWUfHfX49rf3vMj+I3/JPvE//AGC7r/0U1fm5X6SfEX/kn3if/sF3X/opq/NuvkOOL+3oej/M+38Pv92xHqvyZ9i6X+154KtNNs4JLPWi8UKoxW2jxwAD/wAtK5L4q/tXaf4i8L32jeG9PvI3vomgmur5UTZGwIcKqs2SRkckYzxVP/hjfUZ9FF7beJYZ7h7fzo7drNl3sVyq7zIcc8bsV87SRvDI6SIyshKsrDBBHUH3Fc2ZZxn2FpKlikoKasmkr/em7fmdWVZHw5jK7q4OTqOm02m3a/mmldaeht+BfCN3448VabolmjNJdShGZRxEmcs59lUE++B61+kcUK28SRoMIgCqPYcV5v8ABHwH4S8NeFLLVfDMDSf2lbpI19csHncED5CQAFAOQVUAZHtXptfdcN5R/ZeHcpyUpTs9NrdPzZ+fcVZ4s4xUYQg4wp3ST3v1v22SsFfnT8YP+SreLv8AsKXH/obV+i1fnT8Yf+SqeLv+wpcf+htXiccf7rR/xP8AI97w+/3yt/hX5n1r+yr/AMka03/r4n/9GGvYa8d/ZV/5Izp3/XxP/wCjDXsVfZ5R/wAi7D/4I/kj4TOv+Rnif8cvzZ4x+0t8J7r4jeFra80uEz6xpRd44R1miYDegz1b5VI57Ed6+NtB8Qav4J1yPUNNuZtM1K1ONwGCuOCrKw5GMgqR/UV+lU15Bb4Es0cW7pvcDNcB43+Dvg34mb5r6xjF9jH9oWDiOYfVhw3T+IHHbFfM55w7LH1vrmDny1V52vbZ3Wqdj6vh7iiOW4d4HG0+ei/K9r7q2zT/AKueOeDP2ymUJb+KdG3kYDXemHB/GNj1+jD2Fe0aD4q8AfFwRyWzaXrd1GnEF5ApnjXqRscbsZ7jivAPGn7Het6YrzeG9Rh1iBeRaXIEE30B+4x+pWvCruz1XwjrLQTx3Ok6naPnacxyxsOQR0x6g5HXOcV8+86zfKGqeZ0ueHdr8pLR/PU+nWQ5HnadXKK3s6nZP84uzXy0P0e0nw3pGg+b/Zml2enGThza26Rb8dM7QM961q8Y/Zu+LVz8SvDdzaao2/WdLKLLNj/XxsDtc/7WVYH6A969nr9PwOIo4vDwr4f4Jbf16n5HmGFr4LFTw+J+OLs/0/AKKKK7jzwooooAKKKKACiiigAooooAKKKKACiiigAooooAjZN6kFQQRgg1+S37Qnwyf4S/FrWtBVGWw8w3Vgx/it5CWXB9FxsPvGeK/W2vnf8AbH+BMnxX8ApqukW/m+JdD3TQoo+a5gPMkPHJbgMvXkED71fb8I5vHK8fy1XanU91+XZ/L9T2spxawuItP4ZaP9D81KKO+O/p+lBO2v6UP0i6tc6HwB461f4beLLDxDoty1tf2j5BU8Oh4ZGGDkEdQePyFfoJ8O/20fCPiTTbX/hIUn0K8dATMsLTW0nTlSuWU5z8pB29Nxr81metbw74h/s2Ywzsfskhye/lt0yB3GOCPTHXFfn3F3D8s2w/1jDL97D/AMmXb/I51hsrxlWNPHppP7UXZr81b1TP1Suv2kvhxawiQ+JoZQRkLDDK7H2wE4/GvJ/H37XDX9tJZ+E7KS0DgqdQvQDIP9yPkdupJ69K+SLS5G1WDAow3KynII9QfStq1uOnNfynmFfFU+albla37n3+C4ByfCtYi8qvbmat9ySv87o6tb6W8uZJ55JJp5G3u8hLMzE5ySScnJJPJyTWtaz9skVy1rcdOa2LWfpX5jjKDbbZ9RWoKKslZHVWlxjbzXuH7OP7zVNXk/u26L+bZ/pXgWjx3GoXkNrbRtcXMzKkcaDLMx7CvsD4X/D9PAegrHJ+81O4CvdSj1HRB7DJ/HJ44AvhnLKlTMo4mK92ne782rWPyzi/E0sPgnQk/ensvR6v0D41/wDJJ/Fn/YOl/wDQa/Pa1bbdQsTgB1JJ+tfoj8WtPudU+Gfie0s7eS6uptPmWKGJSzu204AA5JPoK+Ef+FV+Nf8AoT9e/wDBZP8A/E16XGVCtVxVGVKDdo9E318jPgTEUKWDrxqzUW5dWl08z76/4WR4S/6GnRf/AAYQ/wDxVH/CyfCP/Q06L/4MIf8A4qvgX/hVXjX/AKE/Xj/3DJ//AImj/hVfjb/oT9e/8Fk//wATXR/rZmH/AEBv8f8AI5P9TMt/6DV/5L/mfoFp/jbw7q94lpYa/pd7dPnZBb3kcjtgZOFDZPAJ/CvCf20/+RZ8N/8AX5J/6BXmvwB+HnirSfi54fvL/wAN6vY2cLymSe6sZYo0/cuMlmUDkkD8a9d/a48L6x4m8N6AmkaXe6o8N45eOyt3mZQU4JCg4HHX6V6WIxuIzfI8ROpRcJbJa67ank4bAYbJuIcNTp1lOO7elk/eVjxT9lbH/C5NN/64XHT/AK5nHb09P0r7pr41/Zp8A+JtD+K9heal4d1XTrWOCcPcXlnJEnKEAZZQMk4/z0+yq6uD6U6OXONSLT5nvp0Rzcb1qdbNFKnJSXItnfq+wV+enx0/5K94rA/5/WU/kMj6Y7YNfoXXwt8aPhx4s1L4peJbqz8Maze2k10zRTW9jLIjqQpyCqkEZ9/yIIrn4zpVK2CpqnFt8y2TfR9jo4FrU6OYVHVkkuTq0uq7nvn7IZx8I+f+ghP/ACSvBf2rP+Sx6ie32a3H/kMV9Efsu6DqXh34Xra6pp91pt0b2Z/Iu4WicKdoB2sAe1eK/tL+AfE2ufFe9vdN8O6pqNpJbwBbizspJU4QBhuVTyOa8zNqFWXDmHpxi21y6W12fQ9bJsRRhxRiqkppRfPrdW3XU7j9iv8A5FrxL/19xf8AoBr6RrwD9kfwvrHhnw94gXV9KvdLea6jMaXtu8LMAnJAYDI5/Svf6+w4ehKnldCM1Z22fqz4jiapCrm+InBpptarbZBRRRX0R8wfnV8YP+SreLf+wncH/wAiGvpv9mXxn4f0X4S6fa6hrum2NytxOWhubuONwC5IJDEHmvBvit8NfF1/8SvE1za+FtZurabUZ5YpoNPldGVmJ3BguDn26g1yv/Cq/G3/AEJ+vgf9gyf/AOJr8HwuJxmU5lWxFPDuV+ZbPvfsf0VjMLgc6yqhhqmIjCyi903pG1tz76/4WR4S/wChp0X/AMGEP/xVH/CyPCX/AENOi/8Agwh/+Kr4F/4VX41/6E/Xv/BZP/8AE0f8Kr8a/wDQna9/4LJ//ia+m/1szD/oDf4/5Hyn+pmWf9By/wDJf8z74+IMyTfDvxJIjq6NpVyyspyCDC2CDX5vV+h82lXjfBuTTvs0hvzoBg+z7Tv8z7Pt249c8Yr4Z/4VX41/6E/Xv/BZP/8AE1zcY0quIq4eUIN6PZN227HTwPWoYaliYVKiXvK12le1/M/Q3Qf+QHp3/XtH/wCgiviv9qP4f/8ACH/EaTUYItlhrIN1GccCXOJVB9dxDf8AA8V9raPC8GkWMUilZEgRWU9iFGa89/aD+HcnxD+Hd3BZwNNqtkwu7RVHzOy8MnvuQsMdztr7LP8ALv7Qy6UIr346r1XT5o+G4bzT+zM0jUm/cn7svR9fk/wueY/sefELzra/8H3cnzw7ryy3HqpOJUH0JDAf7TelfTua/P8A8E+FfH/gjxVpet2nhDX/ADrKZX2f2bPhl6Mn3eAVLKfY9etffVrN9qt4ptjx+YobZIpVlyM4IPQ1x8K4utWwXsMRFqVPTVNXXTftt6WO3jHCUKOP+s4aScamujTtJb7d9/vJ6/On4wf8lW8Xf9hS4/8AQ2r9Fq+DPir8NfF2ofErxPdW3hbWrq3m1CeSOaGwleN0ZyQVYLg8V5/GlGpWw1FU4t+90TfTyPS4Dr0qGLrOrJK8erS6+Z9I/sq/8kZ07/r4n/8ARhr2KvK/2a9D1HQfhPp1pqdjcafdiaZjb3UbRyKDIcZUgEcf416pX2GUxccBQjJWfLH8j4nOZKWZYiUXdOcvzZ8j/toaTOviLw7qe0m2ktHt92OjI+459Mh/0PpXKfs0/FLTfhz4ovYdYc2+napGsZusEiKRTlS2P4fmIz15XtnH1x8Rfh7p3xK8M3GjakrKrESQ3CAb4JBnDrn6kY9Ca+NvG37N/jbwfdSeVpkut2Wfku9NUylhnvGAWBwemMeh71+d51gMdl+af2rhI8yeumtnazTXZ9+h+m5BmOX5nlDybHT5GtNWldXumm9Lrt5H23b+MtAvLQXUOuadLbYz5yXcZX884r42/ak8aaJ4x8fWp0WaG8SztRbzXkOCskm9iAGH3gobg9Mk49a8y/4Q/X/N8v8AsTUTJ08v7LJn6dK7jwX+zl438YXSB9Jl0Wzz891qSGHAyOQn3mPBxgY9eua48wzjMM8oLB0sK1dq+729UkvmehluR5Zw9iPr9XFqVk7bLf0bv6I9F/Yt024bWvEuoYK2yQRwHjhnZy36BT+dfWVcn8N/h9p3wz8L2+i6arOqnzJp3GGmkONznHToAB0AAA6V1lfpOS4GeXYCnh6jvJav1buz8pz7MIZpmNXFU1aLsl6JW/G1wooor3DwAooooAKKKKACiiigAooooAKKKKACiiigAooooAKT8KWigD4j/a6/ZHuLm4vvHHgizMxkJm1TRoVyS3UzQqOuerL1yMjrgfELtjvX7cFc9RmvnL47/sV+F/ixLcavorr4W8TSZdp4IgbW5b1liGME93Ug8kkMa/WeGuMlg4RweY3cFtLdryfdee68z38LmtSlFU6jul1PzPaT+n69KiaSvXviL+yb8T/hxJK114bn1ayTdi+0bN3ER64A3oCODuUenSvGpt8MjRyK0cikhkYYII6giv3DB4zC46HPhaimvJr+l8xV8w5tUzoPD3i6XRnWKdTNZ5yRn5k9Svr/ALp49McmvTtK1S31C3Sa3lW4gJ2hlPfjjHY8jg+teFM2cnPA610/gfwn408SXZk8I6JrGqyZCu2m2ckye4cqCNvQ4bjivgOK+C8DnEHiVNUqnd2SfqfQZHx7iMmfsKy9pS7X1Xp/ke2Wtx059q6Pw/Y3muahBY6fbS3t5M2yOGFcsT/QAdTXW/Bv9k74j+IfLn8cRWXhmwC4Cq4lvZOOpRCUGe+WBH92vr3wH8LPD/w3sxFotiEuGXbLeTfPPL04LdhwPlGB7V/KWP4blRxLoSqxkl1i7p+h+hY7xBy+WG58JCTqPo1ZL1f+X4HL/Bf4KxeArVdS1RUuNflX/eS2BH3V9WxwW9OBxkn1qiivaw+HpYWmqVJWSPw3G42vmFeWIxEryf8AVl5BRRSZrpOEWiiigAooooAKTcPWivM/j38RtQ+G/guG50WKG41/Ub6DTtOgmXcryu2SCMj+FW5zwSM1pTpyrTVOG7Mq1WNGm6k9kembh60ZHrXhOjfGbVfE37OOreLDcW2j+I7COaC5f7OXjt50fH+ryx5UqcHPLdCOK3tL+O2h6Xpuk22vXkzaq/hqPxBLcR23lxTxCPc7IM/eJDfJ+FbywtWLatdptfccscdQklLmsmk035nrGaM15Zrn7RfhDw/4f0HVrh75hrcLXNlZQ2he5eIfekKDouOc55B471k+Lvjlm5+GF34UuLW90bxVqos5ZZom3eXuVWCjI2sCWByO35qOFrSteNt/wvf8mVLGUI395O1tvO1vzR7TketFeY3n7QXhSx0rUdQlkvBb2Gsf2HNi35+089BnleD836VlaP8AHwah8dNU8BvpdwlrbpshvFt33GYDLF+wjODhsYPHPNCwtezfK9Nfy/zQPGYe6XOtXb77/wCTPY8j1ozXhHw/+PVtpnwrm8S+Mtcj1Fjq81hA9jYtG8rAjbEsYxlsZOenbJ6npf8AhozwafAN34uW4um02zuRZ3MP2ci4glJA2sh6deucdfSnLCV4trlb1t8xQxtCSTckna9m1ex6lmjNcJ8O/jH4a+KF1qdros1x9r08r58F3btC+xuVcBhyp/PkZAyK8p+IHxr8W6b8ZNa8JaZrvhTw9YWNvBLFP4iDr5jOiEqGB5OWPGBxTp4OtUnKnazSu79tP8yamPoU6cat7xk7K2uuv+R9I5HrRuHrXlPiT42ad8KdP0fT/G90LrxPPbG4uIdEtnkQKpO6XBwVjGDyT2PFS+Lv2ivBXgzStA1O7vri5sNcheeyms4DIHVQuQRwQSWUY7HrjFSsLXlblg3fbTf0NHjKEb800mt7va/c9R3D1oyPWvLfF37Rng7wTLp8Woz3pnu7SO+eG3tHka1gfGHmA+516dfbkZb42/aM8G+BL/TbS+uLq5k1Kyj1CzNlbmZZonYqhU8cnB60o4WvLltB67abg8Zh4816i93fXa56puHrRuHrXm+r/HjwvosPjKS5e82+E5LeLUtsGcGd9ibOfm569MVmeK/2lvBvg3VZdOvzqJuo7WK8YW9k0gEThSGJHTAYZzjGcdaI4WvLSMH93o/1X3hLGYeCvKol8/Vfmn9zPW8j1oyPWvKfFn7SHg3wfNp0d1LfXZ1LTV1a0NlatKJIG3EHsQcKx5xgDnFW9Y/aA8HaP4S0TxE13cXdrrWRYW9pbtJcTspw6hOvynIOePrkULDV5WtB67aDeMw65rzXu767f1f8T0vIo3D1ryvVf2kfA+keDdI8Uy388uj6nM1tFJDbszI653CReq4wf6ZyK7zwt4ktfGHh2w1qyWZLS+iWaJZ02PtPTK54qJ0atOPNOLS2+a3Lp4ilVlywkm99Oz2+82aKTNFYnQLSUtFADMe1LTqKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAZtJrK1jwloniD/kKaNYanxt/0y1Sbj/gQNbFFOMnF3i7MDmbP4a+EdPlWW08K6Layr914dOhRh+IWui27QAq8ewqSiqlOU9ZO4BRRRUAFFFFABXxXq37IvxrvtUvbiD4qmGCWZ5I4/7UvRtUsSBgDA4x0r7Uor0MJjq2CbdK2vdJ/mcWJwlPFpKpfTs2vyPiD/hjv44f9Fab/wAGt9/hR/wx18b/APorTf8Ag1vf/ia+36K9P+38b/d/8Aj/AJHF/ZGF/vf+BP8AzPiL/hjv43/9Faf/AMG99/hTf+GOvjh/0Vpv/Brff4V9v0Uf2/jf7v8A4BH/ACD+yML/AHv/AAJ/5nmX7P8A8PfE3wz8BHR/FevHxHqn2uScXnnSzfu2C7U3SfNwQePesn4w/B/xB8UPGvhK6tdai0TRNEMl15seXuhcnGxkRlKELtXljxk4GQK9jorxvrNT2zrr4nfp38jvlhac6KoSvyq3XtrufOWl/s7+KtH8KfEzw4utWWo2PiVhcWVxdM8cq3DcytKqptXce6Z+4vAzw34r/s063468GeAbLTdQsbHWtB09NMvbiSSRY5ITEiOFIQlhlWwpC5DnJFfR9Fbxx9eNRVU9Vrt5W/I5XluHlTdJr3Wrb+d/zPFPiB8GvEC+JvCfiPwFd6ZaajoNg2lLa6wrm3a3KlQRsBO5Qx9jx6YOFpP7NereH9B+GGnWmo2c7eGtXbU9RkkZ4xLukVysShT0xjB2g4zxnj6IoqI4ytGCgnt/wf8AN/eXLL6Epuo1q/8Agf8AyK+4+V/E37M/jy+vPEGnadqnh/8A4RvUfEf/AAkK/aPOW63knKkqhUYBx0OcDkcivUV+GniLTvj9P42sbvTm0PULBbK+tpxJ9oXavymPA2n5lQksehYY6GvV6KqeOrVFaVtrbd7f5Imnl1Cm7xvunv2v+GrPmix/Zn8T6d8PdIs4NV0uPxRoviJtdsZG8yS0fJXCSHYGH3QeB298hmrfs0+Kdb+Gvi+yu9S0pvFfibV49UuWjMqWUQVyQqnYzZ5bt6DJxk/TVFV/aGI3v1vt53+6+pH9l4a1rPa277W++3U8w8K/DPVND+NnjHxhPNatpus2trBbxRu5mVo40VtwKhQCV4wxPPbpVC3+CYvPjd4i8X63ZaTq2kX1nBDaW9zF50sUqKilyrJtXhTyCTg9ua9eorm+sVLuSerSXyVv8jr+qUmkmrpNy+bv/mzxD4sfB/xZrXjz/hK/Bt/pUF7daRJot7b6wshj8lmJ8yMoD83PQ8fKOuSKqaT+zvqPh/UPhL9lvrO5svCAu2vWn3rJM853ZiGCMByeCRgY5r3qitVjaygoX0Wm3k1+TZl/Z9BzlUtq3ffrdP8ANI+dfjJ+zlrXjD4g3PifQW0e6XUbJbO8s9aluYlQgACRGgILfKoG1uOvBzx0ej/BK+0X4qeD/ENs2nxaNonh/wDsl7aN5S4lzIcxh93yfvMDc5IHHNez0UfXazgqbeiTXyen5CWX4dVHUS1bT+ad/wAz5l+JX7OvjjxBrXxAXQNU0OPQ/FzWc1xHqHnLOrwMGCgopUDOecHIwCK3NQ+AviC78ReMr9LrTRFrPhNdBtw0km5ZxGib3GzGzKk5GT7V79RV/X69krrTy9P/AJFEf2bh+Zys9fP/ABf/ACTPANF+AfiDTvEnhTUJLrTTDpPhE6BOFeTc1xtkAZR5eDH8/UkHr8tc5efss+IP+Fb+ArCC80mTxF4Zku/MjmluFs7mOeZpCPMj2SKRlegGSSOgFfUVFOOYYiL5k/61/wDkmKWV4aUeVx02/wDSf/kUfO8P7OerQ+HfAVnCui2VxpHiJNc1OO2kuGhcBlysXmb2ZtqqPmKjj6k/QgXAAxUlFctavOvbne1/xdzroYWlh7+zVr2/BWR8x/Hj9nf4m/Eb4h3GteF/Hp8PaRJBFEtl9uuosMq4Y7Y/l5/X+fnn/DHfxw/6Kyf/AAa3v/xNfb9FerRzrF0KcaUOWy0+GL/Gxx1Mrw9Wo6kr3f8Aef8AmfEH/DHPxu/6Kwf/AAa33+FB/Y5+N/8A0Vk/+DW+/wAK+36K2/t/G/3f/AI/5Ef2Rhf73/gT/wAz4f8A+GOfjf8A9FZP/g1vv8K9Z/Zx+BXxC+FfijU9Q8X+Mz4msrmz8iK3N7cT+XJvQ78S8DgMMj17dK+h6K5sRm+JxNJ0anLZ9oxT+9I1o5bQoVFUhe6/vN/qFFFFeKeqFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAf//Z"""
        return base64.b64decode(image_data)

    def upload_gps_excel(self, *_):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GPS Excel File",
            "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.gps_excel_input.setText(file_path)
    @profile
    def setup_tables(self):
        headers = {
            'stagger': ['Time', 'Stagger Distance 1','Stagger Distance 2' ,'Alert', 'Mast Name', 'Latitude', 'Longitude'],
            'height': ['Time', 'Height Distance', 'Alert', 'Mast Name', 'Latitude', 'Longitude'],
            'gradient': ['Time', 'Gradient Measurement', 'Relative Gradient Measurement', 'Alert', 'Mast Name', 'Latitude', 'Longitude'],
            'setting_distance': ['Time', 'Setting Distance Measurement', 'Alert', 'Mast Name', 'Latitude', 'Longitude'],
            'double_contact': ['Time', 'Double Contact Measurement', 'Alert', 'Mast Name', 'Latitude', 'Longitude']
        }

        tables = {
            'stagger': self.stagger_table,
            'height': self.height_table,
            'gradient': self.gradient_table,
            'setting_distance': self.setting_table,
            'double_contact': self.double_contact_table
        }

        for table_name, table in tables.items():
            table.setColumnCount(len(headers[table_name]))
            table.setHorizontalHeaderLabels(headers[table_name])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    def start_video_processing(self, *_):
        try:
            conf_threshold = float(self.conf_threshold_input.text())
            selected_cam_index = self.get_selected_camera_index()
            selected_panto_width = self.get_panto_width()
            selected_gps_port = self.get_gps_port()
        except ValueError:
            conf_threshold = 0.15

        self.thread = VideoThread(
            conf_threshold,
            float(self.train_height_spinbox.text()),
            float(selected_panto_width),
            float(self.camera_offset_spinbox.text()),
            float(self.roof_to_frame_spinbox.text()),
            self.gps_excel_input.text(),
            selected_gps_port,
            selected_cam_index,

            "Left" if self.mast_direction_layout_radio_left.isChecked() else "Right",
            (1 if self.height_correction_radio_plus.isChecked() else -1) * self.height_correction_input.value(),
            (1 if self.setting_distance_correction_radio_plus.isChecked() else -1) * self.setting_distance_correction_input.value(),
            (1 if self.stagger_distance_correction_radio_plus.isChecked() else -1) * self.stagger_distance_correction_input.value()
        )
        logger.info(f"\nPanto width : {selected_panto_width} | Offset : {self.camera_offset_spinbox.text()} | Roof to frame : {self.roof_to_frame_spinbox.text()}")
        logger.info(f"height Correction : {self.height_correction_input.value()} | Setting distance correction : {self.setting_distance_correction_input.value()} | Stagger correction : {self.stagger_distance_correction_input.value()}\n")
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_zed_pixmap_signal.connect(self.update_zed_image)
        self.thread.update_logs_signal.connect(self.update_logs)
        self.thread.update_camera_info_signal.connect(self.update_camera_info)
        self.thread.update_gps_info_signal.connect(self.update_gps_info)
        self.thread.finished_signal.connect(self.video_processing_finished)
        self.thread.alert_signal.connect(self.handle_alert)
        self.thread.start()


        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    def stop_video_processing(self,*_):
        if hasattr(self, 'thread'):
            self.thread.running = False
            self.thread.stop()
            # self.thread.wait()
        self.video_processing_finished()
    @profile
    def video_processing_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    @profile
    def update_image(self, webcam_rgb_frame):
        qt_img = self.convert_cv_qt(webcam_rgb_frame)
        self.video_label.setPixmap(qt_img)
    @profile
    def update_zed_image(self, zed_rgb_frame):
        qt_img = self.convert_cv_qt(zed_rgb_frame)
        self.zed_video_label.setPixmap(qt_img)
    @profile
    def convert_cv_qt(self, cv_img):
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:  # Color image
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif len(cv_img.shape) == 2:  # Grayscale image
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

        h, w, ch = cv_img.shape
        bytes_per_line = ch * w

        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # Use fixed dimensions that won't trigger fullscreen
        p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)
    @profile
    def update_logs(self, logs):
        # if 'setting_distance' in logs and logs['setting_distance']:
            for log_type, data in logs.items():
                if data:
                    if log_type == 'stagger':
                        self.update_table(self.stagger_table, data)
                    elif log_type == 'height':
                        self.update_table(self.height_table, data)
                    elif log_type == 'setting_distance':
                        self.update_table(self.setting_table, data)
                    elif log_type == 'gradient':
                        self.update_table(self.gradient_table, data)
                    elif log_type == 'double_contact':
                        self.update_table(self.double_contact_table, data)
    @profile
    def update_table(self, table, data):
        row = table.rowCount()
        table.insertRow(row)
        alert_column = 2
        is_alert = data[alert_column] == "Yes"

        for col, value in enumerate(data):
            table_item = QTableWidgetItem(str(value))
            if is_alert:
                table_item.setBackground(QColor(252, 128, 3))
                table_item.setForeground(QColor(255, 255, 255))
            table.setItem(row, col, table_item)

        table.scrollToBottom()
    @profile
    def update_camera_info(self, info):
        self.camera_info_label.setText(info)

    @profile
    def update_gps_info(self, info):
        self.gps_info_label.setText(info)

    @profile
    def handle_alert(self, alert_status):
        if alert_status:
            self.alert_label.setText("⚠️ ALERT: Object detected outside safe range (2100mm - 6000mm)")
            self.alert_label.setStyleSheet("QLabel { color: red; font-size: 16px; font-weight: bold; }")
        else:
            self.alert_label.setText("✓ Object within safe range")
            self.alert_label.setStyleSheet("QLabel { color: green; font-size: 16px; }")

def main():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
if __name__ == "__main__":
    main()

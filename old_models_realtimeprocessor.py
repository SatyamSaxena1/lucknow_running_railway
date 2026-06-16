from typing import List, Tuple, Optional
import cupy as np
import queue
import pytz
import cv2
from datetime import datetime
import serial
import pynmea2
from geopy.distance import geodesic
import pandas as pd
import os
import threading
from line_profiler import profile
import torch
cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())
import logging
from ultralytics import YOLO
from ultralytics.utils import LOGGER
os.environ["ULTRALYTICS_ENGINE"] = "torch"
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.drawing.image import Image
import io
import time, requests
LOGGER.setLevel(logging.WARNING)

# Configure the logger
logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

class RealtimeProcessor:
    @profile
    def __init__(self,  train_height: float, selected_panto_width: float, gps_excel: str,selected_gps_port: str,roof_to_frame_height:float,mast_direction:str ,height_correction: float = 0, stagger_distance_correction:float = 0 ):
        self.log_time_stamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_folder = os.path.join("pantograph_logs", self.log_time_stamp)
        os.makedirs(self.log_folder, exist_ok=True)
        self.log_files = {}
        self.height_correction = height_correction
        self.stagger_distance_correction = stagger_distance_correction
        self.roof_to_frame_height = int(roof_to_frame_height)
        self.train_height = int(train_height)
        self.pantograph_width = int(selected_panto_width)
        self.strip_color = (255, 255, 0)
        self.last_mast_detection = 0
        self.frame_number = 0
        self.mast_cooldown = 90
        self.mast_count = 0
        self.last_mast_frame = -float('inf')

        self.mast_heights = []
        self.mast_distances = []
        self.relative_gradient_window = 4
        self.gps_port = selected_gps_port
        self.gps_data = {"lat": None, "lon": None, "connected": False}
        self.gps_thread = None
        self.serial_port = selected_gps_port  # Replace with your GPS module's COM port
        self.baud_rate = 115200
        self.target_per_second = 25  # Adjust as needed
        self.mast_data = self.load_mast_data(gps_excel)
        self.setting_distance_logged = False

        self.distance_between_mast = 0.0001
        self.current_gradient = 0
        self.previous_mast_height = 0
        self.ui_current_mast_direction = mast_direction
        self.current_mast_direction = mast_direction
        self.alert_colors = {
            'height': (0, 255, 0),
            'stagger': (0, 165, 255),
            'setting_distance': (0, 165, 255),
            'double_contact': (255, 0, 255)
        }
        self.arm_model_path = os.path.join(os.getcwd(),"models","arm_medium_90.engine")
        self.arm_model = YOLO(self.arm_model_path, verbose=False, task="detect")
        self.arm_conf_threshold = 0.50

        self.current_overlap_staggers = []  # stagger
        self.was_overlapping = False

    @profile
    def convert_coordinates(self, coord):
        degrees = int(coord / 100)
        minutes = coord - (degrees * 100)
        return degrees + (minutes / 60)

    @profile
    def load_mast_data(self,xls_path):
        df = pd.read_excel(xls_path)
        df = df[['location', 'Lat', 'Lon']]

        # Convert coordinates
        df['Lat'] = df['Lat'].apply(self.convert_coordinates)
        df['Lon'] = df['Lon'].apply(self.convert_coordinates)

        df['next_lat'] = df['Lat'].shift(-1)
        df['next_lon'] = df['Lon'].shift(-1)

        def calculate_distance(row):
            try:
                if pd.notnull(row['next_lat']):
                    return geodesic((row['Lat'], row['Lon']), (row['next_lat'], row['next_lon'])).meters
            except ValueError:
                logger.error(f"Invalid coordinates for row: {row['location']}")
            return None

        df['distance'] = df.apply(calculate_distance, axis=1)
        return df.to_dict('records')

    @profile
    def load_mast_data_2(self, file_path):
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.xlsb'):
            df = pd.read_excel(file_path, engine='pyxlsb')
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a file with one of the following extensions: .xlsx, .xls, .xlsb, .csv")

        df = df[['location', 'Lat', 'Lon']]

        # Convert coordinates
        df['Lat'] = df['Lat'].apply(self.convert_coordinates)
        df['Lon'] = df['Lon'].apply(self.convert_coordinates)

        df['next_lat'] = df['Lat'].shift(-1)
        df['next_lon'] = df['Lon'].shift(-1)

        def calculate_distance(row):
            try:
                if pd.notnull(row['next_lat']):
                    return geodesic((row['Lat'], row['Lon']), (row['next_lat'], row['next_lon'])).meters
            except ValueError:
                logger.error(f"Invalid coordinates for row: {row['location']}")
            return None

        df['distance'] = df.apply(calculate_distance, axis=1)
        return df.to_dict('records')

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Compute the haversine distance between two sets of coordinates.
        Inputs should be in radians.
        """
        R = 6371000  # Earth radius in meters

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    @profile
    def find_nearest_mast(self, lat, lon):
        if lat is None or lon is None:
            return None

        lat = np.radians(float(lat))
        lon = np.radians(float(lon))
        mast_lats = np.radians(np.array([mast['Lat'] for mast in self.mast_data]))
        mast_lons = np.radians(np.array([mast['Lon'] for mast in self.mast_data]))

        # Compute distances in a vectorized manner
        distances = self.haversine_distance(lat, lon, mast_lats, mast_lons)

        # Find the index of the nearest mast
        nearest_idx = np.argmin(distances)
        return self.mast_data[int(nearest_idx)]
    @profile
    def calculate_gradient(self, prev_height: float, curr_height: float, distance: float) -> float:
        if distance == 0:
            return 0

        # Ensure we're working with CPU values
        if torch.is_tensor(prev_height):
            prev_height = prev_height.cpu().numpy()
        if torch.is_tensor(curr_height):
            curr_height = curr_height.cpu().numpy()
        if torch.is_tensor(distance):
            distance = distance.cpu().numpy()

        # Convert to float values
        height_diff_mm = float(curr_height - prev_height)
        # logger.debug(F"Height diff : {height_diff_mm} | Distance : {distance}")
        gradient_mm_per_m = height_diff_mm / float(distance)
        # logger.debug(f"Gradient per mm/M : {gradient_mm_per_m}")
        gradient_mm_per_50m = gradient_mm_per_m * 50
        return gradient_mm_per_m
    @profile
    def calculate_relative_gradient(self, heights: List[float], distances: List[float]) -> float:
        if len(heights) < 2 or len(distances) < 1:
            return 0

        # Convert GPU tensors to CPU numpy arrays if needed
        heights_cpu = [h.cpu().numpy() if torch.is_tensor(h) else h for h in heights]
        distances_cpu = [d.cpu().numpy() if torch.is_tensor(d) else d for d in distances]

        # Convert lists to numpy arrays
        heights_array = np.array(heights_cpu)
        distances_array = np.array(distances_cpu)

        total_height_change = float(heights_array[-1] - heights_array[0])
        total_distance = float(np.sum(distances_array))

        if total_distance == 0:
            return 0

        relative_gradient_mm_per_m = total_height_change / total_distance
        relative_gradient_mm_per_50m = relative_gradient_mm_per_m * 50
        return relative_gradient_mm_per_m

    @profile
    def parse_nmea_sentence(self, sentence):
        """Parse NMEA sentences and extract GNRMC data."""
        if sentence.startswith('$GNRMC'):
            parts = sentence.split(',')
            return {
                'type': 'GNRMC',
                'time': parts[1],
                'latitude': float(parts[3]) if parts[3] else 0.0,
                'lat_dir': parts[4],
                'longitude': float(parts[5]) if parts[5] else 0.0,
                'lon_dir': parts[6],
                'speed': float(parts[7]) if parts[7] else 0.0,
                'date': parts[9]
            }
        return None

    @profile
    def collect_and_interpolate_gps_data(self, serial_port, baud_rate, target_per_second=25):
        # global gps_data
        serialPort = serial.Serial(self.gps_port, baudrate=baud_rate, timeout=0.5)
        while True:
            try:
                data = serialPort.readline().decode('ascii', errors='replace')
                if 'GNRMC' in data:
                    # try:
                    msg = pynmea2.parse(data)
                    lat = self.convert_coordinates(float(msg.lat))
                    lon = self.convert_coordinates(float(msg.lon))
                    try:
                        speed_knots = float(msg.spd_over_grnd)
                    except (ValueError, TypeError):
                        speed_knots = 0.0
                    speed_kmph = speed_knots * 1.852
                    self.gps_data = {
                        "lat": f"{lat:.6f}",
                        "lon": f"{lon:.6f}",
                        "speed":f"{speed_kmph:.3f}",
                        "connected": True
                    }
                    # logger.info(f"GPS Data: Lat: {self.gps_data['lat']}")
                    # logger.info(f"GPS Data: Lon: {self.gps_data['lon']}")
                        # logger.info("GPS Detected")
                    # except:
                        # self.gps_data["connected"] = False
                        # logger.error("GPS Not Connected")
            except:
                self.gps_data["connected"] = False
                logger.error("Last except GPS not connected")
        pass

    def get_ip_location(self):
        try:
            response = requests.get("https://ipinfo.io/json")
            if response.status_code == 200:
                loc = response.json().get("loc", "0.0,0.0")
                lat, lon = map(float, loc.split(","))
                return lat, lon
            else:
                return None
        except:
            return None

    def to_ddmm_mmmm(self, degrees):
        d = int(degrees)
        m = abs(degrees - d) * 60
        return float(f"{d:02d}{m:06.4f}")

    def collect_and_interpolate_gps_data_mobile(self, serial_port, baud_rate, target_per_second=25):
        while True:
            try:
                coords = self.get_ip_location()
                if coords:
                    lat_dec, lon_dec = coords
                    lat_nmea = self.to_ddmm_mmmm(lat_dec)
                    lon_nmea = self.to_ddmm_mmmm(lon_dec)

                    lat = self.convert_coordinates(lat_nmea)
                    lon = self.convert_coordinates(lon_nmea)

                    self.gps_data = {
                        "lat": f"{lat:.6f}",
                        "lon": f"{lon:.6f}",
                        "connected": True
                    }

                else:
                    self.gps_data["connected"] = False

                time.sleep(1 / target_per_second)

            except Exception as e:
                self.gps_data["connected"] = False
                print("Last except GPS not connected:", str(e))

    @profile
    def start_gps_thread(self):
        self.gps_thread = threading.Thread(target=self.collect_and_interpolate_gps_data, args=(self.serial_port, self.baud_rate, self.target_per_second), daemon=True)
        self.gps_thread.start()
    @profile
    def measure_contact_wire_height(self, frame: np.ndarray, strip_bbox: Optional[Tuple[float, float, float, float]], pixel_to_mm_ratio: float, train_height: float) -> Optional[float]:
        if strip_bbox is None:
            return None

        strip_top_right_y = strip_bbox[1]

        measurement_point_y = strip_top_right_y
        distance_to_bottom = frame.shape[0] - measurement_point_y
        height_mm = distance_to_bottom * pixel_to_mm_ratio + train_height
        return float(height_mm)
    @profile
    def measure_stagger(self, frame: np.ndarray, strip_centroid: Tuple[float, float], contact_point: Tuple[float, float], pixel_to_mm_ratio: float) -> Optional[float]:
        if strip_centroid is None or contact_point is None:
            return None

        # Convert strip_centroid[0] and contact_point[0] to Python floats
        if isinstance(strip_centroid[0], torch.Tensor):
            strip_x = float(strip_centroid[0].cpu().numpy())  #Convert to Python float
        else:
            strip_x = float(strip_centroid[0])  # Ensure it's a Python float

        if isinstance(contact_point[0], torch.Tensor):
            contact_x = float(contact_point[0].cpu().numpy())  # Convert to Python float
        else:
            contact_x = float(contact_point[0])  # Ensure it's a Python float

        # Calculate horizontal distance
        horizontal_distance = abs(strip_x - contact_x)  # `abs()` works with Python floats

        # Convert pixel_to_mm_ratio if it is a Tensor
        if isinstance(pixel_to_mm_ratio, torch.Tensor):
            pixel_to_mm_ratio = float(pixel_to_mm_ratio.item())  # Convert to Python float

        stagger = horizontal_distance * pixel_to_mm_ratio
        return stagger
    @profile
    def calculate_pixel_to_mm_ratio(self, pantograph_width, pixel_width):
        return pantograph_width / pixel_width
    @profile
    def get_indian_time(self):
        indian_tz = pytz.timezone('Asia/Kolkata')
        return datetime.now(indian_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    @profile
    def get_class_names(self) -> List[str]:
        return ['CP', 'mast', 'strip']
    @profile
    def draw_measurement_text(self, frame, value, position, color, flash=False):
        if isinstance(value, str):
            text = value
        else:
            text = f"{value:.0f} mm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.3
        thickness = 3

        if flash and (self.frame_number % 30 < 15):
            color = (0, 0, 255)

        cv2.putText(frame, text, position, font, font_scale, color, thickness)
    @profile
    def draw_alert_boundary(self, frame, measurement_type):
        height, width = frame.shape[:2]
        color = self.alert_colors[measurement_type]
        cv2.rectangle(frame, (0, 0), (width-1, height-1), color, 3)
    @profile
    def write_log(self, log_type: str, data: List[str]):
        try:
            # if log_type == 'setting_distance':
            #     self.setting_distance_logged = True
            # elif not self.setting_distance_logged:
            #     return

            if log_type not in self.log_files:
                log_file_path = os.path.join(self.log_folder, f"{log_type}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                self.log_files[log_type] = open(log_file_path, 'w')

                headers = {
                    'stagger': ['Time', 'Stagger 1','Stagger 2' ,'Alert', 'Mast Name', 'Latitude', 'Longitude'],
                    'height': ['Time', 'Height Distance', 'Alert', 'Mast Name', 'Latitude', 'Longitude'],
                    'gradient': ['Time', 'Gradient', 'Relative Gradient', 'Alert', 'Mast Name', 'Latitude', 'Longitude'],
                    'mast': ['Time', 'Mast Name', 'Latitude', 'Longitude'],
                    'setting_distance': ['Time', 'Setting Distance', 'Alert', 'Mast Name', 'Latitude', 'Longitude'],
                    'double_contact': ['Time', 'Double Contact', 'Alert', 'Mast Name', 'Latitude', 'Longitude']
                }

                self.log_files[log_type].write(','.join(headers[log_type]) + '\n')

            log_entry = ','.join(map(str, data))
            self.log_files[log_type].write(f"{log_entry}\n")
            self.log_files[log_type].flush()
        except Exception as e:
            logger.error(f"Error writing to {log_type} log: {e}")

    def automate_all_log(self):
        combined_filename = os.path.join(self.log_folder, "automate_combined_log.csv")
        combined_filename_xlxs = os.path.join(self.log_folder, "FINAL_EXCEL_automate_combined_log.xlsx")
        temp_logo_image_path = os.path.join(self.log_folder, "temp_logo.png")
        stagger_headers = ["Time","Mast Name","Stagger 1","Stagger 2"]
        height_headers = ["Height Distance"]
        gradient_headers = ["Gradient", "Relative Gradient"]
        setting_distance_header = ["Setting Distance","Latitude", "Longitude"]
        double_contact_header = ["Double Contact","Mast Name"]

        stagger_df = pd.DataFrame(columns=stagger_headers)
        height_df = pd.DataFrame(columns=height_headers)
        gradient_df = pd.DataFrame(columns=gradient_headers)
        setting_distance_df = pd.DataFrame(columns=setting_distance_header)
        double_contact_df = pd.DataFrame(columns=double_contact_header)

        for file in os.listdir(os.path.join(self.log_folder)):
            file_path = os.path.join(self.log_folder, file)

            if file.endswith(".csv") and file.startswith("stagger"):
                logger.debug(f"Processing stagger file: {file_path}")
                try:
                    temp_df = pd.read_csv(file_path, usecols=stagger_headers)
                    stagger_df = pd.concat([stagger_df, temp_df], ignore_index=True)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

            elif file.endswith(".csv") and file.startswith("height"):
                logger.debug(f"Processing height file: {file_path}")
                try:
                    temp_df = pd.read_csv(file_path, usecols=height_headers)
                    height_df = pd.concat([height_df, temp_df], ignore_index=True)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

            elif file.endswith(".csv") and file.startswith("gradient"):
                logger.debug(f"Processing gradient file: {file_path}")
                try:
                    temp_df = pd.read_csv(file_path, usecols=gradient_headers)
                    gradient_df = pd.concat([gradient_df,temp_df],ignore_index=True)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
            elif file.endswith(".csv") and file.startswith("setting_distance"):
                logger.debug(f"Processing setting_distance file: {file_path}")
                try:
                    temp_df = pd.read_csv(file_path, usecols=setting_distance_header)
                    setting_distance_df = pd.concat([setting_distance_df,temp_df],ignore_index=True)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
            elif file.endswith(".csv") and file.startswith("double_contact"):
                logger.debug(f"Processing double_contact file: {file_path}")
                try:
                    temp_df = pd.read_csv(file_path, usecols=double_contact_header)
                    double_contact_df = pd.concat([double_contact_df,temp_df],ignore_index=True)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")


        max_len = max(len(stagger_df), len(height_df),len(gradient_df),len(setting_distance_df),len(double_contact_df))

        stagger_df = stagger_df.reindex(range(max_len)).reset_index(drop=True)
        height_df = height_df.reindex(range(max_len)).reset_index(drop=True)
        gradient_df = gradient_df.reindex(range(max_len)).reset_index(drop=True)
        setting_distance_df = setting_distance_df.reindex(range(max_len)).reset_index(drop=True)
        double_contact_df = double_contact_df.reindex(range(max_len)).reset_index(drop=True)

        combined_df = pd.concat([stagger_df, height_df, gradient_df,setting_distance_df,double_contact_df], axis=1)
        # combined_df.to_csv(combined_filename, index=False)

        image_data = self.get_embedded_image()
        with open(temp_logo_image_path, "wb") as img_file:
            img_file.write(image_data)

        with pd.ExcelWriter(combined_filename_xlxs, engine="openpyxl") as writer:
            combined_df.to_excel(writer, sheet_name="Data", index=False)

            # Load workbook & get sheet
            workbook = writer.book
            sheet = workbook["Data"]

            # Insert firm name in A1
            sheet.insert_rows(1)
            cell = sheet["A1"]
            cell.value = "GempertsIN Pvt Ltd"
            sheet.insert_rows(2)

            # Apply formatting (Bold, Underline, Font Size 28)
            cell.font = Font(bold=True, underline="single", size=28)

            img = Image(temp_logo_image_path)
            img.width = 200
            img.height = 50
            sheet.add_image(img, "C1")

            for col in sheet.columns:
                max_length = 0
                col_letter = col[0].column_letter
                for cell in col:
                    try:
                        if cell.value:
                            max_length = max(max_length, len(str(cell.value)))
                    except:
                        pass
                sheet.column_dimensions[col_letter].width = max_length + 2

            sheet.row_dimensions[1].height = 40  # Adjust row height for "GempertsIN Pvt Ltd"
            sheet.row_dimensions[2].height = 30  # Adjust row height for image

            # Save workbook
            workbook.save(combined_filename_xlxs)
        os.remove(temp_logo_image_path)
        logger.debug(f"All log files have been automated")

    def get_embedded_image(self):
        import base64
        image_data = b"""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"""
        return base64.b64decode(image_data)

    # ------------- #arm ----------------------
    @profile
    def bbox_overlap(self, bbox1, bbox2):
        """
        Check if two bounding boxes overlap
        bbox format: [x1, y1, x2, y2]
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Check for overlap
        x_overlap = not (x2_1 < x1_2 or x1_1 > x2_2)
        y_overlap = not (y2_1 < y1_2 or y1_1 > y2_2)

        return x_overlap and y_overlap
        # ------------- #arm ----------------------

    def add_padding(self, bbox, pad, frame_width, frame_height):
        x1 = max(int(bbox[0]) - pad, 0)
        y1 = max(int(bbox[1]) - pad, 0)
        x2 = min(int(bbox[2]) + pad, frame_width - 1)
        y2 = min(int(bbox[3]) + pad, frame_height - 1)
        return (x1, y1, x2, y2)

    @profile
    # def process_frame(self, frame, fps, train_speed, results):
    def process_frame(self, frame, fps, results):

        current_time = self.get_indian_time()

        strip_bbox = None
        contact_points = []
        mast_detected = None
        arm_bboxes = []
        arm_results = self.arm_model(frame, verbose=False, conf=self.arm_conf_threshold, task='detect')
        frame_height, frame_width = frame.shape[:2]
        padding = 10

        # ------------- #arm ----------------------

        for result in arm_results:
            for box in result.boxes:
                bbox = box.xyxy[0]
                confidence = box.conf[0].item()

                if confidence >= self.arm_conf_threshold:
                    padded_bbox = self.add_padding(bbox, padding, frame_width, frame_height)
                    arm_bboxes.append(padded_bbox)
                    # Draw arm bounding box
                    cv2.rectangle(frame, (padded_bbox[0], padded_bbox[1]), (padded_bbox[2], padded_bbox[3]), (0, 165, 255), 2)
                    cv2.putText(frame, "Arm", (padded_bbox[0], padded_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0]
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                if class_id == 2 and confidence >= 0.4:  # strip
                    if strip_bbox is None:
                        strip_bbox = bbox
                        strip_centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                        # Draw bounding box for strip
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

                        # Add label
                        label = "strip" # {detections.confidence[i]:.2f}"
                        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                        # Center Point
                        center_x = int(strip_centroid[0])
                        center_y = int(strip_centroid[1])
                        cv2.circle(frame, (center_x, center_y), 6, (255, 255, 0), -1)
                        cv2.putText(frame, "C", (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                elif int(box.cls[0].item()) == 1 and box.conf[0].item() >= 0.25:  # mast class
                    mast_detected = bbox  # Store the mast bbox
                    mast_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                    # Draw mast detection
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, "mast", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                elif int(box.cls[0].item()) == 0 and box.conf[0].item() >= 0.25:  # contact_point
                    contact_points.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))

                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    label = "contact_point"
                    cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    # label = "mast"
                    # cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Calculate pixel to mm ratio
        pixel_to_mm_ratio = None
        if strip_bbox is not None:
            strip_width_pixels = strip_bbox[2] - strip_bbox[0]
            pixel_to_mm_ratio = self.calculate_pixel_to_mm_ratio(self.pantograph_width, strip_width_pixels)

            text = f"Panto Width: {self.pantograph_width} mm"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.4
            thickness = 3
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            text_x = int((strip_bbox[0] + strip_bbox[2] - text_size[0]) / 2)
            text_y = int(strip_bbox[3] + 40)

            cv2.putText(frame, text, (text_x, text_y),
                        font, font_scale, self.strip_color, thickness)

        logs = {
            'stagger_1': [],
            'height': [],
            'gradient': [],
            'setting_distance': [],
            'mast': [],
            'double_contact': []
        }
        alert_y_positions = {
            "GPS": 30,
            "Stagger": 60,
            "Wire Height": 90,
            "Setting Distance": 120,
            "Double Contact": 150,
            "Gradient": 180
        }

        # Helper function to get GPS data or N/A
        def get_gps_data():
            if self.gps_data["connected"]:
                return self.gps_data["lat"], self.gps_data["lon"]
            else:
                return 2817.10211, 8020.86720

        if pixel_to_mm_ratio is not None:
            lat, lon = get_gps_data()
            gps_connected = self.gps_data["connected"]

            if gps_connected:
                nearest_mast = self.find_nearest_mast(
                    float(lat) if lat else None,
                    float(lon) if lon else None
                )
                mast_name = nearest_mast['location'] if nearest_mast else "N/A"
            else:
                mast_name = "Unknown mast"

            # Approach 2
            # if gps_connected:
                # Find exact match in mast data
            matched_mast = next((mast for mast in self.mast_data
                                if abs(float(mast['Lat']) - float(lat)) < 0.0001
                                and abs(float(mast['Lon']) - float(lon)) < 0.0001),
                                None)

            if matched_mast:
                # If strip_bbox is available, calculate contact wire height
                if strip_bbox is not None:
                    contact_wire_height = self.measure_contact_wire_height(frame, strip_bbox, pixel_to_mm_ratio, self.train_height)
                    contact_wire_height = contact_wire_height + self.height_correction + self.roof_to_frame_height
                    # logger.info(f"Contact Wire Height: {contact_wire_height}")

                    # Store the height for this specific mast location
                    if contact_wire_height:
                        self.mast_heights.append(contact_wire_height)

                        # If we have at least two height measurements
                        if len(self.mast_heights) > 1:
                            # Calculate distance between matched masts from Excel data
                            distance = matched_mast.get('distance', 0)
                            self.distance_between_mast = distance
                            self.mast_distances.append(distance)


            # Rest of your existing code for visualizations and other measurements
            if mast_detected is not None and (self.frame_number - self.last_mast_detection) > self.mast_cooldown:
                self.mast_count += 1
                self.last_mast_detection = self.frame_number

                # mast_center = ((mast_detected[0] + mast_detected[2]) / 2, (mast_detected[1] + mast_detected[3]) / 2)
                # cv2.circle(frame, (int(mast_center[0]), int(mast_center[1])), 5, (0, 255, 0), -1)
                # cv2.putText(frame, mast_name, (int(mast_detected[0]), int(mast_detected[1] - 10)),
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

                self.last_mast_frame = self.frame_number
                logs['mast'] = [current_time, mast_name,
                        lat if gps_connected else "N/A",
                        lon if gps_connected else "N/A"]

            # Stagger calculation and visualization
            if strip_centroid and contact_points:
                for i, cp in enumerate(contact_points[:2]):
                    stagger = self.measure_stagger(frame, strip_centroid, cp, pixel_to_mm_ratio)
                    stagger = stagger + self.stagger_distance_correction
                    if stagger is not None:
                        # if contact_points[0][0] < strip_centroid[0]:
                            #     stagger += 30
                        start_point = (int(strip_centroid[0]), int(strip_centroid[1]))
                        end_point = (int(cp[0]), int(strip_centroid[1]))
                        color = (51, 255, 255) if i == 0 else (255, 102, 255)  # Different color for each line
                        cv2.line(frame, start_point, end_point, color, 2)
                        text_position = (
                            int((strip_centroid[0] + cp[0]) / 2),
                            int(strip_centroid[1]) - 30 - (i * 20)  # Offset each label vertically
                        )
                        self.draw_measurement_text(frame, stagger, text_position, color)

            # -----------#stagger------------ #arm ----------------------
            if not hasattr(self, 'current_overlap_staggers_1'):
                self.current_overlap_staggers_1 = []
                self.was_overlapping = False

            if not hasattr(self, 'current_overlap_staggers_2'):
                self.current_overlap_staggers_2 = []
                self.was_overlapping = False

            arm_overlap = False
            if strip_bbox is not None and contact_points:
                # Check if any arm overlaps with the strip
                arm_overlap = any(
                        self.bbox_overlap(strip_bbox, arm_bbox)
                        for arm_bbox in arm_bboxes
                    )
                if arm_overlap:
                    for i, cp in enumerate(contact_points[:2]):
                        stagger = self.measure_stagger(frame, strip_centroid, cp, pixel_to_mm_ratio)
                        stagger = int(stagger) + self.stagger_distance_correction

                        # logger.debug(f"Current mast direction : {self.current_mast_direction}")
                        if cp[0] < strip_centroid[0]:
                            stagger = stagger * (-1) if self.current_mast_direction == "Left" else stagger
                        elif cp[0] > strip_centroid[0]:
                            stagger = stagger * (-1) if self.current_mast_direction == "Right" else stagger

                        if stagger is not None:
                            if i == 0:
                                self.current_overlap_staggers_1.append(stagger)
                            elif i == 1:
                                self.current_overlap_staggers_2.append(stagger)

                            logger.debug(f"Added stagger[{i}] to collection: {stagger}")
                    self.was_overlapping = True

                elif self.was_overlapping and (self.current_overlap_staggers_1 or self.current_overlap_staggers_2):

                        # Select stagger_1 & 2 (from first contact point)
                        # 1. median way
                        # last_stagger_1 = float(np.median(np.array(self.current_overlap_staggers_1)))
                        # last_stagger_2 = float(np.median(np.array(self.current_overlap_staggers_2)))
                        # logger.debug(f"Overlap ended. Median stagger: {last_stagger_1}")
                        # logger.debug(f"Overlap ended. Median stagger: {last_stagger_2}")

                        # 2. middle term way
                        if self.current_overlap_staggers_1:
                            if len(self.current_overlap_staggers_1) % 2 == 0:
                                last_stagger_1 = self.current_overlap_staggers_1[int(len(self.current_overlap_staggers_1) / 2 - 1)]
                            else:
                                last_stagger_1 = self.current_overlap_staggers_1[int(len(self.current_overlap_staggers_1) / 2)]
                            logger.debug(f"Selected stagger 1 (Middle): {last_stagger_1}")
                        else:
                            last_stagger_1 = None

                        if self.current_overlap_staggers_2:
                            if len(self.current_overlap_staggers_2) % 2 == 0:
                                last_stagger_2 = self.current_overlap_staggers_2[int(len(self.current_overlap_staggers_2) / 2 - 1)]
                            else:
                                last_stagger_2 = self.current_overlap_staggers_2[int(len(self.current_overlap_staggers_2) / 2)]
                            logger.debug(f"Selected stagger 2 (Middle): {last_stagger_2}")
                        else:
                            last_stagger_2 = None

                        # 3. last stagger from list
                        # last_stagger_1 = self.current_overlap_staggers_1[-1]
                        # last_stagger_2 = self.current_overlap_staggers_2[-1]
                        # logger.debug(f"Last stagger: {last_stagger}\n")

                        # # 4. Mean stagger from list
                        # last_stagger_1 = int(np.mean(np.abs(np.array(self.current_overlap_staggers_1))))
                        # last_stagger_2 = int(np.mean(np.abs(np.array(self.current_overlap_staggers_2))))
                        # logger.debug(f"Last stagger 1 (Mean): {last_stagger_1}\n")
                        # logger.debug(f"Last stagger 2 (Mean): {last_stagger_2}\n")

                        # Update GPS and Mast info retrieval (use the previous method)
                        lat, lon = get_gps_data()
                        gps_connected = self.gps_data["connected"]
                        if gps_connected:
                            nearest_mast = self.find_nearest_mast(
                                float(lat) if lat else None,
                                float(lon) if lon else None
                            )
                            mast_name = nearest_mast['location'] if nearest_mast else "N/A"
                        else:
                            mast_name = f"Mast_{self.mast_count + 1}"

                        if abs(stagger) > 300:
                            logs['stagger'] = [current_time, f"{last_stagger_1}", f"{last_stagger_2}" ,"Yes", mast_name, lat if lat else "N/A", lon if lon else "N/A"]
                        else:
                            logs['stagger'] = [current_time, f"{last_stagger_1}", f"{last_stagger_2}" ,"No", mast_name, lat if lat else "N/A", lon if lon else "N/A"]

                        # Reset our collection and overlap flag
                        self.current_overlap_staggers_1 = []
                        self.current_overlap_staggers_2 = []
                        self.was_overlapping = False
                else:
                    self.current_overlap_staggers_1 = []
                    self.current_overlap_staggers_2 = []
                    self.was_overlapping = False
            # -----------#stagger------------ #arm END----------------------

            # Contact wire height visualization
            if strip_bbox is not None:
                if not isinstance(strip_bbox, tuple):
                    strip_bbox = tuple(strip_bbox.cpu().numpy())

                contact_wire_height = self.measure_contact_wire_height(frame, strip_bbox, pixel_to_mm_ratio, self.train_height)
                contact_wire_height = contact_wire_height + self.height_correction + self.roof_to_frame_height
                # logger.info(f"Contact Wire Height 2: {contact_wire_height}")
                if contact_wire_height:
                    measurement_point_x = max(int(strip_bbox[2] - 100), int(strip_bbox[0]))
                    measurement_point_y = int(strip_bbox[1])

                    # Add a small circle to indicate the starting point of measurement
                    cv2.circle(frame, (measurement_point_x, measurement_point_y), 5, (255, 0, 0), -1)

                    cv2.line(frame, (measurement_point_x, measurement_point_y), (measurement_point_x, frame.shape[0]), (51, 255, 153), 2)
                    self.draw_measurement_text(frame, f"Total H: {contact_wire_height:.2f}", (measurement_point_x + 10, frame.shape[0] - 30), (51, 255, 153))

            if not hasattr(self, 'current_overlap_heights'):
                self.current_overlap_heights = []
                self.was_overlapping_heights = False

            arm_overlap = False
            if strip_bbox is not None :
                # Check if any arm overlaps with the strip
                arm_overlap = any(
                    self.bbox_overlap(strip_bbox, arm_bbox)
                    for arm_bbox in arm_bboxes
                )

                if arm_overlap:
                    contact_wire_height = self.measure_contact_wire_height(frame, strip_bbox, pixel_to_mm_ratio, self.train_height)
                    contact_wire_height = contact_wire_height + self.height_correction + self.roof_to_frame_height

                    if contact_wire_height:
                        self.current_overlap_heights.append(contact_wire_height)
                        # logger.debug(f"Added Height to collection: {contact_wire_height}")
                    self.was_overlapping_heights = True

                elif self.was_overlapping_heights and self.current_overlap_heights:
                    # median_height = float(np.median(np.array(self.current_overlap_heights)))
                    last_height = self.current_overlap_heights[-1]
                    # logger.debug(f"Overlap ended. Last height: {last_height}")
                    contact_wire_height = last_height

                    if (contact_wire_height > 7800 or contact_wire_height < 4500):
                        logs['height'] = [current_time, f"{contact_wire_height:.2f}",
                                "Yes", mast_name, lat if lat else "N/A", lon if lon else "N/A"]
                    else:
                        logs['height'] = [current_time, f"{contact_wire_height:.2f}",
                            "No", mast_name, lat if lat else "N/A", lon if lon else "N/A"]

                    if isinstance(contact_wire_height, torch.Tensor):
                            contact_wire_height = last_height.item()

                    # height_diff = self.previous_mast_height - contact_wire_height
                    self.current_gradient = self.calculate_gradient(self.previous_mast_height, contact_wire_height, self.distance_between_mast)
                    self.previous_mast_height = contact_wire_height
                    # logger.debug(f"Height Diff :{height_diff} mm | distance :{self.distance_between_mast:.5f} | Current gradient : {self.current_gradient:.5f}")
                    relative_gradient = self.calculate_relative_gradient(
                                    self.mast_heights[-self.relative_gradient_window:],
                                    self.mast_distances[-self.relative_gradient_window+1:]
                                )
                    logs['gradient'] = [
                        current_time,
                        f"{self.current_gradient:.2f}",
                        f"{relative_gradient:.2f}",
                        "Yes" if abs(self.current_gradient) > 3 else "No",
                        mast_name,
                        lat,
                        lon
                    ]
                    self.current_overlap_heights = []
                    self.was_overlapping_heights = False
            else:
                self.current_overlap_heights = []
                self.was_overlapping_heights = False


            # Double Contact point
            if len(contact_points) >= 2:
                double_contact = abs(contact_points[0][0] - contact_points[1][0]) * pixel_to_mm_ratio
                cv2.line(frame, (int(contact_points[0][0]), int(contact_points[0][1])), (int(contact_points[1][0]), int(contact_points[0][1])), (255, 255, 0), 2)
                self.draw_measurement_text(frame, double_contact, (int((contact_points[0][0] + contact_points[1][0])/2), int(contact_points[0][1]) - 60), (255, 255, 0))

                if abs(double_contact) > 500: #or double_contact > 800:
                    cv2.putText(frame, "ALERT: Double Contact", (10, alert_y_positions["Double Contact"]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    logs['double_contact'] = [current_time, f"{double_contact:.2f}",
                        "Yes", mast_name, lat if lat else "N/A", lon if lon else "N/A"]
                else:
                    logs['double_contact'] = [current_time, f"{double_contact:.2f}",
                        "No", mast_name, "N/A", "N/A"]

        self.frame_number += 1

        # Write logs to files
        for log_type, data in logs.items():
            if data:
                self.write_log(log_type, data)
        return frame, logs

    @profile
    def __del__(self):
        for file in self.log_files.values():
            file.close()

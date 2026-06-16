from datetime import datetime
import pytz
import cv2
import cupy as cp
from cupy.cuda import Device
import time
from old_models_realtimeprocessor import RealtimeProcessor
from line_profiler import profile
cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())

import logging

# Configure the logger
logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)


class ZEDProcessor:
    @profile
    def __init__(self, realtime_processor: RealtimeProcessor, camera_offset: float =28, setting_distance_correction: float = 0):
        # Initialize CUDA device
        self.device = Device(0)  # Use first GPU
        with self.device:

            self.zed_rgb_frame = None
            self.depth = None

            # Thresholds
            self.MIN_SAFE_DISTANCE = 82.67
            self.MAX_SAFE_DISTANCE = 236
            self.MAX_DETECTION_DISTANCE = 236

            # Mast tracking
            self.current_mast_distances = []
            self.last_mast_time = None
            self.mast_cooldown = 2.0
            self.mast_count = 0
            self.current_mast_frames = 0
            self.mast_processed = False
            self.indian_tz = pytz.timezone('Asia/Kolkata')
            self.cached_time = None
            self.realtime_processor = realtime_processor
            self.camera_offset = float(camera_offset)
            self.setting_distance_correction = float(setting_distance_correction)
            logger.info(f"Setting Distance Correction: {self.setting_distance_correction}")

    @profile
    # def process_depth_values(self, depth_values):
    #     return float(cp.median(depth_values)) if depth_values.size > 0 else float('inf')+
    
    # try mode
    def process_depth_values(self, depth_values):
        if len(depth_values) == 0:
            return float('inf')  # Return a large value if no valid depths are found
        
        # Convert to integer bins for mode calculation
        rounded_values = cp.around(depth_values, decimals=2)  # Round to 2 decimal places
        unique_values, counts = cp.unique(rounded_values, return_counts=True)    
        return float(unique_values[cp.argmax(counts)])

    @profile
    def get_depth_values_in_roi(self, depth ,x1, y1, x2, y2):
        with self.device:

            depth_array = cp.array(depth)

            roi_depth_values = depth_array[y1:y2, x1:x2]

            valid_mask = (roi_depth_values > 0) & (roi_depth_values < self.MAX_DETECTION_DISTANCE)
            valid_depth_values = roi_depth_values[valid_mask]

            return valid_depth_values

    @profile
    def process_frame(self, detection_results):
        with self.device:
            if self.zed_rgb_frame is None:
                return None, None, None, None
                
            setting_distance, alert_status, mast_data = None, False, None
            current_time = self.get_indian_time()
            current_time_seconds = time.time()
            mast_detected = False

            # Create a copy of the frame for drawing
            display_frame = self.zed_rgb_frame.copy() if self.zed_rgb_frame is not None else None
            _, frame_width, _ = self.zed_rgb_frame.shape

            center_left = int(frame_width * 0.40)
            center_right = int(frame_width * 0.60)
            for result in detection_results:
                for box in result.boxes:
                    if box.cls != 1:  # Skip if not mast class
                        continue

                    mast_detected = True
                    self.realtime_processor.current_mast_direction = "Right" if mast_detected else None
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x1 >= center_left and x2 <= center_right:
                        conf = float(box.conf[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"Mast {conf:.2f}"
                        cv2.putText(display_frame, label, (x1 - self.zed_rgb_frame.shape[1], y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # width = x2 - x1
                        # height = y2 - y1
                        # width_reduction = width * 0.4  # 20% reduction horizontally
                        # height_reduction = height * 0.50  # 10% reduction vertically
                        # x1 = int(x1 + width_reduction / 2)
                        # x2 = int(x2 - width_reduction / 2)
                        # y1 = int(y1 + height_reduction / 2)
                        # y2 = int(y2 - height_reduction / 2)
                        depth_values = self.get_depth_values_in_roi(self.depth,x1, y1, x2, y2)
                        
                        if depth_values.size > 0:   
                            current_distance = self.process_depth_values(depth_values)
                            
                            if current_distance < self.MAX_DETECTION_DISTANCE:
                                # logger.info(f"Mast Distance: {current_distance}")
                                final_distance = current_distance + self.camera_offset + self.setting_distance_correction
                                final_distance = final_distance * 0.0254
                                # logger.info(f"Mast Distance with OffSet: {final_distance}")
                                self.current_mast_distances.append(final_distance)
                                self.current_mast_frames += 1
                                self.last_mast_time = current_time_seconds
                                self.mast_processed = False

                                # distance_label = f"Distance: {current_distance:.2f} Feet"
                                distance_label_with_offset = f"Distance: {final_distance:.4f} M"
                                
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                # cv2.putText(display_frame, distance_label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(display_frame, distance_label_with_offset, (center_x, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            if (len(self.current_mast_distances) > 0 and 
                not self.mast_processed and
                (not mast_detected and current_time_seconds - self.last_mast_time > 0.5)):
                
                setting_distance = float(cp.mean(cp.array(self.current_mast_distances)))
                # setting_distance_min = float(cp.min(cp.array(self.current_mast_distances)))
                # logger.debug(f"Setting Distance Min : {setting_distance_min}")
                setting_distance = setting_distance
                alert_status = setting_distance < self.MIN_SAFE_DISTANCE or setting_distance > self.MAX_SAFE_DISTANCE
                # if not alert_status:
                lat, lon, mast_name = self._get_mast_info(current_time, setting_distance, alert_status)
                setting_distance_data = [current_time, f"{setting_distance:.4f}", "Yes" if alert_status else "No", mast_name, lat, lon]
                
                mast_data = {
                    'time': current_time,
                    'setting_distance': setting_distance_data,
                    'mast': [current_time, mast_name, lat, lon]
                }
                
                # Log setting distance data
                self.realtime_processor.write_log('setting_distance', setting_distance_data)
                # Reset for next mast
                self.current_mast_distances.clear()
                self.current_mast_frames = 0
                self.mast_processed = True
                self.mast_count += 1
                self.realtime_processor.current_mast_direction = None

                # Draw alert status if needed
                if alert_status:
                    cv2.putText(display_frame, "ALERT: Outside Safe Range", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            return display_frame, setting_distance, alert_status, mast_data

    @profile
    def get_indian_time(self):
        """Get current time in Indian timezone with caching"""
        if self.cached_time is None:
            self.cached_time = datetime.now(self.indian_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        return self.cached_time

    @profile
    def _get_mast_info(self, current_time, setting_distance, alert_status):
        """Get mast information including GPS coordinates and mast name"""
        lat, lon = "N/A", "N/A"
        mast_name = f"Mast_{self.mast_count + 1}"
        
        try:
            if self.realtime_processor and hasattr(self.realtime_processor, 'gps_data'):
                # Debug print
                # logger.info(f"GPS Data: {self.realtime_processor.gps_data}")
                
                if self.realtime_processor.gps_data.get("connected", False):
                    lat = self.realtime_processor.gps_data.get("lat", "N/A")
                    lon = self.realtime_processor.gps_data.get("lon", "N/A")
                    # logger.info(f"{lat},{lon}")
                    
                    if lat != "N/A" and lon != "N/A":
                        nearest_mast = self.realtime_processor.find_nearest_mast(
                            float(lat), float(lon)
                        )
                        
                        if nearest_mast:
                            mast_name = nearest_mast.get('location', mast_name)
                            logger.info(f"Found nearest mast: {mast_name}")
                        else:
                            logger.info("No nearest mast found")
                    else:
                        logger.info("Invalid lat/lon values")
                else:
                    logger.error("GPS not connected")
        except Exception as e:
            logger.info(f"Error in _get_mast_info: {e}")
            
        return lat, lon, mast_name
    @profile
    def close(self):
        """Properly close the ZED camera"""
        self.zed.close()
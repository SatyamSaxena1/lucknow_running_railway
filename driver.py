import numpy as np
import pyzed.sl as sl
from multiprocessing import shared_memory
import logging
import os
import time

logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

def safe_create_shared_memory(name, size):
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        return shm
    except FileExistsError:
        return shared_memory.SharedMemory(name=name)

def sender(frame_shape):
    zed = sl.Camera()
    init_params = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        coordinate_units=sl.UNIT.INCH,
        camera_resolution=sl.RESOLUTION.HD720,
        camera_fps=60,
        depth_maximum_distance=180,
        depth_minimum_distance=40,
        depth_stabilization=50
    )

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        logger.error("Failed to open ZED camera.")
        return

    frame_shape_depth = (720, 1280)
    shm_color = safe_create_shared_memory("shared_color", int(np.prod(frame_shape)) * np.dtype(np.uint8).itemsize)
    shm_depth = safe_create_shared_memory("shared_depth", int(np.prod(frame_shape_depth)) * np.dtype(np.float32).itemsize)

    logger.info("Shared memory for color and depth created.")

    frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm_color.buf)
    frame_buffer_depth = np.ndarray(frame_shape_depth, dtype=np.float32, buffer=shm_depth.buf)

    runtime_parameters = sl.RuntimeParameters(confidence_threshold=50, texture_confidence_threshold=50)
    image = sl.Mat()
    depth = sl.Mat()

    shutdown_flag = "all_shutdown.flag"
    color_flag = "color.flag"
    depth_flag = "depth.flag"

    try:
        while True:
            if os.path.exists(shutdown_flag):
                logger.info("Shutdown flag detected. Exiting sender loop.")
                break
            try:
                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                    frame = image.get_data()[:, :, :3]
                    depth_data = np.array(depth.get_data())

                    np.copyto(frame_buffer, frame)
                    np.copyto(frame_buffer_depth, depth_data)

                    # Signal that new frames are ready
                    open(color_flag, "w").close()
                    open(depth_flag, "w").close()
            except Exception as e:
                logger.warning(f"Exception during frame grab or flag signaling: {e}")
            time.sleep(0.005)  # Slight delay to reduce CPU load

    except KeyboardInterrupt:
        logger.info("Sender interrupted by keyboard.")
    finally:
        logger.info("Cleaning up shared memory and closing ZED.")
        try:
            shm_color.close()
            shm_color.unlink()
            shm_depth.close()
            shm_depth.unlink()
        except Exception as e:
            logger.error(f"Error cleaning shared memory: {e}")

        for flag in [color_flag, depth_flag, shutdown_flag]:
            if os.path.exists(flag):
                os.remove(flag)

        zed.close()
        logger.info("ZED camera closed and resources released.")

if __name__ == "__main__":
    # Cleanup old flag files before starting
    for flag_file in ["color.flag", "depth.flag", "all_shutdown.flag"]:
        if os.path.exists(flag_file):
            os.remove(flag_file)

    sender((720, 1280, 3))

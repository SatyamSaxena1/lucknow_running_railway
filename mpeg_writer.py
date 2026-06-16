import threading
import queue
import subprocess
import time
import os


class FFmpegWriter:
    def __init__(self, output_video, fps, width, height, log_folder_path, codec="h264"):
        self.frame_queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        self.log_folder_path = log_folder_path
        self.codec = codec
        self.process = self._start_ffmpeg_process(output_video, fps, width, height)
        
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        
        self.thread.start()
        self.stderr_thread.start()

    def _start_ffmpeg_process(self, output_video, fps, width, height):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_parts = output_video.rsplit('.', 1)
        unique_output_video = os.path.join(
            self.log_folder_path,
            f"{filename_parts[0]}_{timestamp}.{filename_parts[1]}"
        )

        cmd = [
            "ffmpeg",
            "-loglevel", "warning",  # Reduce spam
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "h264_nvenc" if self.codec == "h264" else "hevc_nvenc",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            unique_output_video,
        ]

        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _read_stderr(self):
        try:
            while True:
                if self.process.stderr is None:
                    break
                line = self.process.stderr.readline()
                if not line:
                    break
                print("[FFmpeg stderr]", line.decode(errors='ignore').strip())
        except Exception as e:
            print(f"[FFmpegWriter] Error reading stderr: {e}")

    def _writer_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break

                # Check if ffmpeg process is alive
                if self.process.poll() is not None:
                    print("[FFmpegWriter] FFmpeg process has exited.")
                    break

                self.process.stdin.write(frame)
            except queue.Empty:
                continue
            except (BrokenPipeError, OSError) as e:
                print(f"[FFmpegWriter] Error writing to FFmpeg stdin: {e}")
                break
            except Exception as e:
                print(f"[FFmpegWriter] Unexpected error in writer loop: {e}")
                break

    def write(self, frame):
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            print("[FFmpegWriter] Frame queue full. Dropping frame.")

    def close(self):
        self.stop_event.set()
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass

        self.thread.join()

        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception as e:
            print(f"[FFmpegWriter] Error closing stdin: {e}")

        self.process.wait()

        try:
            self.stderr_thread.join(timeout=2)
        except Exception:
            pass

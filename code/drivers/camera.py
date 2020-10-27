import numpy as np
from picamera import PiCamera
import io, time
from turbojpeg import TurboJPEG
from threading import Thread

jpeg = TurboJPEG()
"""
calib_data: Dictionary with the following key/value pairs
'K': 3x3 intrinsic
'D': distortion coefficients
'R': rectification matrix
'P': projection matrix
'distortion_model': camera distortion_model
"""


class CameraDevice:
    def __init__(self, img_format='bgr', decode_jpeg=False):
        self.camera = PiCamera()
        self.camera.framerate = 30
        self.img_w = 640
        self.img_h = 480
        self.img_format = img_format
        self.camera.resolution = (self.img_w, self.img_h)
        self.decode_jpeg = decode_jpeg  # Stream mode with img_format=='jpeg', decode jpeg to numpy array or not

        self.camera.exposure_mode = 'sports'
        # For Stream
        self.stream = io.BytesIO()
        # For Capture
        self.cap = io.BytesIO()

        # For streaming
        """
        decode_jpeg=False, img_format='jpeg': stream_img is encoded jpeg in np array type : ~10 fps
        decode_jpeg=True, img_format='jpeg': stream_img is decoded bgr array: ~5fps
        img_format='bgr': stream_img is decoded bgr array: ~4fps
        """
        self.stream_img = None  # Latest streaming image in numpy array (no compression/compressed )
        self.is_streaming = False
        self.stream_thd = None

    def grab_stream_img(self):
        while self.is_streaming:
            yield self.stream  # Data written by capture_sequence
            stream_data = self.stream.getvalue()
            self.stream_img = self.pre_process_img(stream_data)
            self.stream.seek(0)
            self.stream.truncate()

    def pre_process_img(self, stream_data):
        if self.img_format == 'jpeg':
            if self.decode_jpeg:
                return jpeg.decode(stream_data)
            else:
                return np.frombuffer(stream_data, dtype=np.uint8)
        elif self.img_format == 'bgr':
            return np.frombuffer(stream_data, dtype=np.uint8).reshape((self.img_h, self.img_w, 3))  # BGR, 24bit
        else:
            raise NotImplemented

    def capture(self):  # Single image capture, better quality
        self.camera.capture(self.cap, format=self.img_format, use_video_port=False)
        stream_data = self.cap.getvalue()
        self.cap.seek(0)
        self.cap.truncate()
        return self.pre_process_img(stream_data)

    def run_stream(self):
        print('Streaming start')
        while self.is_streaming:
            gen = self.grab_stream_img()
            try:
                self.camera.capture_sequence(
                    gen,
                    self.img_format,  # Use jpeg for faster processing
                    use_video_port=True,  # Faster
                    splitter_port=0
                )
            except StopIteration:  # self.stop_stream=True, one time only
                pass
        print('Streaming stop')

    def start_stream_thd(self):
        self.stream_thd = Thread(target=self.run_stream)
        self.is_streaming = True
        self.stream_thd.start()

    def stop_stream_thd(self):
        if self.is_streaming:
            self.is_streaming = False
            self.stream_thd.join()
            del self.stream_thd


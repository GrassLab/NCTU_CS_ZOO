from drivers.camera import CameraDevice


class CameraControl(object):
    def __init__(self, img_fmt='bgr'):
        self.cam = CameraDevice(img_fmt)

    def capture(self):
        return self.cam.capture()

    def start_stream(self):
        if not self.cam.is_streaming:
            self.cam.start_stream_thd()

    def stop_stream(self):
        if self.cam.is_streaming:
            self.cam.stop_stream_thd()

    def get_stream_img(self):
        if not self.cam.is_streaming:
            self.cam.start_stream_thd()
        while self.cam.stream_img is None:
            pass
        return self.cam.stream_img

    def set_jpeg(self):
        if self.cam.img_format != 'jpeg':
            if self.cam.is_streaming:
                self.cam.stop_stream_thd()
                self.cam.img_format = 'jpeg'
                self.cam.start_stream_thd()
                self.cam.stream_img = None
            else:
                self.cam.img_format = 'jpeg'
            print('Camera format change to jpeg, decoded={}'.format(self.cam.decode_jpeg))

    def set_bgr(self):
        if self.cam.img_format != 'bgr':
            if self.cam.is_streaming:
                self.cam.stop_stream_thd()
                self.cam.img_format = 'bgr'
                self.cam.start_stream_thd()
                self.cam.stream_img = None
            else:
                self.cam.img_format = 'bgr'
            print('Camera format change to bgr (raw) numpy array')

    def get_fmt(self):
        return self.cam.img_format

    def decode_jpeg(self):
        return self.cam.decode_jpeg

    def __del__(self):
        self.stop_stream()


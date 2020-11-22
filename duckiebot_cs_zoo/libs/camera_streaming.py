
class CameraProducer:
    """
    use at camera streaming producer e.g. duckie bot
    """
    try:
        from libs.camera_control import CameraControl
    except ImportError:
        try:
            from .camera_control import CameraControl
        except ImportError:
            CameraControl = None
    camera: CameraControl = CameraControl() if CameraControl is not None else None
    max_framerate: int = 10
    jpeg_quality: int = 50
    prev_frametime: int = None
    NANOSECOND: int = 1000000000

    import cv2
    cv2 = cv2
    import numpy as np
    np = np
    import time
    time = time
    import datetime
    datetime

    font_setting: dict = dict(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=2,
    )

    def __init__(self, max_framerate: int = None, quality: int = None):
        if max_framerate is not None:
            self.max_framerate = max_framerate
        if quality is not None:
            self.jpeg_quality = quality
        self.prev_frametime = self.time.time_ns() - (self.NANOSECOND / self.max_framerate)
        self.camera.start_stream()

    def get_img_bytedata(self) -> bytes:
        # control frame rate
        while self.time.time_ns() < self.prev_frametime + \
                self.NANOSECOND / self.max_framerate:
            self.time.sleep(0.001)
        self.prev_frametime = self.time.time_ns()

        img = self.camera.get_stream_img()
        img = self.cv2.putText(
            img, str(self.datetime.datetime.now()), (0, img.shape[0] - 20),
            color=(255, 255, 255), **self.font_setting,
        )
        ok, encode_img = self.cv2.imencode('.jpg', img, [
            self.cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
        ])
        if not ok:
            raise RuntimeError(
                f'cv2 encode jpg error with quality = {self.jpeg_quality}')
        img_bytedata: bytes = encode_img.tobytes()
        return img_bytedata


class CameraConsumer:
    """
    use at camera streaming consumer e.g. your host or computer
    """
    WINDOW_NAME: str = __file__
    counter: dict = None

    import numpy as np
    np = np
    import cv2
    cv2 = cv2

    def __init__(self, win_name: str = None):
        if win_name is not None:
            self.WINDOW_NAME = win_name
        self.cv2.namedWindow(self.WINDOW_NAME)

        # mininal counter
        import time
        self.time = time
        self.counter = {
            'start': None,
            'count': None,
        }

    def show_img_bytedata(self, img_bytedata: bytes):
        self.update_counter()
        encode_img: self.np.ndarray = self.np.frombuffer(
            img_bytedata, dtype=self.np.uint8,
        )
        img = self.cv2.imdecode(encode_img, -1)
        self.cv2.imshow(self.WINDOW_NAME, img)
        self.cv2.waitKey(1)

    def update_counter(self):
        if self.counter['start'] is None:
            self.counter['start'] = self.time.time()
            self.counter['count'] = 0
            return
        self.counter['count'] += 1

    def __del__(self):
        end = self.time.time()
        start = self.counter['start']
        count = self.counter['count']
        if start is not None:
            print(
                f'{type(self).__name__} calculated frame rate is {count / (end - start)}'
            )
        self.cv2.destroyWindow(self.WINDOW_NAME)

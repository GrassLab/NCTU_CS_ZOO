from drivers.camera import CameraDevice
import cv2


class CameraControl(object):
    def __init__(self, img_fmt='bgr'):
        self.cam = CameraDevice(img_fmt)
        self.data = {}

    def get_capture_img(self):
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

    def set_cam_calibration(self, cam_K, cam_D, img_h, img_w):
        self.data['cam_K'] = cam_K
        self.data['cam_D'] = cam_D
        self.data['cam_K_new'], self.data['cam_ROI_new'] = cv2.getOptimalNewCameraMatrix(cam_K, cam_D, (img_w, img_h),
                                                                                         0, (img_w, img_h))
        self.data['rect_mapx'], self.data['rect_mapy'] = cv2.initUndistortRectifyMap(cam_K, cam_D, None,
                                                                                     self.data['cam_K_new'],
                                                                                     (img_w, img_h), 5)
        self.data['cam_calib'] = True

    def get_rectified_image(self, img=None):
        assert self.data['cam_calib'], 'Call set_cam_calibration to set calibration data '
        if img is None:
            img = self.get_capture_img()
        dst = cv2.remap(img, self.data['rect_mapx'], self.data['rect_mapy'], cv2.INTER_CUBIC)

        # # crop the image
        x, y, w, h = self.data['cam_ROI_new']
        dst_crop = dst[y:y + h, x:x + w]
        return dst_crop

    def __del__(self):
        self.stop_stream()


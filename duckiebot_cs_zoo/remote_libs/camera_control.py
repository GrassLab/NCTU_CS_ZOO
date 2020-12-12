import os
import cv2
import joblib

from utils.network import BasicClient
from nodes.msg_types import CmdStr, FunctionCall


class CameraControl(object):
    client: BasicClient = None

    def __init__(self, server_port: int = 54761, img_fmt='bgr'):
        if CameraControl.client is None:
            duckiebot = BasicClient().start_connection(
                server_addr=os.getenv('DUCKIE', ''),
                server_port=server_port,
            )
            ports = duckiebot.send_data(CmdStr('get_ports')).recv_data()
            duckiebot.stop_connection()
            camera_port = ports['cam']
            CameraControl.client = BasicClient().start_connection(
                server_addr=os.getenv('DUCKIE', ''),
                server_port=camera_port,
            )
        getattr(self, f"set_{img_fmt}")()
        self.data = {}
        self.decode_jpeg_fmt = self.decode_jpeg()
        self.cur_fmt = self.get_fmt()

    def check_decode(self, img): # Local decoding jpeg
        if self.cur_fmt == 'jpeg' and not self.decode_jpeg_fmt:
            return cv2.imdecode(img, 1)
        else:
            return img

    def get_capture_img(self):  # Better image quality
        fc = FunctionCall('get_capture_img')
        return self.check_decode(CameraControl.client.send_data(fc).recv_data())

    def start_stream(self):
        fc = FunctionCall('start_stream')
        return CameraControl.client.send_data(fc).recv_data()

    def stop_stream(self):
        fc = FunctionCall('stop_stream')
        return CameraControl.client.send_data(fc).recv_data()

    def get_stream_img(self):
        fc = FunctionCall('get_stream_img')
        return self.check_decode(CameraControl.client.send_data(fc).recv_data())

    def set_jpeg(self):
        self.cur_fmt = 'jpeg'
        fc = FunctionCall('set_jpeg')
        return CameraControl.client.send_data(fc).recv_data()

    def set_bgr(self):
        self.cur_fmt = 'bgr'
        fc = FunctionCall('set_bgr')
        return CameraControl.client.send_data(fc).recv_data()

    def get_fmt(self):
        fc = FunctionCall('get_fmt')
        return CameraControl.client.send_data(fc).recv_data()

    def decode_jpeg(self):
        fc = FunctionCall('decode_jpeg')
        return CameraControl.client.send_data(fc).recv_data()

    def set_cam_calibration(self, cam_K, cam_D, img_h, img_w):
        self.data['cam_K'] = cam_K
        self.data['cam_D'] = cam_D
        self.data['cam_K_new'], self.data['cam_ROI_new'] = cv2.getOptimalNewCameraMatrix(cam_K, cam_D, (img_w, img_h),
                                                                                         0, (img_w, img_h))
        self.data['rect_mapx'], self.data['rect_mapy'] = cv2.initUndistortRectifyMap(cam_K, cam_D, None,
                                                                                     self.data['cam_K_new'],
                                                                                     (img_w, img_h), 5)
        self.data['cam_calib'] = True

    def set_cam_extrinsics(self, H, carTcam):
        self.data['cam_H'] = H
        self.data['cam_carTcam'] = carTcam

    def get_rectified_image(self, img=None):
        assert self.data['cam_calib'], 'Call set_cam_calibration to set calibration data '
        if img is None:
            img = self.get_stream_img()
        dst = cv2.remap(img, self.data['rect_mapx'], self.data['rect_mapy'], cv2.INTER_CUBIC)

        # # crop the image
        x, y, w, h = self.data['cam_ROI_new']
        dst_crop = dst[y:y + h, x:x + w]
        return dst_crop

    @property
    def cam_mat(self):
        return self.data['cam_K_new']

    def save_data(self, filepath):
        joblib.dump(self.data, filepath)

    def restore_data(self, filepath):
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            assert data['cam_calib'], 'Please Re-Run calibration on your duckiebot'
            assert data['cam_carTcam'] is not None, "Make sure you run the extrinsic_calibration_pose"

            self.data = data
            print('Camera Data Restored')
        else:
            print('Camera Data Not Available, Please manually copy the camera_data.pkl from your duckiebot')

    def __del__(self):
        pass

from utils.network import BasicServer
from libs.camera_control import CameraControl
from nodes.msg_types import FunctionCall


class CameraServer(BasicServer):
    cam: CameraControl = None

    def __init__(self, addr='0.0.0.0', port=54760, img_fmt='bgr'):
        self.cam = CameraControl(img_fmt)
        super(CameraServer, self).__init__(addr, port)
        print('Camera Server: {}:{}'.format(*self.server_address))

    def handle_received_obj(self, recv_obj, handler):
        if isinstance(recv_obj, FunctionCall):
            func = getattr(self.cam, recv_obj.name)
            return func(*recv_obj.args, **recv_obj.kwargs)
        elif isinstance(recv_obj, int):
            if recv_obj == -1:  # Shutdown cleanup
                self.cam.stop_stream()


def main():
    cam_server = CameraServer(port=12345)  # 0 for automatically port selection
    print('Cam server, address={}'.format(cam_server.server_address))
    cam_server.start_server()


if __name__ == '__main__':
    main()

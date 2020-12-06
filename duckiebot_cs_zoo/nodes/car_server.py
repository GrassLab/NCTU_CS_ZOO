from utils.network import BasicServer
from nodes.msg_types import FunctionCall
from libs.car_control import CarControl


class CarServer(BasicServer):
    car = CarControl(0.0)

    def handle_received_obj(self, recv_obj, handler):
        if isinstance(recv_obj, FunctionCall):
            func = getattr(self.car, recv_obj.name)
            return func(*recv_obj.args, **recv_obj.kwargs)

    def finish_handler(self, handler):
        # prevent client exit but setting is remain
        self.car.move(velocity=0, omega=0)

from utils.network import BasicServer
from nodes.msg_types import FunctionCall
from libs.light_control import LightControl


class LightServer(BasicServer):
    light = LightControl()

    def handle_received_obj(self, recv_obj, handler):
        if isinstance(recv_obj, FunctionCall):
            func = getattr(self.light, recv_obj.name)
            return func(*recv_obj.args, **recv_obj.kwargs)

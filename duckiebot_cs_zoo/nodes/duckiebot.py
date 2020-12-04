import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.network import BasicServer
from nodes.msg_types import CmdStr
from nodes.camera_server import CameraServer
from nodes.light_server import LightServer
from nodes.car_server import CarServer
from multiprocessing import Array, Process
import time


def launch_server(server_obj, port_arr, s_idx, **kwargs):
    print('Recv arguments: {},{},{},{}'.format(server_obj, port_arr, s_idx, kwargs))
    server = server_obj(**kwargs)
    port = server.server_address[1]
    port_arr[s_idx] = port
    server.start_server()


class DuckiebotMainServer(BasicServer):
    def __init__(self, addr='0.0.0.0', port=54761):
        port = int(os.getenv('PORT', port))
        super(DuckiebotMainServer, self).__init__(addr, port)  # Command server
        """
        Only DuckiebotMainServer uses the selected port, other nodes use auto port assignment (port=0)
        
        Servers:
        cam_server: camera images
        light_server: RGB-LED lights
        car_server: Motor control & robot pose(2D planar) estimation & robot kinematics(fwd/inv) for wheel command
        
        Launch the servers using the python multiprocessing library
        DuckiebotMainServer communicates to other nodes using shared Array before launch (collects port information)
        """
        self.server_ports = {}

        ports = Array('l', [-1] * 3)
        self.process = {}
        self.process['cam'] = Process(target=launch_server, args=(CameraServer, ports, 0),
                                      kwargs={'addr': '0.0.0.0', 'port': 0, 'img_fmt': 'jpeg'})
        self.process['light'] = Process(target=launch_server, args=(LightServer, ports, 1),
                                        kwargs={'addr': '0.0.0.0', 'port': 0})
        self.process['car'] = Process(target=launch_server, args=(CarServer, ports, 2),
                                      kwargs={'addr': '0.0.0.0', 'port': 0})
        for p in self.process.values():
            p.start()

        while (np.array(ports) == -1).sum() != 0:
            print('Wait for ports')
            time.sleep(0.2)
        print('Duckietbot driver servers launched, ports:{}. Main port:{}'.format(np.array(ports),
                                                                                  self.server_address[1]))
        self.server_ports['cam'] = ports[0]
        self.server_ports['light'] = ports[1]
        self.server_ports['car'] = ports[2]

    def handle_received_obj(self, recv_obj, handler):
        if isinstance(recv_obj, CmdStr):
            if recv_obj.cmd == 'get_ports':
                return self.server_ports


def launch_duckiebot_cs_zoo():
    # launch all driver servers
    # check the ports and make sure all are launched
    # user connect to a main server to get all other server information
    duckiebot = DuckiebotMainServer()
    duckiebot.start_server()


if __name__ == '__main__':
    launch_duckiebot_cs_zoo()

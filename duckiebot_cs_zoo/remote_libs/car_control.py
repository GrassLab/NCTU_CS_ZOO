from typing import Tuple
import os

from utils.network import BasicClient
from nodes.msg_types import CmdStr, FunctionCall


class CarControl:
    """Integrate control for car

    Args:

        `trim_val` (:obj:`float`): motor calibration parameter, default is 0.0
    """
    client: BasicClient = None

    def __init__(self, server_port: int = 54761, trim_val: float = 0.0):
        if CarControl.client is None:
            duckiebot = BasicClient().start_connection(
                server_addr=os.getenv('DUCKIE', ''),
                server_port=server_port,
            )
            ports = duckiebot.send_data(CmdStr('get_ports')).recv_data()
            duckiebot.stop_connection()
            car_port = ports['car']
            CarControl.client = BasicClient().start_connection(
                server_addr=os.getenv('DUCKIE', ''),
                server_port=car_port,
            )
        # TODO: set trim val
        pass

    def move(self, velocity: float, omega: float):
        """
        Set velocity and omega of car

        Args:

            velocity (:obj:`float`): positive value is forward, negative value is backword

            omega (:obj:`float`): positive value is turn left, negative value is turn right
        """
        fc = FunctionCall('move', kwargs=dict(
            velocity=velocity,
            omega=omega,
        ))
        return CarControl.client.send_data(fc).recv_data()

    def get_pose(self) -> Tuple[int, float, float, float]:
        """
        Get estimated pose for car

        Return:

            :obj:`Tuple` of (:obj:`int`, :obj:`float`, :obj:`float`, :obj:`float`)

            time for ns unit (:obj:`int`), X (:obj:`float`), Y (:obj:`float`), theta (:obj:`float`)
        """
        fc = FunctionCall('get_pose')
        return CarControl.client.send_data(fc).recv_data()

    def set_trim(self, val: float):
        """
        Set Trim value

        Args:
           val (:obj:`float`): Trim value
        """
        fc = FunctionCall('set_trim', args=[val])
        return CarControl.client.send_data(fc).recv_data()

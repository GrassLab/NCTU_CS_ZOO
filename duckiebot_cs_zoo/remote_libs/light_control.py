from enum import Enum
from typing import Union
import os

from utils.network import BasicClient
from nodes.msg_types import CmdStr, FunctionCall


class LightControl:
    """control for light

    Example:

    .. code:: python

       light = LightControl()
       # set all light to red color
       light.change_rgb("ALL", red=1, green=0, blue=0)
       # only set front and left light to green
       light.change_rgb("FRONT_LEFT", blue=0, green=1, red=0)

    """
    client: BasicClient = None

    class LightPos(Enum):
        FRONT_LEFT = 0
        REAR_LEFT = 1
        FRONT_MIDDLE = 2
        REAR_RIGHT = 3
        FRONT_RIGHT = 4
        ALL = 5

    def __init__(self, server_port: int = 54761):
        if LightControl.client is None:
            duckiebot = BasicClient().start_connection(
                server_addr=os.getenv('DUCKIE', ''),
                server_port=server_port,
            )
            ports = duckiebot.send_data(CmdStr('get_ports')).recv_data()
            duckiebot.stop_connection()
            light_port = ports['light']
            LightControl.client = BasicClient().start_connection(
                server_addr=os.getenv('DUCKIE', ''),
                server_port=light_port,
            )

    def change_rgb(
            self,
            light_pos: Union[LightPos, int], *,
            red: float, green: float, blue: float,
    ):
        """
        Change red, green and blue brightness of speific light

        Args:

            `light_pos` (:obj:`LightPos` or :obj:`int` or :obj:`str`): indicate which light to control
            e.g. `"FRONT_LEFT"` or `0` or `LightControl.LightPos.FRONT_LEFT`

            red (:obj:`float`): red brightness between 0 and 1

            green (:obj:`float`): green brightness between 0 and 1

            blue (:obj:`float`): blue brightness between 0 and 1
        """
        if not isinstance(light_pos, self.LightPos):
            if isinstance(light_pos, int):
                light_pos = self.LightPos(light_pos)
            elif isinstance(light_pos, str):
                light_pos = self.LightPos[light_pos]
            else:
                raise ValueError(
                    f"light_pos should be one of {str}, {int} and {self.LightPos}"
                )

        fc = FunctionCall('change_rgb', kwargs=dict(
            # use int to prevent pickle use wrong type
            light_pos=light_pos.value,
            red=red, green=green, blue=blue,
        ))
        return LightControl.client.send_data(fc).recv_data()

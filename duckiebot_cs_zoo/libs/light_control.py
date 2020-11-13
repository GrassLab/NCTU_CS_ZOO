from enum import Enum, Flag
from typing import Union

try:
    from drivers.rgb_led import RGB_LED
except ImportError:
    from ..drivers.rgb_led import RGB_LED


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
    rgb_led_driver: RGB_LED = None

    class LightPos(Enum):
        FRONT_LEFT = 0
        REAR_LEFT = 1
        FRONT_MIDDLE = 2
        REAR_RIGHT = 3
        FRONT_RIGHT = 4
        ALL = 5

    def __init__(self):
        self.rgb_led_driver = RGB_LED()

    def __del__(self):
        del self.rgb_led_driver

    def change_rgb(
        self,
        light_pos: Union[LightPos, int], *,
        red: float, green: float, blue: Flag,
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

        if light_pos is self.LightPos.ALL:
            lights = [
                self.LightPos.FRONT_LEFT,
                self.LightPos.FRONT_MIDDLE,
                self.LightPos.FRONT_RIGHT,
                self.LightPos.REAR_LEFT,
                self.LightPos.REAR_RIGHT,
            ]
        else:
            lights = [light_pos]
        for pos in lights:
            self.rgb_led_driver.setRGB(
                led=pos.value,
                color=(red, green, blue),
            )

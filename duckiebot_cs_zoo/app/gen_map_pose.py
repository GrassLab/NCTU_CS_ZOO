import numpy as np
import math3d as m3d


# Static axis, Euler angle in degree, translation (x, y) in pixel, translation (z) in meter
# key:id, value:([rx,ry,rz],[px,py],tz)
tag_spec = {
    # left signs (face forward)
    1: ([90, 0, 0], [100, 171], 0.128),
    2: ([90, 0, 0], [655, 171], 0.128),
    3: ([90, 0, 0], [1210, 171], 0.128),
    4: ([90, 0, 0], [1765, 171], 0.128),
    5: ([90, 0, 0], [2320, 171], 0.128),
    6: ([90, 0, 0], [2875, 171], 0.128),

    # 1st parking space (face upward)
    7: ([0, 0, 0], [311, 1036], 0),
    8: ([0, 0, 0], [311, 1175], 0),
    9: ([0, 0, 0], [141, 1984], 0),
    10: ([0, 0, 0], [311, 1984], 0),
    11: ([0, 0, 0], [481, 1984], 0),

    # 1st parking signs (face leftward)
    12: ([90, 0, -90], [141, 2182], 0.128),
    13: ([90, 0, -90], [452, 2182], 0.128),

    # 1st-2nd parking space (face upward)
    14: ([0, 0, 0], [849.5, 1036], 0),
    15: ([0, 0, 0], [849.5, 1175], 0),

    # 2nd parking space (face upward)
    16: ([0, 0, 0], [1388, 1036], 0),
    17: ([0, 0, 0], [1388, 1175], 0),
    18: ([0, 0, 0], [1218, 1984], 0),
    19: ([0, 0, 0], [1388, 1984], 0),
    20: ([0, 0, 0], [1558, 1984], 0),

    # 2nd parking signs (face leftward)
    21: ([90, 0, -90], [1218, 2182], 0.128),
    22: ([90, 0, -90], [1529, 2182], 0.128),

    # 2nd-3rd parking space (face upward)
    23: ([0, 0, 0], [1926.5, 1036], 0),
    24: ([0, 0, 0], [1926.5, 1175], 0),

    # 3rd parking space (face upward)
    25: ([0, 0, 0], [2465, 1036], 0),
    26: ([0, 0, 0], [2465, 1175], 0),
    27: ([0, 0, 0], [2295, 1984], 0),
    28: ([0, 0, 0], [2465, 1984], 0),
    29: ([0, 0, 0], [2635, 1984], 0),

    # 3rd parking signs (face leftward)
    30: ([90, 0, -90], [2295, 2182], 0.128),
    31: ([90, 0, -90], [2606, 2182], 0.128),

    # back signs (face forward)
    32: ([90, 0, 0], [3031, 1050], 0.128),
    33: ([90, 0, 0], [3031, 1333], 0.128),
    34: ([90, 0, 0], [3031, 1616], 0.128),
    35: ([90, 0, 0], [3031, 1899], 0.128),
}


def pixel2tag_coord(rot, pixel_coord, tag_z, dpi=72, tag_size=0.04, sign_size=(0.05, 0.055)):
    """
    Args:
        rot:
        pixel_coord:
        tag_z: (m)
        dpi: dot per inch
        tag_size: (m)
        sign_size: (m)
    length[mm] = pixel * 25.4mm (1 in) / dpi

    Returns: list
    """
    if rot == [90, 0, 0]:
        center_shift = [sign_size[0]/2, sign_size[1]/2]
    elif rot == [90, 0, -90]:
        center_shift = [sign_size[1]/2, sign_size[0]/2]
    else:
        center_shift = [tag_size/2, tag_size/2]

    m_coord = np.array(pixel_coord) * 25.4 * 1e-3 / dpi  # pixels to meters, and get the tag center
    tag_coord = [m_coord[1] + center_shift[0], m_coord[0] + center_shift[1]]
    tag_coord.append(tag_z)

    return tag_coord


def gen_challenge10_mapping(tag_spec):
    mapping = {}

    for t_id, (rot, pxy, trz) in tag_spec.items():
        tr = pixel2tag_coord(rot, pxy, trz)
        oTtag = m3d.Transform(m3d.Orientation.new_euler(np.deg2rad(rot), 'xyz'), tr)
        mapping[t_id] = oTtag.array

    return mapping
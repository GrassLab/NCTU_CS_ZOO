import numpy as np
import cv2
import io

nothing = lambda x: None


def fig2img(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_img(img, title='Show Image', wait_key=0, destroy=True):
    cv2.imshow(title, img)
    k = cv2.waitKey(wait_key)
    if destroy:
        cv2.destroyWindow(title)
    return chr(k) if k > 0 else None


class RangeHSV:
    def __init__(self, hsv_ranges):
        """
        :param hsv_ranges: "List" of Numpy array with size 2x3 that represents [[h_low,s_low,v_low],[h_high,s_high,v_high]]
        """
        if isinstance(hsv_ranges, list):
            self.hsv_ranges = hsv_ranges
            self.n_ranges = len(hsv_ranges)
        elif isinstance(hsv_ranges, np.ndarray):
            self.hsv_ranges = [hsv_ranges]
            self.n_ranges = 1

    def hsv_in_range(self, hsv_img):
        select = cv2.inRange(hsv_img, self.hsv_ranges[0][0], self.hsv_ranges[0][1])
        for i in range(1, self.n_ranges):
            cur_select = cv2.inRange(hsv_img, self.hsv_ranges[i][0], self.hsv_ranges[i][1])
            select = np.bitwise_or(select, cur_select)
        return select  # uint8, but contains either 0 or 255

    @classmethod
    def from_vis(cls, img):
        cv2.namedWindow('HSV Filter')
        cv2.createTrackbar('h_low', 'HSV Filter', 0, 255, nothing)
        cv2.createTrackbar('h_high', 'HSV Filter', 255, 255, nothing)
        cv2.createTrackbar('s_low', 'HSV Filter', 0, 255, nothing)
        cv2.createTrackbar('s_high', 'HSV Filter', 255, 255, nothing)
        cv2.createTrackbar('v_low', 'HSV Filter', 0, 255, nothing)
        cv2.createTrackbar('v_high', 'HSV Filter', 255, 255, nothing)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print('Press p to show, r to record, q to quit')
        ranges = []
        while True:
            h_low = cv2.getTrackbarPos('h_low', 'HSV Filter')
            h_high = cv2.getTrackbarPos('h_high', 'HSV Filter')
            s_low = cv2.getTrackbarPos('s_low', 'HSV Filter')
            s_high = cv2.getTrackbarPos('s_high', 'HSV Filter')
            v_low = cv2.getTrackbarPos('v_low', 'HSV Filter')
            v_high = cv2.getTrackbarPos('v_high', 'HSV Filter')
            hsv_low = np.array([h_low, s_low, v_low])
            hsv_high = np.array([h_high, s_high, v_high])
            mask = cv2.inRange(hsv_img, hsv_low, hsv_high)
            vis_img = img.copy()
            vis_img[mask == 0, :] = 0
            cv2.imshow('HSV Filter', vis_img)
            k = cv2.waitKey(30)
            cur_range = np.vstack([hsv_low, hsv_high])  # 2x3
            if k == ord('q'):
                if len(ranges) > 0:
                    break
                else:
                    print('Please at least press \'r\' once to record a valid range')
            elif k == ord('p'):
                print('Current Range=\n{}'.format(cur_range))
            elif k == ord('r'):
                print('Current range added')
                ranges.append(cur_range)
        cv2.destroyWindow('HSV Filter')
        return cls(ranges)


def color_balance(img, percent=70, clip_low=True, clip_high=True,
                  image_height_clip_ratio=0.3, ):  # "Illumination compensation" in the duckietown paper
    """
    Source: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
    :param img: H,W,3 uint8 image
    :param percent: percentage in (0,100), e.g., percent=30, then 15% is clipped for low value and 15% is clipped for high value
    :param clip_high: If false, not perform high-value clip (hv is a constant=255)
    :param clip_low: If false, not perform low-value clip (lv is a constant=0)
                    These value may depends on you ground condition
                        e.g., for dark background, clip_low=True, clip_high=False
                        e.g., for light background, clip_low=False, clip_high=True
    :param image_height_clip_ratio: Useful for only considering the "ground pixels" for threshold calculation
    :return: balanced_img
    """
    out_channels = []
    if image_height_clip_ratio != 0:
        h_start = int(img.shape[0] * image_height_clip_ratio)
        img_down = img[h_start:, :, :]
    else:
        img_down = img

    down = cv2.resize(img_down, (0, 0), fx=0.25, fy=0.25)
    down_channels = cv2.split(down)  # Down-sampled (with height_clip) image
    img_channels = cv2.split(img)  # Original image
    total_stop = down_channels[0].shape[0] * down_channels[0].shape[1] * percent / 200.0
    for down_c, img_c in zip(down_channels, img_channels):
        bc = cv2.calcHist([down_c], [0], None, [256], (0, 256), accumulate=False)
        if clip_low:
            lv = np.searchsorted(np.cumsum(bc), total_stop)
        else:
            lv = 0
        if clip_high:
            hv = 255 - np.searchsorted(np.cumsum(bc[::-1]), total_stop)
        else:
            hv = 255
        lut = np.array(
            [0 if i < lv else (255 if i > hv else round(float(i - lv) / float(hv - lv) * 255)) for i in
             np.arange(0, 256)], dtype="uint8")
        out_channels.append(cv2.LUT(img_c, lut))  # Use original image
    return cv2.merge(out_channels)


# Version 2
def calculate_color_balance(img, percent=70, clip_low=True, clip_high=True,
                            image_height_clip_ratio=0.3, ):  # "Illumination compensation" in the duckietown paper
    """
    Source: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
    :param img: H,W,3 uint8 image
    :param percent: percentage in (0,100), e.g., percent=30, then 15% is clipped for low value and 15% is clipped for high value
    :param clip_high: If false, not perform high-value clip (hv is a constant=255)
    :param clip_low: If false, not perform low-value clip (lv is a constant=0)
                   These value may depends on you ground condition
                       e.g., for dark background, clip_low=True, clip_high=False
                       e.g., for light background, clip_low=False, clip_high=True
    :param image_height_clip_ratio: Useful for only considering the "ground pixels" for threshold calculation
    :return: lut_channels: [lut_b,lut_g,lut_r] look-up tables for all colors
    """
    lut_channels = []
    if image_height_clip_ratio != 0:
        h_start = int(img.shape[0] * image_height_clip_ratio)
        img_down = img[h_start:, :, :]
    else:
        img_down = img

    down = cv2.resize(img_down, (0, 0), fx=0.25, fy=0.25)
    down_channels = cv2.split(down)  # Down-sampled (with height_clip) image
    img_channels = cv2.split(img)  # Original image
    total_stop = down_channels[0].shape[0] * down_channels[0].shape[1] * percent / 200.0
    for down_c, img_c in zip(down_channels, img_channels):
        bc = cv2.calcHist([down_c], [0], None, [256], (0, 256), accumulate=False)
        if clip_low:
            lv = np.searchsorted(np.cumsum(bc), total_stop)
        else:
            lv = 0
        if clip_high:
            hv = 255 - np.searchsorted(np.cumsum(bc[::-1]), total_stop)
        else:
            hv = 255
        lut = np.array(
            [0 if i < lv else (255 if i > hv else round(float(i - lv) / float(hv - lv) * 255)) for i in
             np.arange(0, 256)], dtype="uint8")
        lut_channels.append(lut)
    return lut_channels


def apply_color_balance(image, lut_channels):
    img_channels = cv2.split(image)  # Original image
    out_channels = []
    for lut, img_c in zip(lut_channels, img_channels):
        out_channels.append(cv2.LUT(img_c, lut))
    return cv2.merge(out_channels)


def find_edges(img, canny_thresh1=80, canny_thresh2=200, canny_aperture_size=3, vis=False):
    # Check 'Hysteresis Thresholding' on https://docs.opencv.org/master/d7/de1/tutorial_js_canny.html
    if not vis:
        edges = cv2.Canny(img, canny_thresh1, canny_thresh2, apertureSize=canny_aperture_size)
        return edges  # binary map (either 0 or 255)
    else:
        cv2.namedWindow('Canny Edges')
        cv2.createTrackbar('Canny Thresh1', 'Canny Edges', 80, 500, nothing)
        cv2.createTrackbar('Canny Thresh2', 'Canny Edges', 200, 500, nothing)
        cv2.createTrackbar('Aperture size(3-7)', 'Canny Edges', 3, 7, nothing)
        print('Press p to show params, q to quit')
        while True:
            t1 = cv2.getTrackbarPos('Canny Thresh1', 'Canny Edges')
            t2 = cv2.getTrackbarPos('Canny Thresh2', 'Canny Edges')
            apt_sz = np.clip(cv2.getTrackbarPos('Aperture size', 'Canny Edges'), 3, 7)
            edges = cv2.Canny(img, t1, t2, apertureSize=apt_sz)
            vis = np.hstack([img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])
            cv2.imshow('Canny Edges', vis)
            k = cv2.waitKey(30)
            if k == ord('q'):
                break
            elif k == ord('p'):
                print('Canny Params: t1={},t2={},Aperture size={}'.format(t1, t2, apt_sz))
        cv2.destroyWindow('Canny Edges')
        print('Canny Params: t1={},t2={},Aperture size={}'.format(t1, t2, apt_sz))
        return edges


def dilation(binary_map, kernel=cv2.MORPH_ELLIPSE, kernel_size=3, vis=False):
    if not vis:
        kernel = cv2.getStructuringElement(kernel, (kernel_size, kernel_size))
        return cv2.dilate(binary_map, kernel)
    else:

        cv2.namedWindow('Dilation')
        # MORPH_RECT = 0
        # MORPH_CROSS = 1
        # MORPH_ELLIPSE = 2
        kernel_names = ['MORPH_RECT', 'MORPH_CROSS', 'MORPH_ELLIPSE']
        cv2.createTrackbar('Kernel', 'Dilation', 2, 2, nothing)
        cv2.createTrackbar('Kernel size', 'Dilation', 3, 20, nothing)
        print('Press p to show params, q to quit')
        while True:
            k_type = cv2.getTrackbarPos('Kernel', 'Dilation')
            k_size = max(3, cv2.getTrackbarPos('Kernel size', 'Dilation'))
            kernel = cv2.getStructuringElement(k_type, (k_size, k_size))
            dil_map = cv2.dilate(binary_map, kernel)
            vis = np.hstack([binary_map, dil_map])
            cv2.imshow('Dilation', vis)
            k = cv2.waitKey(30)
            if k == ord('q'):
                break
            elif k == ord('p'):
                print('Dilation Params: Kernel={},Kernel size={}'.format(kernel_names[k_type], k_size))
        cv2.destroyWindow('Dilation')
        print('Dilation Params: Kernel={},Kernel size={}'.format(kernel_names[k_type], k_size))
        return dil_map


def hough_lines(edges_binary, rho=1, theta_deg=1, threshold=2, minLineLength=3, maxLineGap=1, vis=False):
    if not vis:
        lines = cv2.HoughLinesP(edges_binary, rho=rho, theta=np.deg2rad(theta_deg), threshold=threshold,
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
    else:
        is_changed = True

        def do_change(x):
            nonlocal is_changed
            is_changed = True

        edge_color = cv2.cvtColor(edges_binary, cv2.COLOR_GRAY2BGR)
        cv2.namedWindow('Hough Lines')
        cv2.createTrackbar('rho', 'Hough Lines', 1, 100, do_change)  # Distance resolution (pixels)
        cv2.createTrackbar('theta_deg', 'Hough Lines', 1, 180,
                           do_change)  # Angle resolution (convert to radian later)
        cv2.createTrackbar('threshold', 'Hough Lines', 2, 100, do_change)  # Accumulator threshold (vote count)
        cv2.createTrackbar('minLineLength', 'Hough Lines', 3, 50, do_change)
        cv2.createTrackbar('maxLineGap', 'Hough Lines', 1, 50, do_change)
        print('Press p to show params, q to quit')
        while True:
            rho = max(1, cv2.getTrackbarPos('rho', 'Hough Lines'))
            theta_deg = max(0.1, cv2.getTrackbarPos('theta_deg', 'Hough Lines'))
            threshold = cv2.getTrackbarPos('threshold', 'Hough Lines')
            minLineLength = cv2.getTrackbarPos('minLineLength', 'Hough Lines')
            maxLineGap = cv2.getTrackbarPos('maxLineGap', 'Hough Lines')
            if is_changed:
                lines = cv2.HoughLinesP(edges_binary, rho=rho, theta=np.deg2rad(theta_deg), threshold=threshold,
                                        minLineLength=minLineLength, maxLineGap=maxLineGap)
                vis_lines = edge_color.copy()
                print(f'Len of line={len(lines)}')
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(vis_lines, (x1, y1), (x2, y2), np.random.randint(128, 256, 3).tolist(), 2)
                is_changed = False
            vis = np.hstack([edge_color, vis_lines])
            cv2.imshow('Hough Lines', vis)
            k = cv2.waitKey(30)
            if k == ord('q'):
                break
            elif k == ord('p'):
                print('Hough Params: rho={},theta_deg={},threshold={},minLineLength={},maxLineGap={}'.format(
                    rho, theta_deg, threshold, minLineLength, maxLineGap
                ))
        cv2.destroyWindow('Hough Lines')
        print('Hough Params: rho={},theta_deg={},threshold={},minLineLength={},maxLineGap={}'.format(
            rho, theta_deg, threshold, minLineLength, maxLineGap
        ))
    if lines is not None:
        lines = lines.reshape((-1, 4))  # it has an extra dimension
    else:
        lines = []
    return lines


def calc_normal_(in_range_map, lines, extend_l):
    H, W = in_range_map.shape
    x1 = lines[:, [0]]  # Nx1
    y1 = lines[:, [1]]
    x2 = lines[:, [2]]
    y2 = lines[:, [3]]
    t = np.hstack([x2 - x1, y2 - y1])  # N,2
    t = t / np.linalg.norm(t, axis=1).reshape(-1, 1)
    mid = np.hstack([(x1 + x2) / 2, (y1 + y2) / 2])  # N,2 (mid_poitn)
    normal = np.hstack([-t[:, [1]], t[:, [0]]])  # => type1 (t CCW 90 degree -> +Z 90 degree)
    mid_extend_1 = np.round(mid + extend_l * normal).astype(np.int)
    mid_extend_2 = np.round(mid - extend_l * normal).astype(np.int)
    # Clip
    mid_extend_1[:, 0] = np.clip(mid_extend_1[:, 0], 0, W - 1)
    mid_extend_1[:, 1] = np.clip(mid_extend_1[:, 1], 0, H - 1)
    mid_extend_2[:, 0] = np.clip(mid_extend_2[:, 0], 0, W - 1)
    mid_extend_2[:, 1] = np.clip(mid_extend_2[:, 1], 0, H - 1)
    # Check if it's within a color chunk
    inside_color_1 = in_range_map[mid_extend_1[:, 1], mid_extend_1[:, 0]]
    inside_color_2 = in_range_map[mid_extend_2[:, 1], mid_extend_2[:, 0]]
    valid_lines = np.logical_xor(inside_color_1 == 255,
                                 inside_color_2 == 255)  # Must be either side of a chunk, modify to logical_or to see why
    lines = lines[valid_lines]  # M,2
    normal = normal[valid_lines]  # M,2
    # Flip normal direction
    normal[inside_color_2[valid_lines] == 255] *= -1  # => type2 (t CW 90 degree -> -Z 90 degree)
    # Normal type (1 or 2)
    normal_type = np.ones(normal.shape[0], dtype=np.uint8)
    normal_type[inside_color_2[valid_lines] == 255] = 2
    return lines, normal, valid_lines, normal_type


def calc_normal(in_range_map, lines, extend_l=5, vis=False):
    # The normal vector is "heading toward its corresponding color chunk"
    if not vis:
        lines, normal, _, normal_type = calc_normal_(in_range_map, lines, extend_l)
        return lines, normal, normal_type
    else:
        is_changed = True

        def do_change(x):
            nonlocal is_changed
            is_changed = True

        cv2.namedWindow('Normals')
        cv2.createTrackbar('extend_l', 'Normals', extend_l, 20, do_change)  # Length (pixel unit)
        print('Press p to show params, q to quit')
        while True:
            extend_l = max(1, cv2.getTrackbarPos('extend_l', 'Normals'))
            if is_changed:
                lines_new, normal, valid_lines, normal_type = calc_normal_(in_range_map, lines, extend_l)
                print('Valid Normal={}/{}'.format(valid_lines.sum(), len(valid_lines)))
                UP_SCALE = 2
                EXTEND_VIS = max(5, extend_l)
                vis_map = cv2.resize(cv2.cvtColor(in_range_map, cv2.COLOR_GRAY2BGR), (0, 0), fx=UP_SCALE, fy=UP_SCALE)
                for l, n in zip(lines_new, normal):
                    mid = (np.round((l[0:2] + l[2:]) / 2.) * UP_SCALE).astype(np.int)
                    mid_extend = np.round(mid + n * EXTEND_VIS * UP_SCALE).astype(np.int)
                    rand_color = np.random.randint(64, 255, 3).tolist()
                    cv2.arrowedLine(vis_map, tuple(mid),
                                    tuple(mid_extend), rand_color,
                                    thickness=2,
                                    line_type=8,
                                    tipLength=0.5)
                    cv2.line(vis_map, tuple((l[0:2] * UP_SCALE).astype(np.int)),
                             tuple((l[2:] * UP_SCALE).astype(np.int)), rand_color, 2)
                is_changed = False
            cv2.imshow('Normals', vis_map)
            k = cv2.waitKey(30)
            if k == ord('q'):
                break
            elif k == ord('p'):
                print('Normal params: extend_l={}'.format(extend_l))
        cv2.destroyWindow('Normals')
        print('Normal params: extend_l={}'.format(extend_l))

        return lines_new, normal, normal_type

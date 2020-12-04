import cv2
from utils.fileutils import check_path
import os.path as osp


class VideoRecorder:
    def __init__(self, img_size, out_path, frame_rate=30, codec=None):
        """
        :param img_size: tuple of (W,H)
        :param out_path: file path (with extension)
        :param frame_rate: video framerate
        :param codec: MJPG(.avi), XVID, X264, mp4v(.mov/.mp4)
        """
        if codec is None:
            ext = osp.splitext(out_path)
            if ext in ['mov', 'mp4']:
                codec = 'mp4v'
            elif ext == 'avi':
                codec = 'MJPG'
            else:
                codec = 'mp4v'  # try

        fourcc = cv2.VideoWriter_fourcc(*codec)
        check_path(osp.dirname(out_path))
        self.writer = cv2.VideoWriter(out_path, fourcc, frame_rate, img_size)
        self.is_done = False
        self.out_path = out_path

    @classmethod
    def from_img(cls, img, out_path, frame_rate=30, codec=None):
        return cls((img.shape[1], img.shape[0]), out_path, frame_rate, codec)

    def write(self, img):
        self.writer.write(img)

    def __del__(self):
        if not self.is_done:
            self.release()
        if not osp.exists(self.out_path):
            print('Video {} write failed, change your file extension or manually assign a codec'.format(self.out_path))

    def release(self):
        self.writer.release()
        self.is_done = True


class VideoPlayer:
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        self.is_done = False

    def read(self, loop=False):
        while True:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                if loop:
                    print('Video loop')
                    self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)  # self.cap.set(CAP_PROP_POS_FRAMES, 0)
                else:
                    print('Video Ended')
                    return None

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self, f_idx):  # For a specified frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        return self.cap.read()

    def release(self):
        self.cap.release()
        self.is_done = True

    def __del__(self):
        if not self.is_done:
            self.release()

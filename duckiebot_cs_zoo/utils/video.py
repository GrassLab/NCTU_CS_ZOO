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


def demo():
    from app.remote_client import DuckiebotClient
    from utils.image_processing import show_img
    app_video_path = check_path(osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'videos'))
    duckiebot = DuckiebotClient()
    rec_path = osp.join(app_video_path, 'my_video.mov')
    rect_rec = VideoRecorder.from_img(duckiebot.get_rectified_image(), rec_path)
    print('Recording, press q to stop')
    while True:
        rect_img = duckiebot.get_rectified_image()
        rect_rec.write(rect_img)
        k = show_img(rect_img, 'Img', wait_key=20, destroy=False)
        if k == 'q':
            break
    rect_rec.release()

    # Replay video, you don't need to connect to your duckiebot anymore
    rect_play = VideoPlayer(rec_path)
    # Info
    print('Video: Len={}, FPS={}'.format(len(rect_play), rect_play.fps))
    print('Playing, press q to leave')
    while True:
        wait_time = int(1000 / rect_play.fps)
        rect_img = rect_play.read(loop=True)  # HERE-> You get your image from the video file
        k = show_img(rect_img, 'Img', wait_key=wait_time, destroy=False)
        if k == 'q':
            break
    rect_play.release()


if __name__ == '__main__':
    demo()

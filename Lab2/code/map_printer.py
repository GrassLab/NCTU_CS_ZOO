import numpy as np  # pip3 install -U opencv-contrib-python numpy
import cv2, time
import multiprocessing
from multiprocessing import Process, Queue


# You don't need to understand
def vis_map(map_list):  # 1,0,*,x
    """
    :param map_list: list of list of characters, only accept characters: 1,0,*,x
    :return: color image for cv2
    """
    block_pix_sz = 20
    map_arr = np.asarray(map_list)
    map_arr[map_arr == 'x'] = '2'  # Transfer point
    map_arr[map_arr == '*'] = '3'  # Car
    try:
        map_arr = map_arr.astype(int)
    except ValueError:
        print('Your map should only contains 1, 0, *, x, other characters not supported')
        raise ValueError
    vis = np.zeros((*map_arr.shape, 3)).astype(np.uint8)
    vis[map_arr == 0, :] = 255  # Road (white)
    # vis[map_arr == 1, 1] = 0 # Wall (black)
    vis[map_arr == 2, :] = 120  # Transfer point (gray)
    vis[map_arr == 3, 2] = 255  # Car (red)
    vis = np.repeat(vis, block_pix_sz, axis=1)
    vis = np.repeat(vis, block_pix_sz, axis=0)
    return vis


# You don't need to understand
def cv_loop(queue):
    cur_maze = [['1', '1'], ['1', '1']]
    img = vis_map(cur_maze)
    while True:
        if not queue.empty():
            obj = queue.get(True)
            if isinstance(obj, bool):
                break
            else:
                cur_maze = obj
                img = vis_map(cur_maze)  # Generate the image for visualization
        cv2.imshow('Maze', img)
        k = cv2.waitKey(10)
        if k == ord('q'):
            break


# You don't need to understand
class MapPrinter:  # Use this to replace the reprint function (with careful modification, see the main3 example)
    def __init__(self):
        import platform
        # Fix for mac only
        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn')

        self.done = False
        self.queue = Queue()
        self.show_proc = Process(target=cv_loop, args=(self.queue,))
        self.show_proc.start()

    def print_maze(self, maze):
        self.queue.put(maze)

    def finish(self):
        self.queue.put(True)
        self.show_proc.join()


# Read the following codes
def main1():
    maze = [['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
            ['1', '*', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
            ['1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1'],
            ['1', '1', '1', '0', '1', '0', '0', '1', '0', '0', '1'],
            ['1', '0', '1', '0', '0', '1', '1', '1', '1', '0', '1'],
            ['1', '1', '0', '0', '1', '0', '0', '0', '0', '0', '1'],
            ['1', '1', '1', '0', '1', '0', '0', '1', '1', '1', '1'],
            ['1', '0', '0', '0', '1', '0', '0', '0', '0', '0', '1'],
            ['1', '0', '1', '1', '0', '1', '1', '1', '1', '0', '1'],
            ['1', '0', '1', '0', 'x', '0', '0', '1', '1', '1', '1'],
            ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
            ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']]
    cur_car = [1, 1]
    while True:
        img = vis_map(maze)  # Generate the image for visualization
        cv2.imshow('Maze main1', img)
        k = cv2.waitKey(0)
        # 0: wait until you press a new key on your keyboard.
        # You can set 30, then it waits 30ms.
        # Question: If no input during the 30ms, what value will k be ?

        if k == ord('q'):  # If you press q then quit (Question: what is ord means?)
            break
        elif k == ord('d'):
            # If you press d, then the car moves right
            # Question: some error may happen, why?
            maze[cur_car[0]][cur_car[1]] = '0'
            cur_car[1] += 1
            maze[cur_car[0]][cur_car[1]] = '*'


def main2():
    maze1 = [['1', '1', '1', '1', '1'],
             ['1', '*', '0', '1', '1'],
             ['1', '0', '0', '0', '1'],
             ['1', '1', '1', '0', '1'],
             ['1', '0', '1', '0', '1'],
             ['1', '1', '0', 'x', '1'],
             ['1', '1', '1', '1', '1']]
    maze2 = [['1', '1', '1', '1', '1'],
             ['1', '0', '0', '1', '1'],
             ['1', '0', '0', '0', '1'],
             ['1', '0', '1', '0', '1'],
             ['1', '*', '1', '0', '1'],
             ['1', '1', '0', 'x', '1'],
             ['1', '1', '1', '1', '1']]
    maze3 = [['1', '1', '1', '1', '1'],
             ['1', '0', '0', '1', '1'],
             ['1', '0', '0', '0', '1'],
             ['1', '1', '1', '*', '1'],
             ['1', '0', '1', '0', '1'],
             ['1', '1', '0', 'x', '1'],
             ['1', '1', '1', '1', '1']]
    mazes = [maze1, maze2, maze3]

    for maze in mazes:
        img = vis_map(maze)  # Generate the image for visualization
        cv2.imshow('Maze main2', img)
        k = cv2.waitKey(0)
        if k == ord('q'):  # If you press q then quit
            break
        # Press any other key to change the maze


def main3():
    maze1 = [['1', '1', '1', '1', '1'],
             ['1', '*', '0', '1', '1'],
             ['1', '0', '0', '0', '1'],
             ['1', '1', '1', '0', '1'],
             ['1', '0', '1', '0', '1'],
             ['1', '1', '0', 'x', '1'],
             ['1', '1', '1', '1', '1']]
    maze2 = [['1', '1', '1', '1', '1'],
             ['1', '0', '0', '1', '1'],
             ['1', '0', '0', '0', '1'],
             ['1', '0', '1', '0', '1'],
             ['1', '*', '1', '0', '1'],
             ['1', '1', '0', 'x', '1'],
             ['1', '1', '1', '1', '1']]
    maze3 = [['1', '1', '1', '1', '1'],
             ['1', '0', '0', '1', '1'],
             ['1', '0', '0', '0', '1'],
             ['1', '1', '1', '*', '1'],
             ['1', '0', '1', '0', '1'],
             ['1', '1', '0', 'x', '1'],
             ['1', '1', '1', '1', '1']]

    map_printer = MapPrinter()  # 1. Initialize a MapPrinter object
    map_printer.print_maze(maze1)  # 2. Call print_maze with the full maze (e.g., after you do some modification)
    time.sleep(3)  # Don't press any key, wait and see ...
    map_printer.print_maze(maze2)
    time.sleep(3)
    map_printer.print_maze(maze3)
    time.sleep(3)

    map_printer.finish()  # 3. Stop the MapPrinter object


if __name__ == '__main__':
    print('Uncomment the function you want to try')
    #main1()  # You should understand
    #main2()  # You should understand
    #main3()  # Similar to reprint, use it if you can "feel" what it is doing or you have problem in using reprint

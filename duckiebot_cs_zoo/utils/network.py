import numpy as np
import struct, pickle
from socketserver import BaseRequestHandler, TCPServer
import socket, time
from threading import Thread
from abc import abstractmethod
from typing import Any
import io
import traceback
import atexit

HEADER_FMT_OBJECT_LEN = '<L'  # <: little endian, L:unsigned long: 4 bytes for data length
HEADER_SIZE_OBJECT_LEN = struct.calcsize(HEADER_FMT_OBJECT_LEN)


def serialize(obj: Any, buf: io.BytesIO):
    data = pickle.dumps(obj)
    header_obj_len_data = struct.pack(HEADER_FMT_OBJECT_LEN, len(data))
    buf.write(header_obj_len_data)
    buf.write(data)
    return


def deserialize(buf: io.BytesIO) -> Any:
    data = b''
    while len(data) < HEADER_SIZE_OBJECT_LEN:
        d = buf.read(HEADER_SIZE_OBJECT_LEN - len(data))
        if d == b'':
            raise EOFError
        data += d
    header_obj_len_data, data = data[:HEADER_SIZE_OBJECT_LEN], data[HEADER_SIZE_OBJECT_LEN:]
    header_obj_len = struct.unpack(HEADER_FMT_OBJECT_LEN, header_obj_len_data)[0]
    while len(data) < header_obj_len:
        d = buf.read(header_obj_len - len(data))
        if d == b'':
            raise EOFError
        data += d
    ret_obj = pickle.loads(data)
    return ret_obj


class BasicTCPHandler(BaseRequestHandler):

    def handle(self):
        while True:
            recv_obj = self.recv_data()
            try:
                send_obj = self.server.handle_received_obj(recv_obj, self)  # The callback function
            except Exception as e:
                traceback.print_exc()
                send_obj = e
            if isinstance(recv_obj, int):
                if recv_obj == 0:
                    try:
                        self.request.shutdown(socket.SHUT_RDWR)
                    except OSError as e:
                        # OSError: [Errno 57] Socket is not connected is ok
                        if e.errno == 57:
                            pass
                        else:
                            raise
                    self.request.close()
                    print('Handler exit for client {}'.format(self.client_address))
                elif recv_obj == -1:
                    print('Server is going to shutdown')
                    self.server.call_shutdown = True
                break
            self.send_data(send_obj)
        # self.request, self.server,self.client_address

    def setup(self) -> None:
        if hasattr(self.server, "setup_handler"):
            self.server.setup_handler(self)

    def finish(self) -> None:
        if hasattr(self.server, "finish_handler"):
            self.server.finish_handler(self)

    def send_data(self, obj):
        with self.request.makefile('wb', buffering=256) as wfile:
            serialize(obj, wfile)
            wfile.flush()
        return self

    def recv_data(self):
        with self.request.makefile("rb", buffering=256) as rfile:
            return deserialize(rfile)


class BasicServer(TCPServer):
    def __init__(self, addr, port, handler_class=BasicTCPHandler):
        super(BasicServer, self).__init__((addr, port), handler_class)
        self.timeout = 3600
        self.call_shutdown = False

    @abstractmethod
    def handle_received_obj(self, recv_obj, handler):
        raise NotImplementedError(
            'Please Implement the method "handle_received_obj(recv_obj,handler)" in your server object')

    def start_server(self):
        self.server_thread = Thread(target=self.serve_forever)
        self.server_thread.start()
        print('Server Thread Started')
        while True:
            if self.call_shutdown:
                self.shutdown()
                self.server_thread.join()
                break
            time.sleep(0.5)


class BasicClient:  # Must follow a send-recv routine
    conn: socket.socket = None

    def __init__(self) -> None:
        atexit.register(self.__del__)

    def start_connection(self, server_addr, server_port):
        if self.conn is not None:
            raise RuntimeError('Client already connected')
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # wait for 10 sec
        self.conn.settimeout(10)
        self.conn.connect((server_addr, server_port))
        print(f'Client connected {self.conn.getpeername()}')
        return self

    def stop_connection(self, stop_server=False):
        try:
            if stop_server:
                self.send_data(-1)
            else:
                self.send_data(0)
            self.conn.shutdown(socket.SHUT_RDWR)
            self.conn.close()
        except OSError as e:
            # OSError: [Errno 57] Socket is not connected is ok
            if e.errno == 57:
                pass
            else:
                raise
        finally:
            self.conn = None
        return self

    def send_data(self, obj):
        with self.conn.makefile('wb', buffering=256) as wfile:
            serialize(obj, wfile)
            wfile.flush()
        return self

    def recv_data(self):
        with self.conn.makefile("rb", buffering=256) as rfile:
            return deserialize(rfile)

    def __del__(self):
        if self.conn is not None:
            try:
                self.conn.settimeout(0.01)
                data = self.conn.recv(1, socket.MSG_PEEK)
                # socket already close
                if len(data) == 0:
                    return
            except ConnectionResetError:
                # socket already close for other reason
                return
            except BlockingIOError:
                pass
            except socket.timeout:
                pass
            self.stop_connection()


def server_client_test():
    """
    Server Demo code
    """

    class ServerExample(BasicServer):  # Used by a server, your logic
        def __init__(self, addr, port):
            super(ServerExample, self).__init__(addr, port)
            self.result = 0
            self.cmd_cnt = 0
            pass

        def handle_received_obj(self, recv_obj, handle):
            print('Server received {} from {}'.format(recv_obj, handle.client_address))
            if isinstance(recv_obj, dict):
                for cmd, cmd_val in recv_obj.items():
                    if cmd == '+':
                        self.result += cmd_val
                    elif cmd == '-':
                        self.result -= cmd_val
                    self.cmd_cnt += 1
            elif isinstance(recv_obj, str):
                print('Server recv string {}'.format(recv_obj))
            return (self.result, self.cmd_cnt)

    server = ServerExample('localhost', 12345)
    print('Starting server')
    # Typically call server.start_server() for typical usage, and wait a client to shut it down
    # Here we use another thread for continuing the code execution
    start_server_thd = Thread(target=server.start_server)
    start_server_thd.start()

    """
    Client Demo code
    """

    client = BasicClient()
    client.start_connection('localhost', 12345)
    result = 0
    for i in range(20):
        cmd = np.random.choice(['+', '-'])
        val = np.random.randint(-5, 5)
        if cmd == '+':
            result += val
        else:
            result -= val
        data = {cmd: val}
        client.send_data(data)
        recv_obj = client.recv_data()
        recv_cmd_result, recv_cmd_cnt = recv_obj
        print('Client result={}, server result={}, cnt={}/{}'.format(result, recv_cmd_result, i + 1, recv_cmd_cnt))
    client.stop_connection()
    # Test2
    client.start_connection('localhost', 12345)
    client.stop_connection(stop_server=True)


if __name__ == '__main__':
    server_client_test()

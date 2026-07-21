import socket
import logging

# 配置基础日志，方便后续调试
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MrmUDP:
    """
    底层 UDP 通信类。
    完全替代原 C 项目中的 mrmIf.c (仅保留以太网功能)。
    """

    # 默认雷达 UDP 端口 (Time Domain MRM 雷达通常使用 21210)
    # 如果你的雷达配置了其他端口，可以在实例化时覆盖
    DEFAULT_PORT = 21210

    def __init__(self, device_ip: str, port: int = DEFAULT_PORT, timeout: float = 2.0):
        """
        初始化 UDP 连接 (替代 mrmIfInit)
        """
        self.ip = device_ip
        self.port = port
        self.timeout = timeout

        try:
            # 创建 UDP 套接字 (AF_INET = IPv4, SOCK_DGRAM = UDP)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB 缓冲区
            # 设置超时时间，防止程序在收不到雷达数据时死锁
            self.sock.settimeout(self.timeout)
            logging.info(f"UDP 初始化完成 -> 目标: {self.ip}:{self.port}, 超时: {self.timeout}s")
        except Exception as e:
            logging.error(f"UDP 套接字创建失败: {e}")
            raise

    def send_packet(self, data: bytes) -> int:
        """
        发送数据包 (替代 mrmIfSendPacket)

        :param data: 要发送的二进制字节流 (由 mrm_protocol.py 打包)
        :return: 实际发送的字节数
        """
        try:
            bytes_sent = self.sock.sendto(data, (self.ip, self.port))
            logging.debug(f"发送了 {bytes_sent} 字节到 {self.ip}:{self.port}")
            return bytes_sent
        except Exception as e:
            logging.error(f"发送数据失败: {e}")
            raise

    def get_packet(self, buffer_size: int = 4096) -> bytes:
        """
        接收数据包 (替代 mrmIfGetPacket)

        :param buffer_size: 接收缓冲区大小，雷达单包最大通常不超过 2048 字节
        :return: 接收到的二进制字节流，如果超时则返回 None
        """
        try:
            # 接收数据，忽略发送方的地址信息（因为我们只连一个雷达）
            data, _ = self.sock.recvfrom(buffer_size)
            logging.debug(f"接收到 {len(data)} 字节数据")
            return data
        except socket.timeout:
            # UDP 是不可靠传输，超时是正常现象，这里不抛出异常，而是返回 None 交给上层处理
            logging.warning("读取雷达数据超时")
            return None
        except Exception as e:
            logging.error(f"接收数据异常: {e}")
            raise

    def close(self):
        """
        关闭连接 (替代 mrmIfClose)
        """
        if self.sock:
            self.sock.close()
            logging.info("UDP 套接字已关闭")
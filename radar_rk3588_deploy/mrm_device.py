import time
import struct
import logging
from mrm_udp import MrmUDP
from mrm_protocol import (
    MrmPacketizer,
    MrmConfiguration,
    FullScanInfo,
    MRM_GET_CONFIG_CONFIRM,
    MRM_SET_CONFIG_CONFIRM,
    MRM_CONTROL_CONFIRM,
    MRM_FULL_SCAN_INFO,
    MRM_CONFIRM_SUCCESS
)


class MrmDevice:
    """
    雷达设备控制核心类。
    封装了读写配置、控制扫描和接收数据的 API，完全替代原 C 项目的 mrm.c。
    """

    def __init__(self, ip: str):
        # 初始化 UDP 底层通信
        self.udp = MrmUDP(ip)
        # 消息 ID 计数器，用于匹配 Request 和 Confirm
        self._msg_id_counter = 0

    def _get_next_msg_id(self) -> int:
        """生成自增的消息 ID (1~65535)"""
        self._msg_id_counter = (self._msg_id_counter + 1) % 65536
        if self._msg_id_counter == 0:
            self._msg_id_counter = 1
        return self._msg_id_counter

    def _wait_for_confirm(self, expected_type: int, expected_id: int, timeout: float = 2.0) -> bytes:
        """
        核心辅助方法：等待并匹配特定类型和 ID 的确认包
        """
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            data = self.udp.get_packet()
            if not data:
                continue  # 超时没收到，继续等

            # 只要是标准消息，前 4 个字节一定是 msgType(2) 和 msgId(2)
            if len(data) >= 4:
                msg_type, msg_id = struct.unpack(">HH", data[:4])

                # 精确匹配类型和 ID，防止收到历史遗留的脏数据
                if msg_type == expected_type and msg_id == expected_id:
                    return data
                elif msg_type == MRM_FULL_SCAN_INFO:
                    # 如果在等配置的时候收到了扫描数据，直接丢弃（或者你可以用队列缓存起来）
                    pass
                else:
                    logging.debug(f"收到未预期的包: Type=0x{msg_type:04X}, ID={msg_id}")

        raise TimeoutError(f"等待 Confirm 消息超时 (Expected Type: 0x{expected_type:04X})")

    def get_config(self) -> MrmConfiguration:
        msg_id = self._get_next_msg_id()
        req_pkt = MrmPacketizer.get_config(msg_id)

        logging.info("正在获取雷达配置...")
        self.udp.send_packet(req_pkt)

        resp_data = self._wait_for_confirm(MRM_GET_CONFIG_CONFIRM, msg_id)

        # GET_CONFIG_CONFIRM 没有 status 字段，config 直接从第 4 字节开始
        config = MrmConfiguration.unpack(resp_data[4:36])
        logging.info(f"成功获取雷达配置。NodeID={config.node_id}, "
                     f"Start={config.scan_start_ps}ps, End={config.scan_end_ps}ps")
        return config

    def set_config(self, config: MrmConfiguration):
        """
        设置雷达配置 (替代 mrmConfigSet)
        """
        msg_id = self._get_next_msg_id()
        req_pkt = MrmPacketizer.set_config(msg_id, config)

        logging.info("正在下发雷达配置...")
        self.udp.send_packet(req_pkt)

        resp_data = self._wait_for_confirm(MRM_SET_CONFIG_CONFIRM, msg_id)
        status = struct.unpack(">I", resp_data[4:8])[0]

        if status != MRM_CONFIRM_SUCCESS:
            raise RuntimeError(f"设置配置失败，雷达返回状态码: {status}")
        logging.info("成功下发雷达配置。")

    def start_scan(self, scan_count: int = 65535, interval_us: int = 125000):
        """
        启动扫描 (替代 mrmControl)
        :param scan_count: 扫描次数 (65535 通常表示无限扫描)
        :param interval_us: 扫描间隔 (微秒，125000us = 8Hz)
        """
        msg_id = self._get_next_msg_id()
        req_pkt = MrmPacketizer.control_scan(msg_id, scan_count, interval_us)

        logging.info(f"发送扫描控制指令: 次数={scan_count}, 间隔={interval_us}us")
        self.udp.send_packet(req_pkt)

        resp_data = self._wait_for_confirm(MRM_CONTROL_CONFIRM, msg_id)
        status = struct.unpack(">I", resp_data[4:8])[0]

        if status != MRM_CONFIRM_SUCCESS:
            raise RuntimeError(f"启动扫描失败，雷达返回状态码: {status}")
        logging.info("雷达已成功启动扫描。")

    def get_scan_info(self) -> FullScanInfo:
        """
        非阻塞地尝试获取一次雷达扫描数据 (替代 mrmInfoGet)
        如果没收到，返回 None。
        """
        data = self.udp.get_packet()
        if not data:
            return None

        if len(data) >= 4:
            msg_type = struct.unpack(">HH", data[:4])[0]
            if msg_type == MRM_FULL_SCAN_INFO:
                return FullScanInfo.from_bytes(data)

        return None

    def close(self):
        """关闭设备连接"""
        self.udp.close()
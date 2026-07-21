import struct
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# ==========================================
# 1. 消息类型常量 (Message Types)
# ==========================================
# 基础消息类型
RCRM_MSG_TYPE_REQUEST = 0xF000
RCRM_MSG_TYPE_CONFIRM = 0xF100
RCRM_MSG_TYPE_INFO = 0xF200

# 通用请求/确认
MRM_GET_STATUS_INFO_REQUEST = RCRM_MSG_TYPE_REQUEST + 1  # 0xF001
MRM_GET_STATUS_INFO_CONFIRM = RCRM_MSG_TYPE_CONFIRM + 1  # 0xF101
MRM_FULL_SCAN_INFO = RCRM_MSG_TYPE_INFO + 1  # 0xF201

# MRM 专用请求/确认
MRM_MSG_TYPE_REQ = 0x1000
MRM_MSG_TYPE_CNF = 0x1100

MRM_SET_CONFIG_REQUEST = MRM_MSG_TYPE_REQ + 1  # 0x1001
MRM_SET_CONFIG_CONFIRM = MRM_MSG_TYPE_CNF + 1  # 0x1101
MRM_GET_CONFIG_REQUEST = MRM_MSG_TYPE_REQ + 2  # 0x1002
MRM_GET_CONFIG_CONFIRM = MRM_MSG_TYPE_CNF + 2  # 0x1102
MRM_CONTROL_REQUEST = MRM_MSG_TYPE_REQ + 3  # 0x1003
MRM_CONTROL_CONFIRM = MRM_MSG_TYPE_CNF + 3  # 0x1103

# ==========================================
# 2. 其他常量
# ==========================================
MRM_MAX_SCAN_SAMPLES = 350
MRM_CONFIRM_SUCCESS = 0


# ==========================================
# 3. 数据结构定义
# ==========================================

@dataclass
class MrmConfiguration:
    """对应 C 结构体 mrmConfiguration (32 字节)"""
    node_id: int = 0
    scan_start_ps: int = 10000
    scan_end_ps: int = 39297
    scan_resolution_bins: int = 32
    base_integration_index: int = 12
    segment_num_samples: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    segment_int_mult: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    antenna_mode: int = 0  # MRM_ANTENNAMODE_TXA_RXA
    tx_gain: int = 63
    code_channel: int = 0
    persist_flag: int = 0

    # 格式串说明: >I (nodeId) ii (start/end) HH (res/baseInt) HHHH (segSamples) BBBB (segMult) BBBB (ant/gain/code/persist)
    STRUCT_FMT = ">IiiHHHHHHBBBBBBBB"

    def pack(self) -> bytes:
        return struct.pack(
            self.STRUCT_FMT,
            self.node_id, self.scan_start_ps, self.scan_end_ps,
            self.scan_resolution_bins, self.base_integration_index,
            *self.segment_num_samples, *self.segment_int_mult,
            self.antenna_mode, self.tx_gain, self.code_channel, self.persist_flag
        )

    @classmethod
    def unpack(cls, data: bytes):
        vals = struct.unpack(cls.STRUCT_FMT, data[:32])
        return cls(
            node_id=vals[0], scan_start_ps=vals[1], scan_end_ps=vals[2],
            scan_resolution_bins=vals[3], base_integration_index=vals[4],
            segment_num_samples=list(vals[5:9]),
            segment_int_mult=list(vals[9:13]),
            antenna_mode=vals[13], tx_gain=vals[14], code_channel=vals[15], persist_flag=vals[16]
        )


@dataclass
class FullScanInfo:
    """对应 C 结构体 mrmMsg_FullScanInfo (包头 52 字节 + 数据)"""
    msg_type: int
    msg_id: int
    source_id: int
    timestamp: int
    channel_rise_time: int
    v_peak_snr: int
    led_index: int
    lockspot_offset: int
    scan_start_ps: int
    scan_stop_ps: int
    scan_step_bins: int
    scan_filtering: int
    antenna_id: int
    operation_mode: int
    num_samples_in_msg: int
    num_samples_total: int
    message_index: int
    num_messages_total: int
    samples: np.ndarray  # 使用 numpy 存储 int32 采样数据

    # 包头格式 (不含最后的 scan 数组)
    HEADER_FMT = ">HHIIIIiiiiHBBBBHIHH"
    HEADER_SIZE = 52


    # mrm_protocol.py 中 FullScanInfo 的修正建议
    @classmethod
    def from_bytes(cls, data: bytes):
        if len(data) < cls.HEADER_SIZE:
            return None

        h = struct.unpack(cls.HEADER_FMT, data[:cls.HEADER_SIZE])

        # 修正后的索引映射 (根据 ">HHIIIIiiiiHBBBBHIHH" 格式串)
        # Index 14 是 B (operation_mode)
        # Index 15 是 H (num_samples_in_msg) <- 关键修正
        num_samples = h[15]

        # 提取采样数据
        samples_raw = np.frombuffer(data, dtype='>i4', offset=cls.HEADER_SIZE, count=num_samples)

        return cls(
            msg_type=h[0], msg_id=h[1], source_id=h[2], timestamp=h[3],
            channel_rise_time=h[4], v_peak_snr=h[5], led_index=h[6],
            lockspot_offset=h[7], scan_start_ps=h[8], scan_stop_ps=h[9],
            scan_step_bins=h[10], scan_filtering=h[11],
            # h[12] 通常是 reserved 字节，可以忽略
            antenna_id=h[13],
            operation_mode=h[14],
            num_samples_in_msg=num_samples,  # 使用修正后的 h[15]
            num_samples_total=h[16],  # 对应 I
            message_index=h[17],  # 对应 H
            num_messages_total=h[18],  # 对应 H
            samples=samples_raw
        )


# ==========================================
# 4. 消息打包工具类
# ==========================================
class MrmPacketizer:
    @staticmethod
    def request_header(msg_type: int, msg_id: int) -> bytes:
        """所有 Request 消息的通用 4 字节头部"""
        return struct.pack(">HH", msg_type, msg_id)

    @staticmethod
    def set_config(msg_id: int, config: MrmConfiguration) -> bytes:
        """构造设置配置请求包"""
        header = MrmPacketizer.request_header(MRM_SET_CONFIG_REQUEST, msg_id)
        return header + config.pack()

    @staticmethod
    def get_config(msg_id: int) -> bytes:
        """构造读取配置请求包"""
        return MrmPacketizer.request_header(MRM_GET_CONFIG_REQUEST, msg_id)

    @staticmethod
    def control_scan(msg_id: int, scan_count: int, interval_us: int) -> bytes:
        """构造扫描控制请求包"""
        header = MrmPacketizer.request_header(MRM_CONTROL_REQUEST, msg_id)
        # msrScanCount(H) + reserved(H) + msrIntervalTimeMicroseconds(I)
        body = struct.pack(">HHI", scan_count, 0, interval_us)
        return header + body
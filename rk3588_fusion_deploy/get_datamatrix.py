import time
import numpy as np
from mrm_device import MrmDevice

# 全局变量，用于优雅退出
running = True

def stream_datamatrix(ip, start_ps, stop_ps, interval_us, chunk_size,program_start_time=None):
    """
    [新增功能] 连续数据流生成器
    只连接一次雷达，启动无限扫描。每攒够 chunk_size 帧数据，就通过 yield 发送出去。
    """
    global running
    running = True

    device = None
    scan_matrix = []

    try:
        device = MrmDevice(ip)
        print(f"\n[*] 正在建立连续数据流连接: {ip}...")

        config = device.get_config()
        config.scan_start_ps = start_ps
        config.scan_end_ps = stop_ps
        device.set_config(config)

        # 关键修改：scan_count=65535 在 P440 协议中通常代表无限连续扫描
        device.start_scan(scan_count=65535, interval_us=interval_us)
        print(f"[*] 连续扫描已启动，数据洪流开启... (按 Ctrl+C 停止)")
        scan_start_time = time.time()
        if program_start_time is not None:
            print(f"[计时] 程序启动 → 雷达开始扫描: {scan_start_time - program_start_time:.3f}s")

        scan_counter = 0
        timeout_count = 0
        pending_scan = {}
        first_yield_time = None

        while running:
            scan_info = device.get_scan_info()
            if scan_info:
                timeout_count = 0
                mid = scan_info.msg_id

                if mid not in pending_scan:
                    pending_scan[mid] = [None] * scan_info.num_messages_total

                if scan_info.message_index < len(pending_scan[mid]):
                    pending_scan[mid][scan_info.message_index] = scan_info.samples

                if all(part is not None for part in pending_scan[mid]):
                    full_samples = np.concatenate(pending_scan.pop(mid))
                    scan_matrix.append(full_samples)
                    scan_counter += 1

                    # ================= [核心逻辑：滑动窗口] =================
                    # 当攒够 chunk_size 帧(约15秒)，yield 给模型推理
                    if len(scan_matrix) >= chunk_size:
                        chunk_ready_time = time.time()
                        if first_yield_time is None:
                            print(f"\n[计时] 雷达开始扫描 → 首次收集够 {chunk_size} 帧: {chunk_ready_time -scan_start_time:.3f}s")
                        else:
                            print(f"\n[计时] 上次收集够 → 本次收集够: {chunk_ready_time - first_yield_time:.3f}s")
                        first_yield_time = chunk_ready_time
                        # --- 计时结束 ---
                        yield np.vstack(scan_matrix[:chunk_size])


                        slide_step = chunk_size // 4
                        #slide_step = chunk_size
                        scan_matrix = scan_matrix[slide_step:]
                    # ===============================================

                # 内存保护：清理丢包的旧数据
                if len(pending_scan) > 3:
                    oldest_mid = min(pending_scan.keys())
                    pending_scan.pop(oldest_mid)


            else:
                timeout_count += 1
                # 连续流模式下，允许偶尔丢包超时，只有长时间(约2秒)收不到才报警
                if timeout_count >= 2000:
                    print(f"\r[-] 警告: 较长时间未收到雷达数据...", end="")
                time.sleep(0.001)



    except Exception as e:
        print(f"\n[!] 数据流运行出错: {e}")
    finally:
        if device is not None:
            device.close()
        print("\n[*] 雷达连接已安全断开。")



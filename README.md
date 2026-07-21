# 智能跌倒检测系统开源说明

本文件夹是从项目中整理出的开源副本，用于竞赛提交和代码公开。

## 项目简介

本项目面向 RK3588 Linux 开发板，完成视觉与雷达融合的跌倒检测，并配套微信小程序、Android App 和屏幕端交互界面。系统包含姿态识别、雷达识别、融合判定、电机跟踪控制、语音对话、联网搜索、定时提醒和双向信息交流等功能。

AI 对话模块以 DeepSeek 作为主要语言模型，用于理解语音指令、判断控制意图和生成回复；联网增强部分通过智谱 GLM 搜索接口获取时效性信息，再交给 DeepSeek 组织回答。发布副本中不包含任何可用密钥，运行前需要自行配置环境变量。

## 目录结构

- `rk3588_fusion_deploy/`：RK3588 上运行的核心融合检测程序，包含视觉模型、雷达模型、AI 对话、联网搜索、提醒、双向消息、语音服务和屏幕界面。
- `android_apk_webview/`：Android APK 工程，封装手机端 WebView 界面，并增加实时语音、提醒、双向交流等 App 功能。
- `radar/`：雷达模型训练与测试相关代码。
- `radar_rk3588_deploy/`：雷达模型在 RK3588 上部署测试的相关代码。
- `rk3588_posture_deploy/`：视觉姿态模型在 RK3588 上部署测试的相关代码。


## API 密钥配置

发布副本中的密钥默认值已经清空。运行 AI 对话、联网搜索和腾讯云语音能力前，需要在开发板环境中设置：

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export ZHIPU_API_KEY="your_zhipu_api_key"
export TENCENTCLOUD_SECRET_ID="your_tencentcloud_secret_id"
export TENCENTCLOUD_SECRET_KEY="your_tencentcloud_secret_key"
```

微信小程序的 `app/project.config.json` 中 AppID 已改为 `touristappid`。如需真机调试或发布，请替换为自己的微信小程序 AppID。

## RK3588 端运行

进入核心部署目录：

```bash
cd rk3588_fusion_deploy
```

典型启动命令：

```bash
python3 fusion_rk3588.py \
  --camera 25 \
  --radar-ip 192.168.1.100 \
  --app-host 127.0.0.1 \
  --vision-model models/posture_classifier_5class.rknn \
  --vision-lying-stable-frames 5 \
  --vision-lying-fall-hold-sec 2 \
  --dynamic-radar-weight 0.4 \
  --static-radar-weight 0.3 \
  --status-image-sec 1
```

语音监听服务可单独启动：

```bash
python3 voice_intercom_server.py \
  --host 0.0.0.0 \
  --port 8890 \
  --capture-device hw:rockchipnau8822,0 \
  --play-device plughw:1,0
```

实际设备上的摄像头编号、雷达 IP、声卡设备名和模型路径需要按硬件环境调整。


# 智能跌倒检测系统开源说明

本文件夹是从项目中整理出的开源副本，用于竞赛提交和代码公开。副本只保留运行、训练、端侧展示和 AI Agent 功能相关的必要文件，已经移除个人 API 密钥、私有 IDE 配置、运行日志、构建缓存、数据集和训练中间文件。

## 项目简介

本项目面向 RK3588 Linux 开发板，完成视觉与雷达融合的跌倒检测，并配套微信小程序、Android App 和屏幕端交互界面。系统包含姿态识别、雷达识别、融合判定、电机跟踪控制、语音对话、联网搜索、定时提醒和双向信息交流等功能。

AI 对话模块以 DeepSeek 作为主要语言模型，用于理解语音指令、判断控制意图和生成回复；联网增强部分通过智谱 GLM 搜索接口获取时效性信息，再交给 DeepSeek 组织回答。发布副本中不包含任何可用密钥，运行前需要自行配置环境变量。

## 目录结构

- `rk3588_fusion_deploy/`：RK3588 上运行的核心融合检测程序，包含视觉模型、雷达模型、AI 对话、联网搜索、提醒、双向消息、语音服务和屏幕界面。
- `app/`：微信小程序和 Node.js 状态中转服务，用于手机端查看状态、控制设备、接收报警和预览图像。
- `android_apk_webview/`：Android APK 工程，封装手机端 WebView 界面，并增加实时语音、提醒、双向交流等 App 功能。
- `vision_training/`：视觉分类模型训练、数据划分、裁剪清洗和 ONNX 导出相关脚本。
- `radar/`：雷达模型训练与测试相关代码。
- `radar_rk3588_deploy/`：雷达模型在 RK3588 上部署测试的相关代码。
- `rk3588_posture_deploy/`：视觉姿态模型在 RK3588 上部署测试的相关代码。
- `web_agent_research/`：联网搜索 Agent 的本地测试程序。
- `report_ui_mockup/`：报告截图用的屏幕界面模拟程序。
- `docs/`：项目改动记录、AI Agent 架构图和相关说明素材。

## 未包含内容

- 原始训练数据集、清洗后的图片数据和临时划分目录。
- 训练得到的 `.pth`、`.pt` 等中间权重文件。
- Android/Node/Gradle 构建缓存，例如 `node_modules`、`build`、`.gradle`。
- 小程序私有配置、运行日志、语音临时文件、提醒运行态文件。
- 个人 API 密钥和账号标识。

部署目录中的 `.rknn` 模型属于运行所需文件，已随开源副本保留。如果需要重新训练模型，可使用 `vision_training/` 中的脚本重新导出 ONNX/RKNN 后替换部署目录中的模型。

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

## 手机端与服务端

微信小程序状态中转服务：

```bash
cd app
node server.js
```

微信小程序可用微信开发者工具打开 `app/` 目录。Android App 可用 Android Studio 打开 `android_apk_webview/`，或在安装好 Android SDK 后执行：

```bash
cd android_apk_webview
./gradlew :app:assembleDebug
```

Windows 下可使用：

```bat
gradlew.bat :app:assembleDebug
```

## 训练流程

视觉模型训练相关脚本位于 `vision_training/`：

- `crop_actions_for_cleaning.py`：检测人体后裁剪图片，便于人工清洗。
- `build_dataset_with_cleaned_class.py`：将清洗后的类别图片合并并划分训练集。
- `train.py`：训练视觉分类模型。
- `export_onnx.py`：导出 ONNX 模型，后续可转换为 RKNN。
- `inference.py`：本地推理测试。

训练数据和中间权重没有放入本开源副本，需要按自己的数据路径重新准备。

## 开源前检查

本副本已经完成以下处理：

- 只在 `开源文件夹` 内复制和修改文件，未修改原项目目录。
- 删除硬编码的 DeepSeek、智谱和腾讯云密钥。
- 将微信小程序 AppID 替换为 `touristappid`。
- 排除数据集、缓存、构建产物、日志和运行态文件。

正式发布前，建议根据主办方要求补充 `LICENSE`、团队信息和模型/数据来源说明。

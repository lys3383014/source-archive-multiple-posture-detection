# Android APK Realtime Voice

This APK now prefers a native low-latency voice stream when it runs inside Android.
The old HTTP chunk voice service is still kept as a fallback.

## Ports

- `8890`: old HTTP chunk service, served by `voice_intercom_server.py`.
- `8891`: native realtime TCP stream service, served by `voice_stream_server.py`.

## Board Command

Run this on the RK3588 board:

```bash
cd /root/Pose_Detection_20260518_1405/rk3588_fusion_deploy
conda activate rknn_lite_env_1

python3 voice_stream_server.py \
  --host 0.0.0.0 \
  --port 8891 \
  --capture-device plughw:rockchipnau8822,0 \
  --play-device plughw:1,0 \
  --rate 16000 \
  --channels 1 \
  --frame-ms 40 \
  --input-gain 8 \
  --output-gain 8
```

## APK Behavior

- In the Android APK, the voice buttons call the native `AndroidVoice` bridge first.
- If the realtime stream cannot start, the page falls back to the old `8890` chunk service.
- In a normal browser or WeChat-like WebView without `AndroidVoice`, only the old HTTP chunk mode is used.

# RK3588 Vision + Radar Fusion

This folder is separate from the existing vision and radar deployment folders.
Those working programs are not modified.

## Models

- `models/yolo11.rknn`: vision person detector, default NPU core 0.
- `models/posture_classifier_large.rknn`: vision posture classifier, default NPU core 1.
- `models/radar_nn1_4class_v2.rknn`: radar classifier, default NPU core 2.

## Fusion Rule

Classes are unified as:

- vision `stand`, radar `stand` -> `stand`
- vision `walk`, radar `walk` -> `walk`
- vision `bendover`, radar `bend` -> `bend`
- vision `lying`, radar `fall` -> `fall`

Radar weight starts at `--radar-weight-start` after each radar update and
linearly decays to `--radar-weight-end` within one second. The defaults are
`0.4 -> 0.3`. Vision weight is `1 - radar_weight`. If radar is stale for more
than `--radar-stale-sec`, its weight becomes `0`.

Vision uses a 5-result mode filter by default. Ties are resolved by the most
recent result. Vision probabilities are softened with `--vision-temperature`
before fusion; the default is `1.6`, and `1.0` keeps the original confidence.
Radar probabilities can also be softened with `--radar-temperature`; the default
is `1.0`.

Camera tracking is off by default. When enabled from the app/web GUI, the vision
thread selects the detected person whose box center is closest to the camera
center, then drives the horizontal motor with PID so both x coordinates align.
If no person is detected, detection is disabled, or privacy mode is active, the
motor command is `0`.

At startup the fusion program sends `01 93 88 01 6B` to set the current motor
position as zero, then waits 20 ms. The app/web GUI can also send a target motor
angle. Each target angle command first sends the home frame from
`serial_location.py`, waits `--motor-location-settle-sec` seconds, then sends
the target angle frame in a background thread so recognition keeps running.
The app/web GUI also supports setting the current position as zero and relative
angle moves. Manual angle controls are ignored while tracking is enabled.

The on-screen display includes an AI voice assistant button. Press `AI REC` to
start recording from the board microphone, press `AI STOP` to end recording, and
the board will send the recording to Tencent Cloud ASR, ask DeepSeek with the
recognized text, synthesize the reply with edge-tts, and play the reply through
the board speaker. While recognizing, waiting for DeepSeek, synthesizing, or
playing the reply, the button is disabled. API keys are read from environment
variables by default. Conversation history is in memory only by default, so
restarting the fusion program clears AI context; use `--ai-persistent-history`
only if you want to save and reload `history.json`.

```bash
export DEEPSEEK_API_KEY="..."
export TENCENTCLOUD_SECRET_ID="..."
export TENCENTCLOUD_SECRET_KEY="..."
```

Useful AI voice options:

```bash
--ai-assistant
--ai-capture-device plughw:rockchipnau8822,0
--ai-play-device plughw:1,0
--ai-tts-voice zh-CN-XiaoxiaoNeural
--ai-persistent-history
--deepseek-api-key ...
--tencent-secret-id ... --tencent-secret-key ...
```

The phone voice intercom runs as a separate lightweight HTTP service. The app
can pull short board-mic WAV chunks from RK3588 and upload short phone-mic audio
chunks back to RK3588 for playback. It is switch based, not push-to-talk. The
default devices are `hw:rockchipnau8822,0` for capture and `plughw:1,0` for
playback, but both are command-line parameters.

Board microphone capture is continuous. The server keeps one `arecord` process
running and serves cursor-based WAV chunks from an in-memory buffer, so the app
does not lose audio between repeated short recordings.

## Run

```bash
cd rk3588_fusion_deploy
pip install -r requirements_rk3588.txt
python3 fusion_rk3588.py --camera 21 --radar-ip 192.168.1.100
```

To connect RK3588 to the app/control server, pass the app host explicitly. If
the app server runs on your PC, use your PC LAN/hotspot IP, not `127.0.0.1` on
RK3588:

```bash
python3 fusion_rk3588.py --camera 21 --radar-ip 192.168.1.100 --app-host 192.168.43.182
```

Start the voice service on RK3588 in another terminal:

```bash
python3 voice_intercom_server.py --port 8890 \
  --capture-device hw:rockchipnau8822,0 \
  --play-device plughw:1,0
```

In the mini program settings page, keep the alarm/control server as before, and
set the voice service IP to the RK3588 board IP with port `8890`.

Voice input and output are amplified on RK3588 by default. The default gain is
`8x` for both directions. Use `--audio-gain` to set both at once, or tune the two
directions separately:

```bash
python3 voice_intercom_server.py --audio-gain 8
python3 voice_intercom_server.py --input-gain 6 --output-gain 10
```

If audio still breaks up on a busy network, try larger chunks:

```bash
python3 voice_intercom_server.py --chunk-seconds 2 --capture-buffer-sec 20
```

By default, alarm/control networking is disabled until `--app-host`,
`--alarm-url`, or `--control-url` is provided. This avoids sending to a stale
hard-coded IP.

When the fused result changes from non-fall to `fall`, the program sends one
`alarm=true` POST. Continuous fall frames do not send repeated alarms. When the
result returns to non-fall, it sends `alarm=false` once so the phone/app can
accept a later alarm again. A new `alarm=true` is also suppressed if it occurs
within `--alarm-cooldown-sec` seconds of the previous fall alarm; the default is
`40.0`. If the phone does not press the alarm acknowledge button within
`--fall-ack-timeout-sec` seconds, the on-board AI assistant asks a short
follow-up question, records the answer for `--fall-followup-record-sec` seconds,
then speaks its advice.

Fall alarms include the current vision frame as compressed JPEG base64 by
default. Use `--no-alarm-image` to disable it.

Set the receiver address to your phone/app server IP. Prefer `--app-host` when
alarm and control use the same server:

```bash
python3 fusion_rk3588.py --camera 21 --radar-ip 192.168.1.100 --app-host 192.168.43.182
```

The same `/status` URL is also polled for app control by default. When the app
switches to privacy mode (`mode=radar`), the vision thread releases the camera,
stops YOLO/classifier inference, and shows a black camera area. Daily mode
(`mode=vision`) reopens the camera and resumes fusion.

If the radar RKNN runtime reports input layout warnings:

```bash
python3 fusion_rk3588.py --camera 21 --radar-ip 192.168.1.100 --radar-input-format nchw
```

Useful knobs:

```bash
--speed-threshold 0.008
--vision-temperature 1.6
--radar-temperature 1.0
--radar-weight-start 0.4
--radar-weight-end 0.3
--vision-filter-window 5
--yolo-conf 0.65
--app-host 192.168.43.182
--alarm-url http://192.168.43.182:8889/status
--alarm-conf 0.0
--alarm-cooldown-sec 40
--fall-ack-timeout-sec 15
--fall-followup-record-sec 15
--audio-control-file /tmp/rk3588_audio_control.json
--alarm-image-width 640
--alarm-jpeg-quality 75
--control-url http://192.168.43.182:8889/status
--control-poll-sec 0.5
--motor-port /dev/ttyS9
--motor-init-zero --motor-init-zero-delay 0.02
--motor-location-settle-sec 2.0
--motor-angle-min -90 --motor-angle-max 90
--track-kp 200.0 --track-ki 20.0 --track-kd 0.8
--track-max-speed 60
--track-deadband-px 10
--track-send-interval 0.08
--radar-stale-sec 2.5
--yolo-npu-core 0 --vision-npu-core 1 --radar-npu-core 2
```

Voice priority is coordinated through `--audio-control-file`: phone-to-board
speaker audio preempts AI playback/recording; fall follow-up preempts manual AI;
AI recording temporarily returns silence for board-microphone listening.

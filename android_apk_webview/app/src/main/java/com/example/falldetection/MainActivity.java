package com.example.falldetection;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends Activity {
    private static final int AUDIO_PERMISSION_REQUEST = 1001;
    private static final String LOCAL_HOST = "appassets.androidplatform.net";
    private static final String LOCAL_PREFIX = "/assets/";
    private WebView webView;
    private NativeVoiceBridge nativeVoiceBridge;
    private final Object fallAlarmLock = new Object();
    private MediaPlayer fallAlarmPlayer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            getWindow().setDecorFitsSystemWindows(true);
        }
        requestAudioPermissionIfNeeded();
        setupWebView();
        webView.loadUrl("https://" + LOCAL_HOST + LOCAL_PREFIX + "www/index.html?v=4");
    }

    @SuppressLint({"SetJavaScriptEnabled", "JavascriptInterface"})
    private void setupWebView() {
        webView = new WebView(this);
        setContentView(webView);

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setCacheMode(WebSettings.LOAD_NO_CACHE);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN) {
            settings.setAllowFileAccessFromFileURLs(true);
            settings.setAllowUniversalAccessFromFileURLs(true);
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        }
        webView.clearCache(true);
        webView.clearHistory();

        nativeVoiceBridge = new NativeVoiceBridge();
        webView.addJavascriptInterface(nativeVoiceBridge, "AndroidVoice");

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                    WebResourceResponse response = interceptLocalAsset(request.getUrl());
                    if (response != null) {
                        return response;
                    }
                }
                return super.shouldInterceptRequest(view, request);
            }

            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, String url) {
                WebResourceResponse response = interceptLocalAsset(Uri.parse(url));
                if (response != null) {
                    return response;
                }
                return super.shouldInterceptRequest(view, url);
            }
        });
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(final PermissionRequest request) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                    runOnUiThread(() -> request.grant(request.getResources()));
                }
            }
        });
    }

    private WebResourceResponse interceptLocalAsset(Uri uri) {
        if (uri == null
                || !"https".equalsIgnoreCase(uri.getScheme())
                || !LOCAL_HOST.equalsIgnoreCase(uri.getHost())) {
            return null;
        }

        String path = uri.getPath();
        if (path == null || !path.startsWith(LOCAL_PREFIX)) {
            return null;
        }

        String assetPath = path.substring(LOCAL_PREFIX.length());
        if (assetPath.length() == 0 || assetPath.contains("..")) {
            return null;
        }

        try {
            InputStream stream = getAssets().open(assetPath);
            return new WebResourceResponse(getMimeType(assetPath), getEncoding(assetPath), stream);
        } catch (IOException ignored) {
            return null;
        }
    }

    private String getMimeType(String assetPath) {
        String lower = assetPath.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".html") || lower.endsWith(".htm")) {
            return "text/html";
        }
        if (lower.endsWith(".js")) {
            return "application/javascript";
        }
        if (lower.endsWith(".css")) {
            return "text/css";
        }
        if (lower.endsWith(".json")) {
            return "application/json";
        }
        if (lower.endsWith(".png")) {
            return "image/png";
        }
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
            return "image/jpeg";
        }
        if (lower.endsWith(".svg")) {
            return "image/svg+xml";
        }
        if (lower.endsWith(".mp3")) {
            return "audio/mpeg";
        }
        if (lower.endsWith(".wav")) {
            return "audio/wav";
        }
        return "application/octet-stream";
    }

    private String getEncoding(String assetPath) {
        String lower = assetPath.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".html")
                || lower.endsWith(".htm")
                || lower.endsWith(".js")
                || lower.endsWith(".css")
                || lower.endsWith(".json")
                || lower.endsWith(".svg")) {
            return "UTF-8";
        }
        return null;
    }

    private boolean hasRecordAudioPermission() {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.M
                || checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestAudioPermissionIfNeeded() {
        if (!hasRecordAudioPermission()) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, AUDIO_PERMISSION_REQUEST);
        }
    }

    @Override
    public void onBackPressed() {
        if (webView != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            webView.evaluateJavascript(
                    "(function(){return !!(window.handleAndroidBack && window.handleAndroidBack());})()",
                    handled -> {
                        if (!"true".equals(handled)) {
                            if (webView.canGoBack()) {
                                webView.goBack();
                            } else {
                                finish();
                            }
                        }
                    });
            return;
        }
        super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        stopFallAlarmSound();
        if (nativeVoiceBridge != null) {
            nativeVoiceBridge.stopAll();
        }
        if (webView != null) {
            webView.destroy();
            webView = null;
        }
        super.onDestroy();
    }

    private boolean playFallAlarmSound() {
        new Thread(this::runFallAlarmSound, "fall-alarm-sound").start();
        return true;
    }

    private void runFallAlarmSound() {
        MediaPlayer previous;
        synchronized (fallAlarmLock) {
            previous = fallAlarmPlayer;
            fallAlarmPlayer = null;
        }
        releaseMediaPlayer(previous);

        MediaPlayer player = new MediaPlayer();
        boolean started = false;
        try (AssetFileDescriptor afd = getAssets().openFd("www/fall_alarm.mp3")) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                player.setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ALARM)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build());
            } else {
                player.setAudioStreamType(AudioManager.STREAM_ALARM);
            }
            player.setDataSource(afd.getFileDescriptor(), afd.getStartOffset(), afd.getLength());
            player.setLooping(false);
            player.setOnCompletionListener(completed -> {
                synchronized (fallAlarmLock) {
                    if (fallAlarmPlayer == completed) {
                        fallAlarmPlayer = null;
                    }
                }
                releaseMediaPlayer(completed);
            });
            player.setOnErrorListener((failed, what, extra) -> {
                synchronized (fallAlarmLock) {
                    if (fallAlarmPlayer == failed) {
                        fallAlarmPlayer = null;
                    }
                }
                releaseMediaPlayer(failed);
                return true;
            });
            player.prepare();
            synchronized (fallAlarmLock) {
                fallAlarmPlayer = player;
            }
            player.start();
            started = true;
        } catch (Exception ignored) {
        } finally {
            if (!started) {
                releaseMediaPlayer(player);
            }
        }
    }

    private void stopFallAlarmSound() {
        MediaPlayer player;
        synchronized (fallAlarmLock) {
            player = fallAlarmPlayer;
            fallAlarmPlayer = null;
        }
        releaseMediaPlayer(player);
    }

    private void releaseMediaPlayer(MediaPlayer player) {
        if (player == null) {
            return;
        }
        try {
            player.stop();
        } catch (Exception ignored) {
        }
        try {
            player.release();
        } catch (Exception ignored) {
        }
    }

    private final class NativeVoiceBridge {
        private static final int SAMPLE_RATE = 16000;
        private static final int FRAME_MS = 40;
        private static final int SAMPLE_WIDTH_BYTES = 2;
        private static final int FRAME_BYTES = SAMPLE_RATE * SAMPLE_WIDTH_BYTES * FRAME_MS / 1000;
        private static final int CONNECT_TIMEOUT_MS = 3000;

        private final AtomicBoolean listening = new AtomicBoolean(false);
        private final AtomicBoolean talking = new AtomicBoolean(false);
        private final Object listenLock = new Object();
        private final Object talkLock = new Object();
        private Socket listenSocket;
        private Socket talkSocket;
        private Thread listenThread;
        private Thread talkThread;

        @JavascriptInterface
        public boolean isAvailable() {
            return true;
        }

        @JavascriptInterface
        public boolean startListen(String hostText, String portText) {
            String host = normalizeHost(hostText);
            int port = parsePort(portText, 8891);
            if (host.length() == 0) {
                notifyVoiceState("listen", false, "语音服务IP为空");
                return false;
            }

            stopListen();
            listening.set(true);
            updateCommunicationAudioMode();
            listenThread = new Thread(() -> runListen(host, port), "rk-voice-downlink");
            listenThread.start();
            notifyVoiceState("listen", true, null);
            return true;
        }

        @JavascriptInterface
        public void stopListen() {
            boolean wasActive = listening.getAndSet(false);
            closeListenSocket();
            if (listenThread != null) {
                listenThread.interrupt();
                listenThread = null;
            }
            updateCommunicationAudioMode();
            if (wasActive) {
                notifyVoiceState("listen", false, null);
            }
        }

        @JavascriptInterface
        public boolean startTalk(String hostText, String portText) {
            if (!hasRecordAudioPermission()) {
                runOnUiThread(MainActivity.this::requestAudioPermissionIfNeeded);
                notifyVoiceState("talk", false, "手机麦克风权限未开启");
                return false;
            }

            String host = normalizeHost(hostText);
            int port = parsePort(portText, 8891);
            if (host.length() == 0) {
                notifyVoiceState("talk", false, "语音服务IP为空");
                return false;
            }

            stopTalk();
            talking.set(true);
            updateCommunicationAudioMode();
            talkThread = new Thread(() -> runTalk(host, port), "rk-voice-uplink");
            talkThread.start();
            notifyVoiceState("talk", true, null);
            return true;
        }

        @JavascriptInterface
        public void stopTalk() {
            boolean wasActive = talking.getAndSet(false);
            closeTalkSocket();
            if (talkThread != null) {
                talkThread.interrupt();
                talkThread = null;
            }
            updateCommunicationAudioMode();
            if (wasActive) {
                notifyVoiceState("talk", false, null);
            }
        }

        @JavascriptInterface
        public void stopAll() {
            stopTalk();
            stopListen();
        }

        @JavascriptInterface
        public boolean playFallAlarm() {
            return MainActivity.this.playFallAlarmSound();
        }

        private void runListen(String host, int port) {
            AudioTrack audioTrack = null;
            String error = null;
            try {
                int minBuffer = AudioTrack.getMinBufferSize(
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT);
                int bufferSize = Math.max(minBuffer, FRAME_BYTES * 6);
                audioTrack = createAudioTrack(bufferSize);
                audioTrack.play();

                Socket socket = openSocket(host, port);
                synchronized (listenLock) {
                    listenSocket = socket;
                }
                writeHandshake(socket, "downlink");

                InputStream input = socket.getInputStream();
                byte[] buffer = new byte[FRAME_BYTES * 2];
                while (listening.get()) {
                    int read = input.read(buffer);
                    if (read < 0) {
                        break;
                    }
                    if (read > 0) {
                        audioTrack.write(buffer, 0, read);
                    }
                }
            } catch (Exception exc) {
                if (listening.get()) {
                    error = "实时监听连接失败";
                }
            } finally {
                listening.set(false);
                closeListenSocket();
                if (audioTrack != null) {
                    try {
                        audioTrack.stop();
                    } catch (Exception ignored) {
                    }
                    audioTrack.release();
                }
                updateCommunicationAudioMode();
                notifyVoiceState("listen", false, error);
            }
        }

        private void runTalk(String host, int port) {
            AudioRecord audioRecord = null;
            String error = null;
            try {
                int minBuffer = AudioRecord.getMinBufferSize(
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT);
                int bufferSize = Math.max(minBuffer, FRAME_BYTES * 6);
                audioRecord = new AudioRecord(
                        MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufferSize);
                if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                    throw new IllegalStateException("AudioRecord init failed");
                }

                Socket socket = openSocket(host, port);
                synchronized (talkLock) {
                    talkSocket = socket;
                }
                writeHandshake(socket, "uplink");

                OutputStream output = socket.getOutputStream();
                byte[] buffer = new byte[FRAME_BYTES];
                audioRecord.startRecording();
                while (talking.get()) {
                    int read = audioRecord.read(buffer, 0, buffer.length);
                    if (read > 0) {
                        output.write(buffer, 0, read);
                    }
                }
                output.flush();
            } catch (Exception exc) {
                if (talking.get()) {
                    error = "实时发送连接失败";
                }
            } finally {
                talking.set(false);
                closeTalkSocket();
                if (audioRecord != null) {
                    try {
                        audioRecord.stop();
                    } catch (Exception ignored) {
                    }
                    audioRecord.release();
                }
                updateCommunicationAudioMode();
                notifyVoiceState("talk", false, error);
            }
        }

        private AudioTrack createAudioTrack(int bufferSize) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                return new AudioTrack.Builder()
                        .setAudioAttributes(new AudioAttributes.Builder()
                                .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                                .build())
                        .setAudioFormat(new AudioFormat.Builder()
                                .setSampleRate(SAMPLE_RATE)
                                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                                .build())
                        .setBufferSizeInBytes(bufferSize)
                        .setTransferMode(AudioTrack.MODE_STREAM)
                        .build();
            }
            return new AudioTrack(
                    AudioManager.STREAM_VOICE_CALL,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSize,
                    AudioTrack.MODE_STREAM);
        }

        private Socket openSocket(String host, int port) throws IOException {
            Socket socket = new Socket();
            socket.setTcpNoDelay(true);
            socket.connect(new InetSocketAddress(host, port), CONNECT_TIMEOUT_MS);
            return socket;
        }

        private void writeHandshake(Socket socket, String role) throws IOException {
            String header = "role=" + role
                    + "\nrate=" + SAMPLE_RATE
                    + "\nchannels=1"
                    + "\nformat=s16le\n\n";
            socket.getOutputStream().write(header.getBytes(StandardCharsets.US_ASCII));
            socket.getOutputStream().flush();
        }

        private void closeListenSocket() {
            synchronized (listenLock) {
                if (listenSocket != null) {
                    try {
                        listenSocket.close();
                    } catch (IOException ignored) {
                    }
                    listenSocket = null;
                }
            }
        }

        private void closeTalkSocket() {
            synchronized (talkLock) {
                if (talkSocket != null) {
                    try {
                        talkSocket.close();
                    } catch (IOException ignored) {
                    }
                    talkSocket = null;
                }
            }
        }

        private void updateCommunicationAudioMode() {
            AudioManager manager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
            if (manager == null) {
                return;
            }
            boolean active = listening.get() || talking.get();
            if (active) {
                manager.setMode(AudioManager.MODE_IN_COMMUNICATION);
                manager.setSpeakerphoneOn(true);
            } else {
                manager.setSpeakerphoneOn(false);
                manager.setMode(AudioManager.MODE_NORMAL);
            }
        }

        private String normalizeHost(String hostText) {
            String host = hostText == null ? "" : hostText.trim();
            if (host.contains("://")) {
                Uri uri = Uri.parse(host);
                host = uri.getHost() == null ? "" : uri.getHost();
            }
            int slash = host.indexOf('/');
            if (slash >= 0) {
                host = host.substring(0, slash);
            }
            int colon = host.indexOf(':');
            if (colon > 0 && host.indexOf(':', colon + 1) < 0) {
                host = host.substring(0, colon);
            }
            return host;
        }

        private int parsePort(String portText, int fallback) {
            try {
                int port = Integer.parseInt(String.valueOf(portText).trim());
                if (port > 0 && port <= 65535) {
                    return port;
                }
            } catch (Exception ignored) {
            }
            return fallback;
        }

        private void notifyVoiceState(String kind, boolean active, String error) {
            if (webView == null || Build.VERSION.SDK_INT < Build.VERSION_CODES.KITKAT) {
                return;
            }
            String payload = "{\"kind\":\"" + escapeJson(kind)
                    + "\",\"active\":" + (active ? "true" : "false")
                    + ",\"error\":" + (error == null ? "null" : "\"" + escapeJson(error) + "\"")
                    + "}";
            String script = "window.onAndroidVoiceState && window.onAndroidVoiceState(" + payload + ");";
            runOnUiThread(() -> {
                if (webView != null) {
                    webView.evaluateJavascript(script, null);
                }
            });
        }

        private String escapeJson(String value) {
            return value.replace("\\", "\\\\")
                    .replace("\"", "\\\"")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r");
        }
    }
}

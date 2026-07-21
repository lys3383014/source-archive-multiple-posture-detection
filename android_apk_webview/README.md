# FallDetectionWebView APK 工程

这个目录是独立新建的 Android WebView 包装工程，没有修改旧的小程序或板端程序。

## 文件说明

- `app/src/main/assets/www/`：从当前微信小程序 `app/pages` 迁移出的 H5 单页界面资源，包含首页、检测模式、系统控制、语音对话、报警历史、报警详情和设置。
- `app/src/main/java/com/example/falldetection/MainActivity.java`：Android 壳程序，只负责打开 WebView 并通过 `https://appassets.androidplatform.net/assets/www/index.html` 加载本地界面资源。
- `app/src/main/AndroidManifest.xml`：声明联网、麦克风权限，并允许访问局域网 HTTP 服务。
- `app/src/main/res/xml/network_security_config.xml`：允许明文 HTTP，用于连接开发板 `8889` 和 `8890` 服务。
- `app/build.gradle`：Android app 模块配置。
- `build.gradle`、`settings.gradle`、`gradle.properties`：Gradle 工程配置。

## 构建方法

1. 用 Android Studio 打开本目录 `android_apk_webview`。
2. 等待 Android Studio 同步 Gradle 和 SDK。
3. 选择 `Build > Build Bundle(s) / APK(s) > Build APK(s)`。
4. 生成的 APK 通常在 `app/build/outputs/apk/debug/app-debug.apk`。

也可以在命令行中执行：

```bash
gradle :app:assembleDebug
```

如果本机没有 JDK、Gradle 或 Android SDK，需要先安装 Android Studio。

## 使用注意

- App 打开后加载的是复制出来的 `app/www` 界面。
- 在设置页里把服务器 IP 填成开发板 Wi-Fi IP，状态服务端口填 `8889`，语音服务端口填 `8890`。
- 手机和开发板需要在同一个局域网。

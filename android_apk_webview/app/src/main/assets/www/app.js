const ANGLE_MIN = -90;
const ANGLE_MAX = 90;
const WHEEL_ITEM_HEIGHT = 38;
const STATUS_POLL_INTERVAL_MS = 1000;
const FALL_ALARM_SOUND_SRC = "fall_alarm.mp3?v=1";
const TAB_PAGES = new Set(["home", "history", "settings"]);
const DEBUG_LABELS = ["stand", "lying", "sit", "fall", "walk", "bend"];
const WEEKDAY_OPTIONS = [
  { value: 1, text: "一" },
  { value: 2, text: "二" },
  { value: 3, text: "三" },
  { value: 4, text: "四" },
  { value: 5, text: "五" },
  { value: 6, text: "六" },
  { value: 7, text: "日" }
];

const state = {
  page: "home",
  detailAlarm: null,
  serverUrl: "",
  serverPort: "8889",
  voiceServerUrl: "",
  voicePort: "8890",
  voiceStreamPort: "8891",
  connected: false,
  detectionEnabled: false,
  detectionMode: "vision",
  trackingEnabled: false,
  debugVisionLabel: "",
  debugRadarLabel: "",
  currentPosture: "",
  currentPostureRaw: "",
  previewImage: "",
  previewImageTimestamp: 0,
  messageEvents: [],
  lastMessageTimestamp: 0,
  lastMessageToastAt: 0,
  messageClearTimestamp: 0,
  messageDraft: "",
  messageScrollTop: 0,
  reminders: [],
  reminderNotification: null,
  lastReminderNotificationAt: 0,
  reminderForm: {
    id: "",
    label: "",
    repeat: "once",
    date: "",
    time: "",
    weekdays: []
  },
  motorAngle: null,
  motorAngleValue: 0,
  motorRelativeValue: 0,
  alarmHistory: [],
  lastAlarmTimestamp: 0,
  lastAlarmStatus: false,
  pollingTimer: null,
  connecting: false,
  boardMicListening: false,
  boardMicAudio: null,
  fallAlarmAudio: null,
  boardMicAbort: null,
  boardMicQueue: [],
  boardMicPlaying: false,
  boardMicCursor: "",
  boardMicNative: false,
  phoneMicSending: false,
  phoneMicNative: false,
  phoneMediaStream: null,
  phoneMediaRecorder: null,
  phoneRecordTimer: null,
  phoneUploadControllers: [],
  lastVoiceErrorAt: 0,
  androidVoiceLastError: "",
  editingField: "",
  pollingPaused: false,
  reminderEditingActive: false,
  historyFilter: {
    startYear: "",
    startMonth: "",
    startDay: "",
    endYear: "",
    endMonth: "",
    endDay: ""
  }
};

function loadConfig() {
  let serverUrl = localStorage.getItem("serverUrl") || "127.0.0.1";
  let serverPort = localStorage.getItem("serverPort") || "8889";
  let voiceServerUrl = localStorage.getItem("voiceServerUrl") || serverUrl;
  let voicePort = localStorage.getItem("voicePort") || "8890";
  let voiceStreamPort = localStorage.getItem("voiceStreamPort") || "8891";

  if (serverPort === "8888") {
    serverPort = "8889";
    localStorage.setItem("serverPort", serverPort);
  }

  state.serverUrl = serverUrl;
  state.serverPort = serverPort;
  state.voiceServerUrl = voiceServerUrl;
  state.voicePort = voicePort;
  state.voiceStreamPort = voiceStreamPort;
}

function loadAlarmHistory() {
  try {
    state.alarmHistory = JSON.parse(localStorage.getItem("alarmHistory") || "[]");
  } catch (error) {
    state.alarmHistory = [];
  }

  const storedTimestamp = Number(localStorage.getItem("lastAlarmTimestamp") || 0);
  if (storedTimestamp > 0) {
    state.lastAlarmTimestamp = storedTimestamp;
  } else if (state.alarmHistory.length > 0 && state.alarmHistory[0].timestamp) {
    state.lastAlarmTimestamp = Number(state.alarmHistory[0].timestamp);
    localStorage.setItem("lastAlarmTimestamp", String(state.lastAlarmTimestamp));
  }
}

function saveAlarmHistory() {
  localStorage.setItem("alarmHistory", JSON.stringify(state.alarmHistory));
}

function getVoiceHost() {
  const host = state.voiceServerUrl || state.serverUrl;
  return host;
}

function getVoiceBaseUrl() {
  const host = getVoiceHost();
  return `http://${host}:${state.voicePort || "8890"}`;
}

function getVoiceStreamPort() {
  return state.voiceStreamPort || "8891";
}

function hasAndroidVoice() {
  try {
    return !!(window.AndroidVoice && window.AndroidVoice.isAvailable && window.AndroidVoice.isAvailable());
  } catch (error) {
    return false;
  }
}

function normalizeDebugLabel(label) {
  const text = String(label || "").trim().toLowerCase();
  return DEBUG_LABELS.includes(text) ? text : "";
}

function formatPostureLabel(label) {
  const labelMap = {
    stand: "站立",
    lying: "躺在地上",
    sit: "坐立",
    fall: "跌倒",
    walk: "行走",
    bend: "弯腰",
    bendover: "弯腰",
    static: "静止"
  };
  return labelMap[label] || label || "";
}

function modeText(mode = state.detectionMode) {
  return mode === "radar" ? "隐私模式" : "日常模式";
}

function todayDateValue() {
  const date = new Date();
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function nextHourTimeValue() {
  const date = new Date(Date.now() + 60 * 60 * 1000);
  return `${String(date.getHours()).padStart(2, "0")}:00`;
}

function resetReminderForm() {
  state.reminderForm = {
    id: "",
    label: "",
    repeat: "once",
    date: todayDateValue(),
    time: nextHourTimeValue(),
    weekdays: [new Date().getDay() || 7]
  };
}

function weekdayText(weekdays) {
  const values = Array.isArray(weekdays) ? weekdays.map(Number) : [];
  const map = new Map(WEEKDAY_OPTIONS.map(item => [item.value, item.text]));
  return values.filter(value => map.has(value)).map(value => map.get(value)).join("");
}

function formatReminder(reminder) {
  if (!reminder) {
    return "";
  }
  if (reminder.nextText) {
    return reminder.nextText;
  }
  if (reminder.repeat === "weekly") {
    return `每周${weekdayText(reminder.weekdays)} ${reminder.time || ""}`;
  }
  return `${reminder.date || ""} ${reminder.time || ""}`.trim();
}

function isTabPage(page = state.page) {
  return TAB_PAGES.has(page);
}

function navTitle(page = state.page) {
  const titleMap = {
    mode: "检测模式",
    control: "系统控制",
    voice: "语音对话",
    reminders: "提醒",
    messages: "双向交流",
    more: "更多功能"
  };
  return titleMap[page] || "智能检测系统";
}

function statusClass(value) {
  return value ? "status status-active" : "status status-inactive";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function showToast(message) {
  const toast = document.getElementById("toast");
  if (!toast) {
    return;
  }
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    toast.hidden = true;
  }, 2200);
}

async function checkServerStatus() {
  if (state.pollingPaused) {
    return;
  }

  const since = Number(state.lastAlarmTimestamp || 0);
  const previewSince = Number(state.previewImageTimestamp || 0);
  const messageSince = Number(state.lastMessageTimestamp || 0);
  const url = `http://${state.serverUrl}:${state.serverPort}/status?events=1&since=${since}&previewSince=${previewSince}&messageSince=${messageSince}`;
  const requestStartedAt = Date.now();

  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    if (data.previewImage !== undefined) {
      console.log("[app-preview] received", {
        seq: data.previewImageSeq || 0,
        hasImage: Boolean(data.previewImage),
        fetchMs: Date.now() - requestStartedAt,
        sourceAgeMs: data.previewImageTimestamp ? Date.now() - Number(data.previewImageTimestamp) : null,
        nodeCacheAgeMs: data.previewImageReceivedAt ? Date.now() - Number(data.previewImageReceivedAt) : null,
        chars: data.previewImage ? data.previewImage.length : 0
      });
    }
    state.connected = true;
    handleServerResponse(data);
  } catch (error) {
    state.connected = false;
    renderUnlessEditingForm();
  }
}

function handleServerResponse(data) {
  if (data.enabled !== undefined) {
    state.detectionEnabled = Boolean(data.enabled);
  }
  if (data.mode === "radar" || data.mode === "vision") {
    state.detectionMode = data.mode;
  }
  if (typeof data.tracking === "boolean") {
    state.trackingEnabled = data.tracking;
  }
  if (data.debugVisionLabel !== undefined) {
    state.debugVisionLabel = normalizeDebugLabel(data.debugVisionLabel);
  }
  if (data.debugRadarLabel !== undefined) {
    state.debugRadarLabel = normalizeDebugLabel(data.debugRadarLabel);
  }
  if (data.motorAngle !== undefined && data.motorAngle !== null) {
    state.motorAngle = Number(data.motorAngle);
  }
  if (data.label !== undefined) {
    state.currentPostureRaw = data.label || "";
    state.currentPosture = formatPostureLabel(state.currentPostureRaw);
  }
  if (data.previewImage !== undefined) {
    updatePreviewImage(data.previewImage || "", Number(data.previewImageTimestamp || Date.now()));
  } else if (Number(data.previewImageTimestamp || 0) === 0) {
    state.previewImageTimestamp = 0;
  }
  if (Array.isArray(data.reminders)) {
    state.reminders = data.reminders;
  }
  if (data.reminderNotification !== undefined) {
    handleReminderNotification(data.reminderNotification);
  }
  if (data.messageClearTimestamp !== undefined) {
    handleMessageClearTimestamp(Number(data.messageClearTimestamp || 0));
  }
  if (Array.isArray(data.messageEvents)) {
    handleMessageEvents(data.messageEvents);
  }

  const alarmEvents = extractAlarmEvents(data);
  if (alarmEvents.length > 0) {
    handleAlarmEvents(alarmEvents);
  }
  state.lastAlarmStatus = Boolean(data.alarm);
  renderUnlessEditingForm();
}

function handleReminderNotification(notification) {
  state.reminderNotification = notification || null;
  if (!notification) {
    return;
  }
  const triggeredAt = Number(notification.triggeredAt || Date.now());
  if (triggeredAt > state.lastReminderNotificationAt) {
    state.lastReminderNotificationAt = triggeredAt;
    showToast(notification.message || `提醒：${notification.label || ""}`);
  }
}

function handleMessageEvents(events) {
  const known = new Set((state.messageEvents || []).map(item => String(item.id || "")));
  const sorted = events
    .filter(item => item && typeof item === "object")
    .map(item => Object.assign({}, item, { timestamp: Number(item.timestamp || Date.now()) }))
    .sort((a, b) => a.timestamp - b.timestamp);

  sorted.forEach((event) => {
    if (!event.id || !known.has(String(event.id))) {
      state.messageEvents.push(event);
      if (event.id) {
        known.add(String(event.id));
      }
    }
    state.lastMessageTimestamp = Math.max(state.lastMessageTimestamp, Number(event.timestamp || 0));
    if (event.direction === "board_to_app" && Number(event.timestamp || 0) > state.lastMessageToastAt) {
      state.lastMessageToastAt = Number(event.timestamp || Date.now());
      showToast(`收到信息：${event.text || ""}`);
    }
  });

  if (state.messageEvents.length > 80) {
    state.messageEvents = state.messageEvents.slice(-80);
  }
}

function handleMessageClearTimestamp(timestamp) {
  if (!Number.isFinite(timestamp) || timestamp <= state.messageClearTimestamp) {
    return;
  }
  state.messageClearTimestamp = timestamp;
  state.messageEvents = [];
  state.lastMessageTimestamp = Math.max(state.lastMessageTimestamp, timestamp);
  state.lastMessageToastAt = Math.max(state.lastMessageToastAt, timestamp);
  state.messageScrollTop = 0;
}

function renderUnlessEditingForm() {
  if ((state.page === "settings" || state.page === "reminders") && state.editingField) {
    return;
  }
  if (state.page === "reminders" && state.reminderEditingActive) {
    return;
  }
  if (state.page === "messages" && String(state.editingField || "") === "message") {
    return;
  }
  render();
}

function beginReminderEditing(field = "") {
  state.reminderEditingActive = true;
  state.editingField = field || state.editingField || "reminder";
  state.pollingPaused = true;
}

function endReminderEditing() {
  state.reminderEditingActive = false;
  if (String(state.editingField || "").startsWith("reminder")) {
    state.editingField = "";
  }
  state.pollingPaused = false;
}

function updatePreviewImage(imageBase64, timestamp) {
  if (!imageBase64) {
    state.previewImage = "";
    state.previewImageTimestamp = Number(timestamp || 0);
    return;
  }

  const commaIndex = imageBase64.indexOf(",");
  const rawBase64 = imageBase64.startsWith("data:") && commaIndex >= 0
    ? imageBase64.slice(commaIndex + 1)
    : imageBase64;
  state.previewImage = imageBase64.startsWith("data:")
    ? imageBase64
    : `data:image/jpeg;base64,${rawBase64}`;
  state.previewImageTimestamp = Number(timestamp || Date.now());
}

function extractAlarmEvents(data) {
  if (Array.isArray(data.alarmEvents) && data.alarmEvents.length > 0) {
    return data.alarmEvents;
  }
  if (data.lastAlarm) {
    return [data.lastAlarm];
  }
  if (data.alarm) {
    return [data];
  }
  return [];
}

function handleAlarmEvents(events) {
  const sortedEvents = events
    .map(item => Object.assign({}, item, {
      timestamp: Number(item.timestamp || Date.now())
    }))
    .filter(item => item.timestamp > Number(state.lastAlarmTimestamp || 0))
    .sort((a, b) => a.timestamp - b.timestamp);

  sortedEvents.forEach((event) => {
    const alarmMessage = {
      type: "alarm",
      fall_detected: true,
      confidence: event.confidence !== undefined ? event.confidence : 0.9,
      mode: event.mode || state.detectionMode,
      image: event.image || null,
      timestamp: event.timestamp,
      sourceTimestamp: event.sourceTimestamp || event.timestamp
    };

    state.lastAlarmTimestamp = event.timestamp;
    localStorage.setItem("lastAlarmTimestamp", String(event.timestamp));
    addAlarmHistory(alarmMessage);
    showAlarmNotification(alarmMessage);
  });
}

function addAlarmHistory(alarm) {
  if (state.alarmHistory.length > 0 && state.alarmHistory[0].timestamp === alarm.timestamp) {
    return;
  }
  state.alarmHistory.unshift(alarm);
  saveAlarmHistory();
}

function showAlarmNotification(alarm) {
  localStorage.setItem("currentAlarm", JSON.stringify(alarm));
  playFallAlarmSound();
  try {
    if (navigator.vibrate) {
      navigator.vibrate([600, 250, 600]);
    }
  } catch (error) {
    console.warn("vibrate failed", error);
  }

  if (window.confirm("检测到摔倒，请确认是否收到。")) {
    sendControlCommand("ack_alarm", {
      alarmTimestamp: Number(alarm.sourceTimestamp || alarm.timestamp || Date.now())
    });
  }
}

function playFallAlarmSound() {
  try {
    if (hasAndroidVoice() && window.AndroidVoice.playFallAlarm) {
      const nativeStarted = window.AndroidVoice.playFallAlarm();
      if (nativeStarted) {
        return;
      }
    }

    if (state.fallAlarmAudio) {
      state.fallAlarmAudio.pause();
      state.fallAlarmAudio.currentTime = 0;
      state.fallAlarmAudio = null;
    }

    const audio = new Audio(FALL_ALARM_SOUND_SRC);
    audio.preload = "auto";
    audio.volume = 1.0;
    state.fallAlarmAudio = audio;

    const cleanup = () => {
      if (state.fallAlarmAudio === audio) {
        state.fallAlarmAudio = null;
      }
    };
    audio.addEventListener("ended", cleanup, { once: true });
    audio.addEventListener("error", cleanup, { once: true });

    window.setTimeout(() => {
      if (state.fallAlarmAudio === audio) {
        audio.pause();
        state.fallAlarmAudio = null;
      }
    }, 5200);

    audio.play().catch((error) => {
      console.warn("fall alarm sound failed", error);
      cleanup();
    });
  } catch (error) {
    console.warn("fall alarm sound error", error);
  }
}

function startPolling() {
  if (state.pollingTimer) {
    return;
  }
  state.pollingTimer = setInterval(checkServerStatus, STATUS_POLL_INTERVAL_MS);
  checkServerStatus();
}

function stopPolling() {
  if (state.pollingTimer) {
    clearInterval(state.pollingTimer);
    state.pollingTimer = null;
  }
  state.connected = false;
  render();
}

async function sendControlCommand(action, params = {}) {
  const url = `http://${state.serverUrl}:${state.serverPort}/status`;
  const data = Object.assign({ action, timestamp: Date.now() }, params);
  console.log("send control command:", data);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    await response.json().catch(() => ({}));
    await checkServerStatus();
  } catch (error) {
    console.warn("send control command failed", error);
    showToast("发送失败");
  }
}

function navigate(page) {
  const previousPage = state.page;
  if (page !== state.page) {
    stopVoiceStreams();
    if (state.page === "reminders") {
      endReminderEditing();
    }
    if (state.page === "messages") {
      state.editingField = "";
      state.pollingPaused = false;
    }
  }
  if (page === "messages" && previousPage !== "messages") {
    state.messageScrollTop = Number.MAX_SAFE_INTEGER;
  }
  state.page = page;
  state.detailAlarm = null;
  render();
}

function openDetail(index) {
  const alarms = filteredAlarmHistory();
  state.detailAlarm = alarms[index] || null;
  state.page = "detail";
  render();
}

function goBackFromSubpage() {
  handleBackNavigation();
}

function handleBackNavigation() {
  if (state.page === "home") {
    return false;
  }
  if (state.page === "reminders") {
    endReminderEditing();
  }
  if (state.page === "messages") {
    state.editingField = "";
    state.pollingPaused = false;
  }
  if (state.page === "detail") {
    state.page = "history";
  } else if (state.page === "more") {
    state.page = "settings";
  } else {
    state.page = "home";
  }
  state.detailAlarm = null;
  stopVoiceStreams();
  render();
  return true;
}

function render() {
  const root = document.getElementById("app");
  if (!root) {
    return;
  }

  if (state.page === "messages") {
    const currentList = document.querySelector(".message-list");
    if (currentList) {
      state.messageScrollTop = currentList.scrollTop;
    }
  }

  const tabPage = isTabPage();
  root.innerHTML = `
    ${renderTopbar()}
    <section class="app-shell ${tabPage ? "with-tabbar" : "without-tabbar"}">
      ${renderPage()}
    </section>
    ${tabPage ? renderTabbar() : ""}
  `;
  bindPageEvents();

  if (state.page === "messages") {
    const nextList = document.querySelector(".message-list");
    if (nextList) {
      nextList.scrollTop = Math.min(
        Number(state.messageScrollTop || 0),
        Math.max(0, nextList.scrollHeight - nextList.clientHeight)
      );
    }
  }
}

function renderTopbar() {
  return `
    <header class="mini-nav">
      <div class="mini-nav-side">
        ${isTabPage() ? "" : `<button class="mini-back" data-action="back">‹</button>`}
      </div>
      <div class="mini-nav-title">${navTitle()}</div>
      <div class="mini-capsule" aria-hidden="true">
        <span class="mini-dot">•••</span>
        <span class="mini-divider"></span>
        <span class="mini-ring"></span>
      </div>
    </header>
  `;
}

function renderHeading(title, subtitle, showConnection = true) {
  return `
    <div class="page-heading">
      <div>
        <div class="page-title">${title}</div>
        <div class="page-subtitle">${subtitle}</div>
      </div>
      ${showConnection ? `<div class="connect-pill ${state.connected ? "online" : "offline"}">${state.connected ? "已连接" : "未连接"}</div>` : ""}
    </div>
  `;
}

function renderPage() {
  if (state.page === "mode") {
    return renderModePage();
  }
  if (state.page === "control") {
    return renderControlPage();
  }
  if (state.page === "voice") {
    return renderVoicePage();
  }
  if (state.page === "reminders") {
    return renderRemindersPage();
  }
  if (state.page === "messages") {
    return renderMessagesPage();
  }
  if (state.page === "more") {
    return renderMorePage();
  }
  if (state.page === "history") {
    return renderHistoryPage();
  }
  if (state.page === "detail") {
    return renderDetailPage();
  }
  if (state.page === "settings") {
    return renderSettingsPage();
  }
  return renderHomePage();
}

function renderHomePage() {
  return `
    ${renderHeading("智能检测系统", "跌倒报警、摄像头控制与语音交互")}
    <div class="card">
      <div class="section-title">系统状态</div>
      <div class="status-grid">
        <div class="status-cell">
          <span class="status-label">检测状态</span>
          <span class="${statusClass(state.detectionEnabled)}">${state.detectionEnabled ? "已启用" : "已禁用"}</span>
        </div>
        <div class="status-cell">
          <span class="status-label">跟踪状态</span>
          <span class="${statusClass(state.trackingEnabled)}">${state.trackingEnabled ? "开启" : "关闭"}</span>
        </div>
        <div class="status-cell">
          <span class="status-label">人物姿势</span>
          <span class="status-value">${escapeHtml(state.currentPosture || "-")}</span>
        </div>
        <div class="status-cell">
          <span class="status-label">当前模式</span>
          <span class="status-value">${modeText()}</span>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="section-title">现场图像</div>
      <div class="preview-frame">
        ${state.previewImage
          ? `<img class="preview-image" src="${state.previewImage}" alt="现场图像">`
          : `<div class="preview-empty">暂无图像</div>`}
      </div>
    </div>
    <div class="feature-list">
      <button class="feature-item" data-page="mode">
        <div class="feature-icon">◎</div>
        <div class="feature-copy">
          <div class="feature-title">检测模式</div>
          <div class="feature-desc">切换日常 / 隐私检测方式</div>
        </div>
        <div class="feature-arrow">›</div>
      </button>
      <button class="feature-item" data-page="control">
        <div class="feature-icon">▣</div>
        <div class="feature-copy">
          <div class="feature-title">系统控制</div>
          <div class="feature-desc">检测、跟踪与云台角度控制</div>
        </div>
        <div class="feature-arrow">›</div>
      </button>
      <button class="feature-item" data-page="voice">
        <div class="feature-icon">◉</div>
        <div class="feature-copy">
          <div class="feature-title">语音对话</div>
          <div class="feature-desc">开发板监听与远程语音通话</div>
        </div>
        <div class="feature-arrow">›</div>
      </button>
      <button class="feature-item" data-page="reminders">
        <div class="feature-icon">◷</div>
        <div class="feature-copy">
          <div class="feature-title">提醒</div>
          <div class="feature-desc">管理一次性和每周重复提醒</div>
        </div>
        <div class="feature-arrow">›</div>
      </button>
      <button class="feature-item" data-page="messages">
        <div class="feature-icon">●</div>
        <div class="feature-copy">
          <div class="feature-title">双向交流</div>
          <div class="feature-desc">手机和开发板之间发送交流信息</div>
        </div>
        <div class="feature-arrow">›</div>
      </button>
    </div>
    <div class="card">
      <div class="section-title">系统信息</div>
      <div class="info-row"><span>当前开发板</span><span>RK3588开发板</span></div>
      <div class="info-row"><span>控制端口</span><span>${escapeHtml(state.serverPort)}</span></div>
      <div class="info-row"><span>语音端口</span><span>${escapeHtml(state.voicePort)}</span></div>
      <div class="info-row"><span>实时语音端口</span><span>${escapeHtml(state.voiceStreamPort)}</span></div>
    </div>
  `;
}

function renderModePage() {
  return `
    ${renderHeading("检测模式", "选择系统当前使用的图像检测方式", true, true)}
    <div class="card">
      <div class="section-title">当前状态</div>
      <div class="info-row"><span>连接状态</span><span>${state.connected ? "已连接" : "未连接"}</span></div>
      <div class="info-row"><span>当前模式</span><span>${modeText()}</span></div>
    </div>
    <button class="mode-card ${state.detectionMode === "radar" ? "active" : ""}" data-mode="radar">
      <div class="mode-icon">◌</div>
      <div class="mode-copy">
        <div class="mode-title">隐私模式</div>
        <div class="mode-desc">关闭视觉画面显示，仅保留雷达姿态判断。</div>
      </div>
      ${state.detectionMode === "radar" ? `<div class="mode-badge">使用中</div>` : ""}
    </button>
    <button class="mode-card ${state.detectionMode === "vision" ? "active" : ""}" data-mode="vision">
      <div class="mode-icon">◉</div>
      <div class="mode-copy">
        <div class="mode-title">日常模式</div>
        <div class="mode-desc">开启视觉识别与雷达融合，用于日常跌倒检测。</div>
      </div>
      ${state.detectionMode === "vision" ? `<div class="mode-badge">使用中</div>` : ""}
    </button>
  `;
}

function renderControlPage() {
  return `
    ${renderHeading("系统控制", "检测开关、自动跟踪与云台角度", false, true)}
    <div class="card">
      <div class="section-title">运行控制</div>
      <button class="switch-row" data-action="toggle-detection">
        <div>
          <div class="switch-title">跌倒检测</div>
          <div class="switch-desc">控制开发板是否进行跌倒检测</div>
        </div>
        <div class="switch ${state.detectionEnabled ? "on" : ""}"></div>
      </button>
      <button class="switch-row" data-action="toggle-tracking">
        <div>
          <div class="switch-title">自动跟踪</div>
          <div class="switch-desc">开启后由系统控制摄像头方向</div>
        </div>
        <div class="switch ${state.trackingEnabled ? "on" : ""}"></div>
      </button>
    </div>
    <div class="card">
      <div class="section-title">开发板电源</div>
      <div class="button-group">
        <button class="btn btn-secondary" data-action="reboot-board">重启开发板</button>
        <button class="btn btn-danger" data-action="poweroff-board">关闭开发板</button>
      </div>
    </div>
    <div class="card">
      <div class="section-title">云台角度</div>
      <div class="wheel-block">
        <div class="wheel-header">
          <div class="form-label">旋转到指定角度</div>
          <div class="wheel-value" id="absoluteValue">${state.motorAngleValue}°</div>
        </div>
        <div id="absoluteWheel" class="angle-wheel ${state.trackingEnabled ? "disabled" : ""}">${renderWheelItems()}</div>
        <button class="btn btn-primary" data-action="set-motor-angle" ${state.trackingEnabled ? "disabled" : ""}>旋转到该角度</button>
      </div>
      <button class="btn btn-secondary" data-action="set-motor-zero" ${state.trackingEnabled ? "disabled" : ""}>当前位置设为0</button>
      <div class="wheel-block">
        <div class="wheel-header">
          <div class="form-label">相对角度移动</div>
          <div class="wheel-value" id="relativeValue">${state.motorRelativeValue}°</div>
        </div>
        <div id="relativeWheel" class="angle-wheel ${state.trackingEnabled ? "disabled" : ""}">${renderWheelItems()}</div>
        <button class="btn btn-primary" data-action="move-motor-relative" ${state.trackingEnabled ? "disabled" : ""}>按该角度相对移动</button>
      </div>
    </div>
  `;
}

function renderWheelItems() {
  const items = ['<div class="wheel-spacer"></div>'];
  for (let value = ANGLE_MIN; value <= ANGLE_MAX; value += 1) {
    items.push(`<div class="wheel-item" data-angle="${value}">${value}°</div>`);
  }
  items.push('<div class="wheel-spacer"></div>');
  return items.join("");
}

function renderVoicePage() {
  const active = state.boardMicListening || state.phoneMicSending;
  const nativeReady = hasAndroidVoice();
  const voiceChannelText = nativeReady
    ? `实时通道：${escapeHtml(getVoiceHost())}:${escapeHtml(getVoiceStreamPort())}`
    : `分片通道：${escapeHtml(getVoiceBaseUrl())}`;
  return `
    ${renderHeading("语音对话", "手机与开发板进行远程沟通", false, true)}
    <div class="card">
      <div class="section-title">语音状态</div>
      <div class="voice-state ${active ? "active" : ""}">
        <div class="voice-dot"></div>
        <div>
          <div class="voice-title">${voiceStatusText()}</div>
          <div class="voice-desc">${voiceChannelText}</div>
        </div>
      </div>
    </div>
    <div class="voice-action-card">
      <div class="voice-action-icon">⌕</div>
      <div class="voice-action-copy">
        <div class="voice-action-title">环境语音监听</div>
        <div class="voice-action-desc">接收开发板采集到的现场声音。</div>
      </div>
      <button class="voice-button ${state.boardMicListening ? "active" : ""}" data-action="toggle-board-mic">
        ${state.boardMicListening ? "停止监听" : "开始监听"}
      </button>
    </div>
    <div class="voice-action-card">
      <div class="voice-action-icon">●</div>
      <div class="voice-action-copy">
        <div class="voice-action-title">远程语音通话</div>
        <div class="voice-action-desc">将手机端声音发送到看护设备播放。</div>
      </div>
      <button class="voice-button ${state.phoneMicSending ? "active" : ""}" data-action="toggle-phone-mic">
        ${state.phoneMicSending ? "停止发送" : "开始发送"}
      </button>
    </div>
  `;
}

function voiceStatusText() {
  if (state.boardMicListening && state.phoneMicSending) {
    return "双向语音中";
  }
  if (state.boardMicListening) {
    return "正在收听居住环境情况";
  }
  if (state.phoneMicSending) {
    return "正在发送声音";
  }
  return "语音关闭";
}

function renderRemindersPage() {
  const reminders = Array.isArray(state.reminders) ? state.reminders : [];
  const form = state.reminderForm;
  const isWeekly = form.repeat === "weekly";
  return `
    ${renderHeading("提醒", "建立和管理提醒队列", false, true)}
    <div class="card">
      <div class="section-title">提醒队列</div>
      ${reminders.length
        ? reminders.map(renderReminderItem).join("")
        : `<div class="empty-text">暂无提醒</div>`}
    </div>
    <div class="card">
      <div class="section-title">${form.id ? "修改提醒" : "新增提醒"}</div>
      <div class="form-item">
        <label class="form-label" for="reminderLabel">提醒标签</label>
        <input id="reminderLabel" class="form-input" type="text" value="${escapeHtml(form.label)}" placeholder="例如：吃药" data-reminder-field="label">
      </div>
      <div class="form-item">
        <div class="form-label">提醒类型</div>
        <div class="segmented">
          <button class="segment ${form.repeat === "once" ? "active" : ""}" data-reminder-repeat="once">提醒一次</button>
          <button class="segment ${form.repeat === "weekly" ? "active" : ""}" data-reminder-repeat="weekly">每周重复</button>
        </div>
      </div>
      ${isWeekly ? "" : `
        <div class="form-item">
          <label class="form-label" for="reminderDate">提醒日期</label>
          <input id="reminderDate" class="form-input" type="date" value="${escapeHtml(form.date || todayDateValue())}" data-reminder-field="date">
        </div>
      `}
      <div class="form-item">
        <label class="form-label" for="reminderTime">提醒时间</label>
        <input id="reminderTime" class="form-input" type="time" value="${escapeHtml(form.time || nextHourTimeValue())}" data-reminder-field="time">
      </div>
      ${isWeekly ? `
        <div class="form-item">
          <div class="form-label">重复星期</div>
          <div class="weekday-row">
            ${WEEKDAY_OPTIONS.map(item => `
              <button class="weekday-chip ${(form.weekdays || []).map(Number).includes(item.value) ? "active" : ""}" data-reminder-weekday="${item.value}">
                ${item.text}
              </button>
            `).join("")}
          </div>
        </div>
      ` : ""}
      <div class="button-group">
        <button class="btn btn-primary" data-action="save-reminder">${form.id ? "保存修改" : "新增提醒"}</button>
        <button class="btn btn-secondary" data-action="reset-reminder">清空</button>
      </div>
    </div>
  `;
}

function renderReminderItem(reminder) {
  const enabled = reminder.enabled !== false && reminder.done !== true;
  return `
    <div class="reminder-item">
      <div class="reminder-copy">
        <div class="reminder-title">${escapeHtml(reminder.label || "提醒")}</div>
        <div class="reminder-desc">${escapeHtml(formatReminder(reminder))}</div>
      </div>
      <span class="${enabled ? "status status-active" : "status status-inactive"}">${enabled ? "启用" : "关闭"}</span>
      <button class="mini-button" data-reminder-edit="${escapeHtml(reminder.id || "")}">编辑</button>
      <button class="mini-button danger" data-reminder-delete="${escapeHtml(reminder.id || "")}">删除</button>
    </div>
  `;
}

function renderMessagesPage() {
  const messages = Array.isArray(state.messageEvents) ? state.messageEvents.slice(-80) : [];
  return `
    ${renderHeading("双向交流", "手机和开发板之间发送交流信息", false, true)}
    <div class="card message-card">
      <div class="section-title">聊天记录</div>
      <div class="message-list">
        ${messages.length
          ? messages.map(renderMessageItem).join("")
          : `<div class="empty-text">暂无信息</div>`}
      </div>
      <button class="btn btn-secondary" data-action="clear-message-history" ${messages.length ? "" : "disabled"}>清除聊天记录</button>
    </div>
    <div class="card">
      <div class="section-title">发送到开发板</div>
      <textarea class="message-input" data-message-input placeholder="请输入要发送到开发板的信息">${escapeHtml(state.messageDraft)}</textarea>
      <div class="button-group">
        <button class="btn btn-primary" data-action="send-message">发送信息</button>
        <button class="btn btn-secondary" data-action="clear-message">清空输入</button>
      </div>
    </div>
  `;
}

function renderMessageItem(message) {
  const isApp = message.direction === "app_to_board";
  return `
    <div class="message-item ${isApp ? "from-app" : "from-board"}">
      <div class="message-bubble">
        <div class="message-meta">${isApp ? "App" : "开发板"} · ${formatTime(message.timestamp)}</div>
        <div class="message-text">${escapeHtml(message.text || "")}</div>
      </div>
    </div>
  `;
}

function renderHistoryPage() {
  const alarms = filteredAlarmHistory();
  return `
    <h2 class="wx-title">报警历史</h2>
    <div class="time-filter">
      <div class="quick-select">
        <button class="quick-item" data-days="1">近一天</button>
        <button class="quick-item" data-days="3">近三天</button>
        <button class="quick-item" data-days="30">近一个月</button>
      </div>
      ${renderDateRow("开始时间", "start")}
      ${renderDateRow("结束时间", "end")}
      <div class="filter-actions">
        <button class="btn btn-secondary" data-action="reset-filter">重置</button>
        <button class="btn btn-primary" data-action="apply-filter">应用筛选</button>
      </div>
    </div>
    ${alarms.length > 0 ? `
      <div class="list">
        ${alarms.map((item, index) => `
          <button class="list-item" data-detail-index="${index}">
            <div>
              <div class="alarm-title">检测到人体摔倒</div>
              <div class="alarm-time">${formatTime(item.timestamp)}</div>
              <div class="alarm-confidence">置信度：${Math.round(Number(item.confidence || 0) * 100)}%</div>
            </div>
            <span class="status status-inactive">已报警</span>
          </button>
        `).join("")}
      </div>
    ` : `<div class="empty">暂无报警记录</div>`}
    <div class="hint">注：报警记录仅保存在本地，清除应用数据会导致记录丢失</div>
  `;
}

function renderDateRow(label, prefix) {
  const filter = state.historyFilter;
  return `
    <div class="filter-row">
      <div class="filter-label">${label}</div>
      <div class="date-inputs">
        <input class="date-input" type="number" inputmode="numeric" placeholder="年" data-filter="${prefix}Year" value="${escapeHtml(filter[`${prefix}Year`])}" maxlength="4">
        <span>-</span>
        <input class="date-input" type="number" inputmode="numeric" placeholder="月" data-filter="${prefix}Month" value="${escapeHtml(filter[`${prefix}Month`])}" maxlength="2">
        <span>-</span>
        <input class="date-input" type="number" inputmode="numeric" placeholder="日" data-filter="${prefix}Day" value="${escapeHtml(filter[`${prefix}Day`])}" maxlength="2">
      </div>
    </div>
  `;
}

function filteredAlarmHistory() {
  const f = state.historyFilter;
  const hasStartDate = f.startYear && f.startMonth && f.startDay;
  const hasEndDate = f.endYear && f.endMonth && f.endDay;
  if (!hasStartDate && !hasEndDate) {
    return state.alarmHistory;
  }

  return state.alarmHistory.filter((item) => {
    const itemDate = new Date(item.timestamp);
    let startOk = true;
    let endOk = true;
    if (hasStartDate) {
      startOk = itemDate >= new Date(`${f.startYear}-${f.startMonth}-${f.startDay}`);
    }
    if (hasEndDate) {
      const end = new Date(`${f.endYear}-${f.endMonth}-${f.endDay}`);
      end.setHours(23, 59, 59, 999);
      endOk = itemDate <= end;
    }
    return startOk && endOk;
  });
}

function renderDetailPage() {
  const alarm = state.detailAlarm || readCurrentAlarm();
  if (!alarm) {
    return `
      <h2 class="wx-title">报警详情</h2>
      <div class="empty">未找到报警记录</div>
    `;
  }
  return `
    <h2 class="wx-title">报警详情</h2>
    <div class="card">
      <div class="text-row"><span>报警时间</span><span>${formatTime(alarm.timestamp)}</span></div>
      <div class="text-row"><span>置信度</span><span>${Math.round(Number(alarm.confidence || 0) * 100)}%</span></div>
      <div class="text-row"><span>检测模式</span><span>${modeText(alarm.mode)}</span></div>
    </div>
    <div class="card">
      <div class="section-title">跌倒图片</div>
      <div class="image-container">
        ${alarm.image
          ? `<img class="fall-image" src="${normalizeImage(alarm.image)}" alt="跌倒图片">`
          : `<div class="image-placeholder"><div class="placeholder-text">暂无图片</div><div class="placeholder-subtext">实际报警图片会显示在这里</div></div>`}
      </div>
    </div>
    <button class="btn btn-primary" data-action="back">返回</button>
  `;
}

function readCurrentAlarm() {
  try {
    return JSON.parse(localStorage.getItem("currentAlarm") || "null");
  } catch (error) {
    return null;
  }
}

function normalizeImage(image) {
  const text = String(image || "");
  return text.startsWith("data:") ? text : `data:image/jpeg;base64,${text}`;
}

function renderSettingsPage() {
  return `
    <h2 class="wx-title">设备连接设置</h2>
    <div class="card">
      ${renderInput("报警/控制服务器IP地址", "serverUrl", state.serverUrl, "例如：127.0.0.1 或电脑IP")}
      ${renderInput("报警/控制端口号", "serverPort", state.serverPort, "例如：8889", "number")}
      ${renderInput("语音服务IP地址", "voiceServerUrl", state.voiceServerUrl, "例如：RK3588的IP")}
      ${renderInput("语音服务端口号", "voicePort", state.voicePort, "例如：8890", "number")}
      ${renderInput("实时语音端口号", "voiceStreamPort", state.voiceStreamPort, "例如：8891", "number")}
      <div class="button-group">
        <button class="btn btn-primary" data-action="connect" ${state.connecting || state.connected ? "disabled" : ""}>
          ${state.connecting ? "连接中..." : (state.connected ? "已连接" : "开始连接")}
        </button>
        <button class="btn btn-secondary" data-action="disconnect" ${!state.connecting && !state.connected ? "disabled" : ""}>断开连接</button>
      </div>
    </div>
    <div class="feature-list">
      <button class="feature-item" data-page="more">
        <div class="feature-icon">●</div>
        <div class="feature-copy">
          <div class="feature-title">更多功能</div>
          <div class="feature-desc">扩展功能</div>
        </div>
        <div class="feature-arrow">›</div>
      </button>
    </div>
    <div class="card">
      <div class="section-title">关于</div>
      <div class="text-row"><span>应用名称</span><span>智能检测系统</span></div>
      <div class="text-row"><span>版本</span><span>2.0.0</span></div>
      <div class="text-row"><span>描述</span><span>RK3588开发板人体摔倒检测系统</span></div>
    </div>
  `;
}

function renderMorePage() {
  return `
    ${renderHeading("更多功能", "扩展功能", false, true)}
    ${renderDebugOverrideSettings()}
  `;
}

function renderDebugOverrideSettings() {
  return `
    <div class="card">
      <div class="section-title">调试功能</div>
      <div class="debug-row">
        <div class="debug-row-label">Vision</div>
        <div class="debug-button-grid">
          ${renderDebugButtons("vision", state.debugVisionLabel)}
        </div>
      </div>
      <div class="debug-row">
        <div class="debug-row-label">Radar</div>
        <div class="debug-button-grid">
          ${renderDebugButtons("radar", state.debugRadarLabel)}
        </div>
      </div>
      <div class="debug-help">再次点击已选按钮可恢复正常识别。</div>
    </div>
  `;
}

function renderDebugButtons(kind, selectedLabel) {
  return DEBUG_LABELS.map((label) => `
    <button class="debug-button ${selectedLabel === label ? "active" : ""}" data-debug-kind="${kind}" data-debug-label="${label}">
      ${label}
    </button>
  `).join("");
}

function renderInput(label, key, value, placeholder, type = "text") {
  return `
    <div class="form-item">
      <label class="form-label" for="${key}">${label}</label>
      <input id="${key}" class="form-input" type="${type}" value="${escapeHtml(value)}" placeholder="${escapeHtml(placeholder)}" data-setting="${key}">
    </div>
  `;
}

function renderTabbar() {
  const tabs = [
    { page: "home", text: "首页", icon: "images/home.png", activeIcon: "images/home-active.png" },
    { page: "history", text: "历史", icon: "images/history.png", activeIcon: "images/history-active.png" },
    { page: "settings", text: "设置", icon: "images/settings.png", activeIcon: "images/settings-active.png" }
  ];
  return `
    <nav class="nav">
      ${tabs.map((tab) => {
        const active = state.page === tab.page;
        return `
          <button class="nav-item ${active ? "active" : ""}" data-page="${tab.page}">
            <img class="nav-icon" src="${active ? tab.activeIcon : tab.icon}" alt="">
            <span>${tab.text}</span>
          </button>
        `;
      }).join("")}
    </nav>
  `;
}

function bindPageEvents() {
  document.querySelectorAll("[data-page]").forEach((button) => {
    button.addEventListener("click", () => navigate(button.dataset.page));
  });

  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", () => selectMode(button.dataset.mode));
  });

  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => handleAction(button.dataset.action));
  });

  document.querySelectorAll("[data-debug-kind][data-debug-label]").forEach((button) => {
    button.addEventListener("click", () => toggleDebugOverride(button.dataset.debugKind, button.dataset.debugLabel));
  });

  document.querySelectorAll("[data-reminder-field]").forEach((input) => {
    const beginEdit = () => beginReminderEditing(`reminder.${input.dataset.reminderField}`);
    input.addEventListener("pointerdown", beginEdit);
    input.addEventListener("touchstart", beginEdit, { passive: true });
    input.addEventListener("mousedown", beginEdit);
    input.addEventListener("focus", () => {
      beginEdit();
    });
    input.addEventListener("input", () => {
      state.reminderForm[input.dataset.reminderField] = input.value.trim();
    });
    input.addEventListener("change", () => {
      state.reminderForm[input.dataset.reminderField] = input.value.trim();
    });
    input.addEventListener("blur", () => {
      state.reminderForm[input.dataset.reminderField] = input.value.trim();
      if (!state.reminderEditingActive) {
        state.editingField = "";
        state.pollingPaused = false;
        setTimeout(checkServerStatus, 100);
      }
    });
  });

  document.querySelectorAll("[data-reminder-repeat]").forEach((button) => {
    button.addEventListener("click", () => {
      state.reminderForm.repeat = button.dataset.reminderRepeat;
      if (!state.reminderForm.date) {
        state.reminderForm.date = todayDateValue();
      }
      if (!state.reminderForm.time) {
        state.reminderForm.time = nextHourTimeValue();
      }
      if (!state.reminderForm.weekdays || state.reminderForm.weekdays.length === 0) {
        state.reminderForm.weekdays = [new Date().getDay() || 7];
      }
      render();
    });
  });

  document.querySelectorAll("[data-reminder-weekday]").forEach((button) => {
    button.addEventListener("click", () => toggleReminderWeekday(Number(button.dataset.reminderWeekday)));
  });

  document.querySelectorAll("[data-reminder-edit]").forEach((button) => {
    button.addEventListener("click", () => editReminder(button.dataset.reminderEdit));
  });

  document.querySelectorAll("[data-reminder-delete]").forEach((button) => {
    button.addEventListener("click", () => deleteReminder(button.dataset.reminderDelete));
  });

  document.querySelectorAll("[data-message-input]").forEach((input) => {
    input.addEventListener("focus", () => {
      state.editingField = "message";
      state.pollingPaused = true;
    });
    input.addEventListener("input", () => {
      state.messageDraft = input.value;
    });
    input.addEventListener("blur", () => {
      state.messageDraft = input.value;
      state.editingField = "";
      state.pollingPaused = false;
      setTimeout(checkServerStatus, 100);
    });
  });

  const messageList = document.querySelector(".message-list");
  if (messageList) {
    messageList.addEventListener("scroll", () => {
      state.messageScrollTop = messageList.scrollTop;
    }, { passive: true });
  }

  document.querySelectorAll("[data-setting]").forEach((input) => {
    input.addEventListener("focus", () => {
      state.editingField = input.dataset.setting;
      state.pollingPaused = true;
    });
    input.addEventListener("input", () => {
      state[input.dataset.setting] = input.value.trim();
    });
    input.addEventListener("blur", () => {
      state[input.dataset.setting] = input.value.trim();
      state.editingField = "";
      state.pollingPaused = false;
      saveConfig();
      setTimeout(checkServerStatus, 100);
    });
  });

  document.querySelectorAll("[data-filter]").forEach((input) => {
    input.addEventListener("input", () => {
      state.historyFilter[input.dataset.filter] = input.value.trim();
    });
  });

  document.querySelectorAll("[data-days]").forEach((button) => {
    button.addEventListener("click", () => selectQuickRange(Number(button.dataset.days)));
  });

  document.querySelectorAll("[data-detail-index]").forEach((button) => {
    button.addEventListener("click", () => openDetail(Number(button.dataset.detailIndex)));
  });

  if (state.page === "control") {
    initWheel("absoluteWheel", state.motorAngleValue, (value) => {
      state.motorAngleValue = value;
      const label = document.getElementById("absoluteValue");
      if (label) {
        label.textContent = `${value}°`;
      }
    });
    initWheel("relativeWheel", state.motorRelativeValue, (value) => {
      state.motorRelativeValue = value;
      const label = document.getElementById("relativeValue");
      if (label) {
        label.textContent = `${value}°`;
      }
    });
  }
}

function handleAction(action) {
  if (action === "back") {
    goBackFromSubpage();
  } else if (action === "toggle-detection") {
    toggleDetection();
  } else if (action === "toggle-tracking") {
    toggleTracking();
  } else if (action === "set-motor-angle") {
    setMotorAngle();
  } else if (action === "set-motor-zero") {
    setMotorCurrentZero();
  } else if (action === "move-motor-relative") {
    moveMotorRelative();
  } else if (action === "poweroff-board") {
    poweroffBoard();
  } else if (action === "reboot-board") {
    rebootBoard();
  } else if (action === "save-reminder") {
    saveReminder();
  } else if (action === "reset-reminder") {
    endReminderEditing();
    resetReminderForm();
    render();
  } else if (action === "toggle-board-mic") {
    toggleBoardMicListen();
  } else if (action === "toggle-phone-mic") {
    togglePhoneMicSend();
  } else if (action === "send-message") {
    sendMessageToBoard();
  } else if (action === "clear-message") {
    state.messageDraft = "";
    render();
  } else if (action === "clear-message-history") {
    clearMessageHistory();
  } else if (action === "connect") {
    startConnection();
  } else if (action === "disconnect") {
    stopConnection();
  } else if (action === "reset-filter") {
    resetFilter();
  } else if (action === "apply-filter") {
    render();
  }
}

function initWheel(id, value, onChange) {
  const wheel = document.getElementById(id);
  if (!wheel || state.trackingEnabled) {
    return;
  }

  let timer = null;
  const clampIndex = (index) => Math.max(0, Math.min(ANGLE_MAX - ANGLE_MIN, index));
  const indexFromValue = (angle) => clampIndex(Math.round(Number(angle)) - ANGLE_MIN);
  const valueFromScroll = () => ANGLE_MIN + clampIndex(Math.round(wheel.scrollTop / WHEEL_ITEM_HEIGHT));
  wheel.scrollTop = indexFromValue(value) * WHEEL_ITEM_HEIGHT;

  wheel.addEventListener("scroll", () => {
    const selected = valueFromScroll();
    onChange(selected);
    clearTimeout(timer);
    timer = setTimeout(() => {
      wheel.scrollTo({ top: indexFromValue(selected) * WHEEL_ITEM_HEIGHT, behavior: "smooth" });
    }, 120);
  });

  wheel.querySelectorAll(".wheel-item").forEach((item) => {
    item.addEventListener("click", () => {
      const angle = Number(item.dataset.angle);
      onChange(angle);
      wheel.scrollTo({ top: indexFromValue(angle) * WHEEL_ITEM_HEIGHT, behavior: "smooth" });
    });
  });
}

function ensureTrackingOff() {
  if (!state.trackingEnabled) {
    return true;
  }
  showToast("请先关闭跟踪");
  return false;
}

function toggleDetection() {
  const enabled = !state.detectionEnabled;
  state.detectionEnabled = enabled;
  render();
  sendControlCommand(enabled ? "enable_detection" : "disable_detection");
}

function toggleTracking() {
  const enabled = !state.trackingEnabled;
  state.trackingEnabled = enabled;
  render();
  sendControlCommand(enabled ? "enable_tracking" : "disable_tracking");
}

function poweroffBoard() {
  if (!window.confirm("确定要关闭开发板吗？")) {
    return;
  }
  sendControlCommand("system_poweroff", { source: "app" });
  showToast("已发送关机指令");
}

function rebootBoard() {
  if (!window.confirm("确定要重启开发板吗？")) {
    return;
  }
  sendControlCommand("system_reboot", { source: "app" });
  showToast("已发送重启指令");
}

function sendMessageToBoard() {
  const text = String(state.messageDraft || "").trim();
  if (!text) {
    showToast("请输入要发送的信息");
    return;
  }
  sendControlCommand("message_to_board", { text, source: "app" });
  state.messageDraft = "";
  state.editingField = "";
  state.pollingPaused = false;
  showToast("已发送信息");
  render();
}

function clearMessageHistory() {
  if (!window.confirm("确定要清除聊天记录吗？")) {
    return;
  }
  state.messageEvents = [];
  state.messageScrollTop = 0;
  render();
  sendControlCommand("clear_messages", { source: "app" });
  showToast("已清除聊天记录");
}

function selectMode(mode) {
  if (mode !== "radar" && mode !== "vision") {
    return;
  }
  if (mode === state.detectionMode) {
    return;
  }
  state.detectionMode = mode;
  render();
  sendControlCommand(mode === "radar" ? "set_radar_mode" : "set_vision_mode");
}

function toggleDebugOverride(kind, label) {
  const cleanLabel = normalizeDebugLabel(label);
  if (!cleanLabel) {
    return;
  }
  if (kind === "vision") {
    state.debugVisionLabel = state.debugVisionLabel === cleanLabel ? "" : cleanLabel;
  } else if (kind === "radar") {
    state.debugRadarLabel = state.debugRadarLabel === cleanLabel ? "" : cleanLabel;
  } else {
    return;
  }
  render();
  sendControlCommand("set_debug_overrides", {
    source: "app",
    debugVisionLabel: state.debugVisionLabel,
    debugRadarLabel: state.debugRadarLabel
  });
}

function setMotorAngle() {
  if (!ensureTrackingOff()) {
    return;
  }
  const angle = clampAngle(state.motorAngleValue);
  state.motorAngle = angle;
  render();
  sendControlCommand("set_motor_angle", { angle });
}

function setMotorCurrentZero() {
  if (!ensureTrackingOff()) {
    return;
  }
  state.motorAngle = 0;
  state.motorAngleValue = 0;
  render();
  sendControlCommand("set_motor_angle", { motorCommand: "zero", angle: 0 });
}

function moveMotorRelative() {
  if (!ensureTrackingOff()) {
    return;
  }
  const angle = clampAngle(state.motorRelativeValue);
  render();
  sendControlCommand("set_motor_angle", { motorCommand: "relative", angle });
}

function clampAngle(angle) {
  return Math.max(ANGLE_MIN, Math.min(ANGLE_MAX, Math.round(Number(angle) || 0)));
}

function toggleReminderWeekday(value) {
  const weekdays = (state.reminderForm.weekdays || []).map(Number);
  if (weekdays.includes(value)) {
    state.reminderForm.weekdays = weekdays.filter(item => item !== value);
  } else {
    state.reminderForm.weekdays = weekdays.concat(value).sort((a, b) => a - b);
  }
  render();
}

function reminderPayloadFromForm() {
  const form = state.reminderForm;
  const label = String(form.label || "").trim();
  const time = String(form.time || "").trim();
  const repeat = form.repeat === "weekly" ? "weekly" : "once";
  const payload = {
    label,
    repeat,
    time,
    enabled: true
  };
  if (repeat === "weekly") {
    payload.date = "";
    payload.weekdays = (form.weekdays || []).map(Number).filter(value => value >= 1 && value <= 7);
  } else {
    payload.date = form.date || todayDateValue();
    payload.weekdays = [];
  }
  return payload;
}

function saveReminder() {
  const payload = reminderPayloadFromForm();
  if (!payload.label) {
    showToast("请输入提醒标签");
    return;
  }
  if (!payload.time) {
    showToast("请选择提醒时间");
    return;
  }
  if (payload.repeat === "weekly" && payload.weekdays.length === 0) {
    showToast("请选择重复星期");
    return;
  }

  if (state.reminderForm.id) {
    endReminderEditing();
    sendControlCommand("reminder_update", {
      reminderId: state.reminderForm.id,
      reminder: payload
    });
    showToast("已发送修改提醒");
  } else {
    endReminderEditing();
    sendControlCommand("reminder_add", { reminder: payload });
    showToast("已发送新增提醒");
  }
  resetReminderForm();
  render();
}

function editReminder(id) {
  const reminder = (state.reminders || []).find(item => String(item.id || "") === String(id || ""));
  if (!reminder) {
    showToast("未找到提醒");
    return;
  }
  state.reminderForm = {
    id: reminder.id || "",
    label: reminder.label || "",
    repeat: reminder.repeat === "weekly" ? "weekly" : "once",
    date: reminder.date || todayDateValue(),
    time: reminder.time || nextHourTimeValue(),
    weekdays: Array.isArray(reminder.weekdays) ? reminder.weekdays.map(Number) : []
  };
  beginReminderEditing("reminder");
  render();
}

function deleteReminder(id) {
  const reminder = (state.reminders || []).find(item => String(item.id || "") === String(id || ""));
  sendControlCommand("reminder_delete", {
    reminderId: id,
    targetLabel: reminder ? reminder.label || "" : ""
  });
  if (state.reminderForm.id === id) {
    endReminderEditing();
    resetReminderForm();
  }
  showToast("已发送删除提醒");
  render();
}

async function toggleBoardMicListen() {
  if (state.boardMicListening) {
    stopBoardMicListen();
    return;
  }
  state.boardMicListening = true;
  state.boardMicNative = false;
  state.boardMicCursor = "";
  state.boardMicQueue = [];

  if (hasAndroidVoice()) {
    try {
      const started = window.AndroidVoice.startListen(getVoiceHost(), getVoiceStreamPort());
      if (started) {
        state.boardMicNative = true;
        render();
        return;
      }
    } catch (error) {
      state.boardMicNative = false;
    }
  }

  render();
  fetchBoardMicChunk();
}

async function fetchBoardMicChunk() {
  if (!state.boardMicListening) {
    return;
  }

  state.boardMicAbort = new AbortController();
  try {
    const query = [`t=${Date.now()}`];
    if (state.boardMicCursor) {
      query.push(`cursor=${encodeURIComponent(state.boardMicCursor)}`);
    }
    const response = await fetch(`${getVoiceBaseUrl()}/mic-chunk?${query.join("&")}`, {
      cache: "no-store",
      signal: state.boardMicAbort.signal
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const blob = await response.blob();
    if (!state.boardMicListening) {
      return;
    }
    const nextCursor = response.headers.get("X-Next-Cursor");
    if (nextCursor !== null) {
      state.boardMicCursor = nextCursor;
    }
    if (state.boardMicQueue.length >= 8) {
      state.boardMicQueue.shift();
    }
    state.boardMicQueue.push(blob);
    playNextBoardMicChunk();
  } catch (error) {
    if (state.boardMicListening && error.name !== "AbortError") {
      showVoiceError("连接开发板麦克风失败");
      await new Promise(resolve => setTimeout(resolve, 800));
    }
  } finally {
    state.boardMicAbort = null;
    if (state.boardMicListening) {
      setTimeout(fetchBoardMicChunk, 50);
    }
  }
}

function playNextBoardMicChunk() {
  if (state.boardMicPlaying || !state.boardMicListening || state.boardMicQueue.length === 0) {
    return;
  }

  const blob = state.boardMicQueue.shift();
  const objectUrl = URL.createObjectURL(blob);
  const audio = new Audio(objectUrl);
  state.boardMicAudio = audio;
  state.boardMicPlaying = true;
  let finished = false;
  const finish = () => {
    if (finished) {
      return;
    }
    finished = true;
    URL.revokeObjectURL(objectUrl);
    if (state.boardMicAudio === audio) {
      state.boardMicAudio = null;
    }
    state.boardMicPlaying = false;
    playNextBoardMicChunk();
  };
  audio.onended = finish;
  audio.onerror = () => {
    finish();
    if (state.boardMicListening) {
      showVoiceError("播放开发板声音失败");
    }
  };
  audio.play().catch(() => {
    finish();
    if (state.boardMicListening) {
      showVoiceError("播放开发板声音失败");
    }
  });
}

function stopBoardMicListen() {
  if (state.boardMicNative && hasAndroidVoice()) {
    try {
      window.AndroidVoice.stopListen();
    } catch (error) {
    }
  }
  state.boardMicListening = false;
  state.boardMicNative = false;
  if (state.boardMicAbort) {
    state.boardMicAbort.abort();
    state.boardMicAbort = null;
  }
  if (state.boardMicAudio) {
    state.boardMicAudio.pause();
    state.boardMicAudio = null;
  }
  state.boardMicQueue = [];
  state.boardMicPlaying = false;
  state.boardMicCursor = "";
  render();
}

async function togglePhoneMicSend() {
  if (state.phoneMicSending) {
    stopPhoneMicSend();
    return;
  }

  if (hasAndroidVoice()) {
    try {
      const started = window.AndroidVoice.startTalk(getVoiceHost(), getVoiceStreamPort());
      if (started) {
        state.phoneMicSending = true;
        state.phoneMicNative = true;
        render();
        return;
      }
    } catch (error) {
      state.phoneMicNative = false;
    }
  }

  startPhoneMicSendFallback();
}

async function startPhoneMicSendFallback() {
  try {
    state.phoneMediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.phoneMicSending = true;
    state.phoneMicNative = false;
    render();
    startPhoneRecordChunk();
  } catch (error) {
    showVoiceError("无法打开手机麦克风");
  }
}

function getRecorderMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus"
  ];
  return candidates.find(type => window.MediaRecorder && MediaRecorder.isTypeSupported(type)) || "";
}

function startPhoneRecordChunk() {
  if (!state.phoneMicSending || !state.phoneMediaStream) {
    return;
  }

  const chunks = [];
  const mimeType = getRecorderMimeType();
  const options = mimeType ? { mimeType } : undefined;
  state.phoneMediaRecorder = new MediaRecorder(state.phoneMediaStream, options);
  state.phoneMediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  };
  state.phoneMediaRecorder.onstop = () => {
    if (chunks.length > 0 && state.phoneMicSending) {
      const blob = new Blob(chunks, { type: state.phoneMediaRecorder.mimeType || "audio/webm" });
      uploadPhoneAudio(blob);
    }
    state.phoneMediaRecorder = null;
    if (state.phoneMicSending) {
      state.phoneRecordTimer = setTimeout(startPhoneRecordChunk, 100);
    }
  };
  state.phoneMediaRecorder.start();
  state.phoneRecordTimer = setTimeout(() => {
    if (state.phoneMediaRecorder && state.phoneMediaRecorder.state !== "inactive") {
      state.phoneMediaRecorder.stop();
    }
  }, 1800);
}

function uploadPhoneAudio(blob) {
  const controller = new AbortController();
  state.phoneUploadControllers.push(controller);
  fetch(`${getVoiceBaseUrl()}/speaker-chunk`, {
    method: "POST",
    headers: { "Content-Type": blob.type || "audio/webm" },
    body: blob,
    signal: controller.signal
  })
    .catch((error) => {
      if (state.phoneMicSending && error.name !== "AbortError") {
        showVoiceError("发送声音到开发板失败");
      }
    })
    .finally(() => {
      state.phoneUploadControllers = state.phoneUploadControllers.filter(item => item !== controller);
    });
}

function stopPhoneMicSend() {
  const wasNative = state.phoneMicNative;
  if (state.phoneMicNative && hasAndroidVoice()) {
    try {
      window.AndroidVoice.stopTalk();
    } catch (error) {
    }
  }
  state.phoneMicSending = false;
  state.phoneMicNative = false;
  clearTimeout(state.phoneRecordTimer);
  state.phoneRecordTimer = null;

  if (state.phoneMediaRecorder && state.phoneMediaRecorder.state !== "inactive") {
    state.phoneMediaRecorder.stop();
  }
  state.phoneMediaRecorder = null;

  if (state.phoneMediaStream) {
    state.phoneMediaStream.getTracks().forEach(track => track.stop());
    state.phoneMediaStream = null;
  }

  state.phoneUploadControllers.forEach(controller => controller.abort());
  state.phoneUploadControllers = [];
  if (!wasNative) {
    fetch(`${getVoiceBaseUrl()}/speaker-stop`, { method: "POST" }).catch(() => {});
  }
  render();
}

function stopVoiceStreams() {
  if (state.boardMicListening) {
    stopBoardMicListen();
  }
  if (state.phoneMicSending) {
    stopPhoneMicSend();
  }
}

function showVoiceError(message) {
  const now = Date.now();
  if (now - state.lastVoiceErrorAt < 3000) {
    return;
  }
  state.lastVoiceErrorAt = now;
  showToast(message);
}

window.onAndroidVoiceState = function(payload) {
  if (!payload || !payload.kind) {
    return;
  }

  if (payload.kind === "listen") {
    const wasNative = state.boardMicNative;
    state.boardMicListening = !!payload.active;
    state.boardMicNative = !!payload.active;
    if (payload.error) {
      showVoiceError(`${payload.error}，已切换分片监听`);
      if (wasNative) {
        state.boardMicListening = true;
        state.boardMicNative = false;
        state.boardMicCursor = "";
        state.boardMicQueue = [];
        render();
        fetchBoardMicChunk();
        return;
      }
    }
    render();
    return;
  }

  if (payload.kind === "talk") {
    const wasNative = state.phoneMicNative;
    state.phoneMicSending = !!payload.active;
    state.phoneMicNative = !!payload.active;
    if (payload.error) {
      showVoiceError(`${payload.error}，已切换分片发送`);
      if (wasNative) {
        state.phoneMicSending = false;
        state.phoneMicNative = false;
        render();
        startPhoneMicSendFallback();
        return;
      }
    }
    render();
  }
};

async function startConnection() {
  if (!state.serverUrl || !state.serverPort) {
    showToast("请输入服务器地址和端口");
    return;
  }

  stopPolling();
  state.voiceServerUrl = state.voiceServerUrl || state.serverUrl;
  state.voicePort = state.voicePort || "8890";
  state.voiceStreamPort = state.voiceStreamPort || "8891";
  state.connecting = true;
  state.connected = false;
  saveConfig();
  render();

  const url = `http://${state.serverUrl}:${state.serverPort}/status`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    const connected = response.ok;
    state.connected = connected;
    state.connecting = false;
    if (connected) {
      await response.json().then(handleServerResponse).catch(() => {});
      startPolling();
      showToast("连接成功");
    } else {
      showToast("连接失败");
    }
  } catch (error) {
    state.connected = false;
    state.connecting = false;
    showToast("连接失败");
  }
  render();
}

function stopConnection() {
  stopPolling();
  state.connecting = false;
  state.connected = false;
  showToast("已断开连接");
  render();
}

function saveConfig() {
  localStorage.setItem("serverUrl", state.serverUrl);
  localStorage.setItem("serverPort", state.serverPort);
  localStorage.setItem("voiceServerUrl", state.voiceServerUrl || state.serverUrl);
  localStorage.setItem("voicePort", state.voicePort || "8890");
  localStorage.setItem("voiceStreamPort", state.voiceStreamPort || "8891");
}

function selectQuickRange(days) {
  const now = new Date();
  const end = new Date(now);
  const start = new Date(now);
  start.setDate(start.getDate() - days);

  state.historyFilter.startYear = String(start.getFullYear());
  state.historyFilter.startMonth = String(start.getMonth() + 1).padStart(2, "0");
  state.historyFilter.startDay = String(start.getDate()).padStart(2, "0");
  state.historyFilter.endYear = String(end.getFullYear());
  state.historyFilter.endMonth = String(end.getMonth() + 1).padStart(2, "0");
  state.historyFilter.endDay = String(end.getDate()).padStart(2, "0");
  render();
}

function resetFilter() {
  state.historyFilter = {
    startYear: "",
    startMonth: "",
    startDay: "",
    endYear: "",
    endMonth: "",
    endDay: ""
  };
  render();
}

function formatTime(timestamp) {
  const date = new Date(Number(timestamp || Date.now()));
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const seconds = String(date.getSeconds()).padStart(2, "0");
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

window.addEventListener("beforeunload", stopVoiceStreams);

window.handleAndroidBack = function() {
  return handleBackNavigation();
};

document.addEventListener("DOMContentLoaded", () => {
  loadConfig();
  loadAlarmHistory();
  resetReminderForm();
  render();
  startPolling();
});

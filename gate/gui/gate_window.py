"""
门禁系统窗口模块 — 沉浸式 Kiosk 风格
摄像头画面铺满，信息以浮层叠加显示，模拟真实门禁机体验
"""

import cv2
import sys
import os
import time
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMessageBox, QListWidget,
    QListWidgetItem, QFrame, QComboBox, QGraphicsOpacityEffect,
    QSizePolicy, QSpacerItem,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QLinearGradient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gate.access_control import AccessControl, DOOR_OPEN_DURATION
from common.user_manager import UserManager
from common.database import Database
from common.theme import (
    GLOBAL_STYLESHEET,
    btn_primary, btn_success, btn_danger, btn_outline,
    PRIMARY, PRIMARY_LIGHT, SUCCESS, DANGER, WARNING,
    BG_MAIN, BG_CARD, BORDER, DIVIDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_WHITE,
    KIOSK_BG, KIOSK_SURFACE, KIOSK_OVERLAY,
    KIOSK_PASS, KIOSK_PASS_GLOW,
    KIOSK_REJECT, KIOSK_REJECT_GLOW,
    KIOSK_STANDBY, KIOSK_TEXT, KIOSK_TEXT_DIM, KIOSK_ACCENT,
    FONT_FAMILY,
)


# ── 样式常量 ──────────────────────────────────────────────

KIOSK_BASE = f"""
    QMainWindow {{
        background-color: {KIOSK_BG};
    }}
    QWidget {{
        color: {KIOSK_TEXT};
        font-family: {FONT_FAMILY};
    }}
    QToolTip {{
        background-color: #1e1e2e;
        color: {KIOSK_TEXT};
        border: 1px solid #333;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
    }}
"""

CAMERA_STYLE = f"""
    QLabel {{
        border: none;
        background-color: #000000;
        border-radius: 0px;
    }}
"""

OVERLAY_PANEL = f"""
    QFrame {{
        background-color: {KIOSK_OVERLAY};
        border-radius: 12px;
    }}
"""

SETTINGS_PANEL = f"""
    QFrame {{
        background-color: rgba(18, 18, 26, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
    }}
    QLabel {{
        color: {KIOSK_TEXT};
        font-size: 13px;
    }}
    QComboBox {{
        background-color: rgba(255, 255, 255, 0.06);
        color: {KIOSK_TEXT};
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 13px;
        min-height: 20px;
    }}
    QComboBox:focus {{
        border-color: {KIOSK_ACCENT};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 28px;
    }}
    QComboBox::down-arrow {{
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {KIOSK_TEXT_DIM};
        margin-right: 6px;
    }}
    QComboBox QAbstractItemView {{
        background-color: #1a1a2e;
        color: {KIOSK_TEXT};
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        selection-background-color: rgba(79, 195, 247, 0.2);
        selection-color: {KIOSK_ACCENT};
        padding: 4px;
    }}
    QListWidget {{
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        font-size: 12px;
        color: {KIOSK_TEXT_DIM};
    }}
    QListWidget::item {{
        padding: 6px 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }}
    QListWidget::item:selected {{
        background-color: rgba(79, 195, 247, 0.12);
        color: {KIOSK_ACCENT};
    }}
"""


# ── 状态颜色映射 ──────────────────────────────────────────

STATUS_COLORS = {
    "pass": {
        "bg": KIOSK_PASS,
        "glow": KIOSK_PASS_GLOW,
        "text": "#ffffff",
    },
    "reject": {
        "bg": KIOSK_REJECT,
        "glow": KIOSK_REJECT_GLOW,
        "text": "#ffffff",
    },
    "standby": {
        "bg": "transparent",
        "glow": "transparent",
        "text": KIOSK_STANDBY,
    },
}


class GateWindow(QMainWindow):
    # 自定义信号：模型加载完成(model_path, success)
    model_loaded = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()

        # 初始化数据库和用户管理器
        self.db = Database()
        self.user_manager = UserManager(self.db)

        # 初始化门禁控制器
        self.access_control = AccessControl(self.user_manager, gate_name="Main Gate")

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头！")
                sys.exit(1)

        # 启动人脸识别后台线程
        self.access_control.start_processing()

        # 状态追踪
        self._current_status = "standby"
        self._last_pass_user = ""
        self._settings_visible = False

        # 设置定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # ~10 FPS

        # 时钟定时器
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)

        # 状态栏定时更新
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_door_status)
        self.status_timer.start(500)

        # 初始化UI
        self.init_ui()

        # 连接模型加载信号
        self.model_loaded.connect(self._on_model_loaded)

    # ── UI 构建 ────────────────────────────────────────────

    def init_ui(self):
        """初始化沉浸式 Kiosk 界面"""
        self.setWindowTitle("人脸识别门禁系统")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(KIOSK_BASE)

        # 中央部件（铺满窗口）
        central = QWidget()
        self.setCentralWidget(central)

        # 使用栈式布局：底层摄像头 + 上层浮层
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 底层：摄像头画面（铺满） ──
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(CAMERA_STYLE)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.camera_label, 1)

        # ── 顶部浮层：时钟 + 系统信息 ──
        self._build_top_overlay(central)

        # ── 底部浮层：状态栏 ──
        self._build_bottom_overlay(central)

        # ── 设置面板（默认隐藏） ──
        self._build_settings_panel(central)

        # 初始化时钟
        self._update_clock()

        # 填充模型列表
        self._populate_model_list()

        # 更新界面信息
        self.update_ui_info()

    def _build_top_overlay(self, parent):
        """顶部浮层：时钟 + 系统状态"""
        self.top_bar = QFrame(parent)
        self.top_bar.setStyleSheet(OVERLAY_PANEL)
        self.top_bar.setFixedHeight(56)
        # 位置由 resizeEvent 管理

        bar_layout = QHBoxLayout(self.top_bar)
        bar_layout.setContentsMargins(24, 0, 24, 0)

        # 左侧：时钟
        self.clock_label = QLabel("00:00")
        self.clock_label.setStyleSheet(
            f"font-size: 28px; font-weight: bold; color: #ffffff; "
            f"background: transparent; letter-spacing: 2px;"
        )
        bar_layout.addWidget(self.clock_label)

        # 日期
        self.date_label = QLabel("")
        self.date_label.setStyleSheet(
            f"font-size: 14px; color: {KIOSK_TEXT_DIM}; "
            f"background: transparent; margin-left: 16px;"
        )
        bar_layout.addWidget(self.date_label)

        bar_layout.addStretch()

        # 右侧：系统状态
        self.system_label = QLabel("")
        self.system_label.setStyleSheet(
            f"font-size: 13px; color: {KIOSK_TEXT_DIM}; background: transparent;"
        )
        bar_layout.addWidget(self.system_label)

        # 设置按钮
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(36, 36)
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255, 255, 255, 0.08);
                color: {KIOSK_TEXT_DIM};
                border: none;
                border-radius: 18px;
                font-size: 18px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 255, 255, 0.15);
                color: {KIOSK_TEXT};
            }}
        """)
        self.settings_btn.clicked.connect(self._toggle_settings)
        bar_layout.addWidget(self.settings_btn)

    def _build_bottom_overlay(self, parent):
        """底部浮层：识别状态 + 用户信息"""
        self.bottom_bar = QFrame(parent)
        self.bottom_bar.setStyleSheet(OVERLAY_PANEL)
        self.bottom_bar.setFixedHeight(100)
        # 位置由 resizeEvent 管理

        bar_layout = QHBoxLayout(self.bottom_bar)
        bar_layout.setContentsMargins(32, 0, 32, 0)
        bar_layout.setSpacing(24)

        # 左侧：状态指示灯 + 状态文字
        left_layout = QVBoxLayout()
        left_layout.setSpacing(4)

        # 状态指示圆点 + 文字
        status_row = QHBoxLayout()
        status_row.setSpacing(10)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(
            f"font-size: 14px; color: {KIOSK_STANDBY}; background: transparent;"
        )
        status_row.addWidget(self.status_dot)

        self.status_text = QLabel("等待识别")
        self.status_text.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {KIOSK_STANDBY}; "
            f"background: transparent;"
        )
        status_row.addWidget(self.status_text)
        status_row.addStretch()

        left_layout.addLayout(status_row)

        # 注册用户数 / 门禁名称
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            f"font-size: 12px; color: {KIOSK_TEXT_DIM}; background: transparent;"
        )
        left_layout.addWidget(self.info_label)

        bar_layout.addLayout(left_layout, 1)

        # 右侧：用户信息（识别到时显示）
        self.user_frame = QFrame()
        self.user_frame.setStyleSheet(
            f"background-color: rgba(255, 255, 255, 0.05); border-radius: 10px;"
        )
        self.user_frame.setFixedHeight(68)
        self.user_frame.setFixedWidth(280)

        user_layout = QHBoxLayout(self.user_frame)
        user_layout.setContentsMargins(16, 8, 16, 8)
        user_layout.setSpacing(12)

        # 用户头像占位
        self.user_avatar = QLabel("👤")
        self.user_avatar.setFixedSize(48, 48)
        self.user_avatar.setAlignment(Qt.AlignCenter)
        self.user_avatar.setStyleSheet(
            f"font-size: 24px; background-color: rgba(255, 255, 255, 0.08); "
            f"border-radius: 24px; color: {KIOSK_TEXT_DIM};"
        )
        user_layout.addWidget(self.user_avatar)

        # 用户名 + 置信度
        user_info_layout = QVBoxLayout()
        user_info_layout.setSpacing(2)

        self.user_name_label = QLabel("—")
        self.user_name_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {KIOSK_TEXT}; "
            f"background: transparent;"
        )
        user_info_layout.addWidget(self.user_name_label)

        self.confidence_label = QLabel("置信度 —")
        self.confidence_label.setStyleSheet(
            f"font-size: 12px; color: {KIOSK_TEXT_DIM}; background: transparent;"
        )
        user_info_layout.addWidget(self.confidence_label)

        user_layout.addLayout(user_info_layout, 1)

        bar_layout.addWidget(self.user_frame)

    def _build_settings_panel(self, parent):
        """右侧设置面板（默认隐藏，点击齿轮弹出）"""
        self.settings_panel = QFrame(parent)
        self.settings_panel.setStyleSheet(SETTINGS_PANEL)
        self.settings_panel.setFixedWidth(340)
        self.settings_panel.hide()
        # 位置由 resizeEvent 管理

        panel_layout = QVBoxLayout(self.settings_panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(16)

        # 标题栏
        title_row = QHBoxLayout()
        title = QLabel("系统设置")
        title.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: #ffffff;"
        )
        title_row.addWidget(title)
        title_row.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {KIOSK_TEXT_DIM};
                border: none;
                font-size: 16px;
                border-radius: 14px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 255, 255, 0.1);
                color: {KIOSK_TEXT};
            }}
        """)
        close_btn.clicked.connect(self._toggle_settings)
        title_row.addWidget(close_btn)
        panel_layout.addLayout(title_row)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background-color: rgba(255, 255, 255, 0.06); max-height: 1px;")
        panel_layout.addWidget(sep)

        # 模型选择
        model_title = QLabel("识别模型")
        model_title.setStyleSheet(f"font-size: 12px; color: {KIOSK_TEXT_DIM}; text-transform: uppercase; letter-spacing: 1px;")
        panel_layout.addWidget(model_title)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(38)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        panel_layout.addWidget(self.model_combo)

        # 模型信息
        self.model_label = QLabel("")
        self.model_label.setWordWrap(True)
        self.model_label.setStyleSheet(f"font-size: 12px; color: {KIOSK_TEXT_DIM};")
        panel_layout.addWidget(self.model_label)

        # 安全模式
        self.security_label = QLabel("")
        self.security_label.setWordWrap(True)
        self.security_label.setStyleSheet(f"font-size: 12px; color: {KIOSK_TEXT_DIM};")
        panel_layout.addWidget(self.security_label)

        # 人脸检测方式切换
        detect_title = QLabel("人脸检测引擎")
        detect_title.setStyleSheet(f"font-size: 12px; color: {KIOSK_TEXT_DIM}; text-transform: uppercase; letter-spacing: 1px;")
        panel_layout.addWidget(detect_title)

        self.detection_toggle_btn = QPushButton()
        self.detection_toggle_btn.setMinimumHeight(42)
        self.detection_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.detection_toggle_btn.clicked.connect(self._toggle_detection_method)
        panel_layout.addWidget(self.detection_toggle_btn)

        # 检测方式说明（必须在 _update_detection_toggle_style 之前创建）
        self.detection_hint = QLabel("")
        self.detection_hint.setWordWrap(True)
        self.detection_hint.setStyleSheet(f"font-size: 11px; color: {KIOSK_TEXT_DIM};")
        panel_layout.addWidget(self.detection_hint)

        self._update_detection_toggle_style()

        # 刷新按钮
        self.reload_btn = QPushButton("🔄 刷新人脸库")
        self.reload_btn.setStyleSheet(btn_primary(
            f"min-height: 38px; font-size: 13px; "
            f"background-color: {KIOSK_ACCENT}; color: #000;"
        ))
        self.reload_btn.clicked.connect(self.reload_faces)
        panel_layout.addWidget(self.reload_btn)

        # 分隔线
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"background-color: rgba(255, 255, 255, 0.06); max-height: 1px;")
        panel_layout.addWidget(sep2)

        # 通行日志标题
        log_title = QLabel("近期通行记录")
        log_title.setStyleSheet(f"font-size: 12px; color: {KIOSK_TEXT_DIM}; text-transform: uppercase; letter-spacing: 1px;")
        panel_layout.addWidget(log_title)

        self.log_list = QListWidget()
        self.log_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        panel_layout.addWidget(self.log_list, 1)

    # ── 事件处理 ────────────────────────────────────────────

    def resizeEvent(self, event):
        """窗口大小变化时重新定位浮层"""
        super().resizeEvent(event)
        w = self.width()
        h = self.height()

        # 顶部栏：铺满顶部，留 16px 边距
        margin = 16
        self.top_bar.setGeometry(margin, margin, w - margin * 2, 56)

        # 底部栏：铺满底部
        self.bottom_bar.setGeometry(margin, h - 100 - margin, w - margin * 2, 100)

        # 设置面板：右侧弹出
        panel_w = self.settings_panel.width()
        if self._settings_visible:
            self.settings_panel.setGeometry(w - panel_w - margin, 80, panel_w, h - 200)
        else:
            self.settings_panel.setGeometry(w, 80, panel_w, h - 200)

    def _toggle_settings(self):
        """切换设置面板显示/隐藏"""
        w = self.width()
        margin = 16
        panel_w = 340
        target_x = w - panel_w - margin if not self._settings_visible else w

        anim = QPropertyAnimation(self.settings_panel, b"geometry")
        anim.setDuration(250)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        current = self.settings_panel.geometry()
        anim.setStartValue(current)
        anim.setEndValue(QRect(target_x, 80, panel_w, self.height() - 200))
        anim.start()
        self._anim = anim  # prevent GC

        if not self._settings_visible:
            self.settings_panel.show()
            self.settings_panel.raise_()
        self._settings_visible = not self._settings_visible

    # ── 帧更新 ────────────────────────────────────────────

    def update_frame(self):
        """更新摄像头画面 + 识别状态"""
        ret, frame = self.cap.read()
        if not ret:
            return

        # 处理帧
        frame, status, user_name, confidence = self.access_control.process_frame(frame)

        # 更新状态 UI
        self._apply_status(status, user_name, confidence)

        # 转换为 Qt 图像并显示
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled)

    def _apply_status(self, status, user_name, confidence):
        """根据识别结果更新底部状态栏"""
        colors = STATUS_COLORS

        if status == "Pass":
            c = colors["pass"]
            self._current_status = "pass"
            self._last_pass_user = user_name
            self.status_dot.setStyleSheet(f"font-size: 14px; color: {c['bg']}; background: transparent;")
            self.status_text.setText(f"欢迎，{user_name}")
            self.status_text.setStyleSheet(
                f"font-size: 22px; font-weight: bold; color: {c['bg']}; background: transparent;"
            )
            self.user_name_label.setText(user_name)
            self.user_name_label.setStyleSheet(
                f"font-size: 16px; font-weight: bold; color: {c['bg']}; background: transparent;"
            )
            self.confidence_label.setText(f"置信度 {confidence:.1%}")
            self.confidence_label.setStyleSheet(
                f"font-size: 12px; color: {KIOSK_TEXT_DIM}; background: transparent;"
            )
            self.bottom_bar.setStyleSheet(
                f"QFrame {{ background-color: rgba(0, 230, 118, 0.08); "
                f"border: 1px solid rgba(0, 230, 118, 0.2); border-radius: 12px; }}"
            )

        elif status == "Reject":
            c = colors["reject"]
            self._current_status = "reject"
            self.status_dot.setStyleSheet(f"font-size: 14px; color: {c['bg']}; background: transparent;")
            self.status_text.setText("未授权人员")
            self.status_text.setStyleSheet(
                f"font-size: 22px; font-weight: bold; color: {c['bg']}; background: transparent;"
            )
            self.user_name_label.setText("未知")
            self.user_name_label.setStyleSheet(
                f"font-size: 16px; font-weight: bold; color: {c['bg']}; background: transparent;"
            )
            self.confidence_label.setText("未注册用户")
            self.confidence_label.setStyleSheet(
                f"font-size: 12px; color: {KIOSK_TEXT_DIM}; background: transparent;"
            )
            self.bottom_bar.setStyleSheet(
                f"QFrame {{ background-color: rgba(255, 23, 68, 0.08); "
                f"border: 1px solid rgba(255, 23, 68, 0.2); border-radius: 12px; }}"
            )

        else:
            c = colors["standby"]
            if self._current_status != "standby":
                self._current_status = "standby"
                self.user_name_label.setText("—")
                self.user_name_label.setStyleSheet(
                    f"font-size: 16px; font-weight: bold; color: {KIOSK_TEXT}; background: transparent;"
                )
                self.confidence_label.setText("置信度 —")
                self.confidence_label.setStyleSheet(
                    f"font-size: 12px; color: {KIOSK_TEXT_DIM}; background: transparent;"
                )
            self.status_dot.setStyleSheet(f"font-size: 14px; color: {c}; background: transparent;")
            self.status_text.setText("等待识别")
            self.status_text.setStyleSheet(
                f"font-size: 20px; font-weight: bold; color: {c}; background: transparent;"
            )
            self.bottom_bar.setStyleSheet(OVERLAY_PANEL)

    # ── 时钟 ────────────────────────────────────────────────

    def _update_clock(self):
        """更新时钟显示"""
        now = datetime.now()
        self.clock_label.setText(now.strftime("%H:%M"))
        self.date_label.setText(now.strftime("%Y年%m月%d日 %A").replace(
            "Monday", "周一").replace("Tuesday", "周二").replace(
            "Wednesday", "周三").replace("Thursday", "周四").replace(
            "Friday", "周五").replace("Saturday", "周六").replace(
            "Sunday", "周日"))

    # ── 门禁状态 ────────────────────────────────────────────

    def _update_door_status(self):
        """更新门禁状态到系统信息区"""
        ac = self.access_control
        now = time.time()

        if ac.door_open:
            remaining = max(0, DOOR_OPEN_DURATION - (now - ac.door_open_time))
            self.system_label.setText(f"🔓 门已开启  {ac.current_user}  {remaining:.0f}s")
            self.system_label.setStyleSheet(
                f"font-size: 13px; color: {KIOSK_PASS}; background: transparent;"
            )
        elif now < ac.door_cooldown_until:
            remaining = ac.door_cooldown_until - now
            self.system_label.setText(f"⏳ 冷却中 {remaining:.0f}s")
            self.system_label.setStyleSheet(
                f"font-size: 13px; color: {KIOSK_TEXT_DIM}; background: transparent;"
            )
        else:
            method = ac.get_detection_method()
            known = ac.get_known_faces_count()
            self.system_label.setText(f"● 就绪  人脸库: {known}")
            self.system_label.setStyleSheet(
                f"font-size: 13px; color: {KIOSK_TEXT_DIM}; background: transparent;"
            )

    # ── UI 信息 ────────────────────────────────────────────

    def update_ui_info(self):
        """更新界面信息"""
        active_count = self.access_control.get_active_users_count()
        known_faces = self.access_control.get_known_faces_count()
        self.info_label.setText(f"注册用户: {active_count}  |  人脸编码: {known_faces}")

        # 模型信息（使用模块化后端获取识别器名称）
        rec_method = self.access_control.get_recognition_method()
        if "InsightFace" in rec_method:
            self.model_label.setText("InsightFace ArcFace-R100\n512维 embedding, 高精度预训练")
        elif "MobileFaceNet" in rec_method:
            threshold = getattr(self.access_control.recognizer, '_threshold', 0.55)
            model_dir = getattr(self.access_control.recognizer, 'model_dir', '')
            model_name = os.path.basename(model_dir) if model_dir else "MobileFaceNet"
            self.model_label.setText(f"{model_name}\n阈值: {threshold:.2f}")
        else:
            self.model_label.setText("face_recognition (内置)")

        # 检测方式按钮状态同步
        self._update_detection_toggle_style()

        # 安全模式
        if self.access_control.is_strict_mode():
            self.security_label.setText("🔒 严格模式：仅注册用户可通过")
            self.security_label.setStyleSheet(f"color: {KIOSK_PASS}; font-size: 12px;")
        else:
            self.security_label.setText("⚠️ 普通模式：可自动注册")
            self.security_label.setStyleSheet(f"color: {WARNING}; font-size: 12px;")

        # 更新日志
        self.log_list.clear()
        logs = self.db.get_access_log(limit=10)
        for log in logs:
            is_pass = log["result"] == "pass"
            icon = "✓" if is_pass else "✗"
            ts = log['timestamp']
            # 只取时间部分
            if ' ' in ts:
                ts = ts.split(' ')[1]
            item_text = f"  {icon}  {ts}   {log['user_name']}"
            item = QListWidgetItem(item_text)
            item.setForeground(QColor(KIOSK_PASS if is_pass else KIOSK_REJECT))
            self.log_list.addItem(item)

    def _populate_model_list(self):
        """填充模型选择下拉框（支持 mobilenet / insightface / face_recognition）"""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        # 获取当前识别器类型
        current_method = self.access_control.get_recognition_method()
        current_model_dir = ""
        if hasattr(self.access_control.recognizer, 'model_dir'):
            current_model_dir = self.access_control.recognizer.model_dir or ""

        # 1. InsightFace 选项
        try:
            from common.recognizers import InsightFaceRecognizer
            self.model_combo.addItem("InsightFace ArcFace (512维, 高精度)", "insightface")
        except Exception:
            pass

        # 2. MobileFaceNet 训练版本
        models = self.access_control.get_available_models()
        for m in models:
            if m.get('is_builtin'):
                continue  # 后面单独加 face_recognition
            self.model_combo.addItem(f"MobileFaceNet: {m['name']}", m['path'])

        # 3. face_recognition 兜底
        self.model_combo.addItem("face_recognition (内置)", "face_recognition")

        # 定位当前选中项
        selected = False
        for i in range(self.model_combo.count()):
            data = self.model_combo.itemData(i)
            if data == "insightface" and "InsightFace" in current_method:
                self.model_combo.setCurrentIndex(i)
                selected = True
                break
            elif data == "face_recognition" and "FaceRecognition" in current_method:
                self.model_combo.setCurrentIndex(i)
                selected = True
                break
            elif data and data == current_model_dir:
                self.model_combo.setCurrentIndex(i)
                selected = True
                break

        if not selected and self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)

        self.model_combo.blockSignals(False)

    def on_model_changed(self, index):
        """切换识别模型"""
        model_data = self.model_combo.currentData()
        if not model_data:
            return

        self.model_combo.blockSignals(True)
        self.model_combo.setEnabled(False)

        if model_data == "insightface":
            import threading
            threading.Thread(target=self._switch_recognizer_thread,
                           args=("insightface",), daemon=True).start()
        elif model_data == "face_recognition":
            import threading
            threading.Thread(target=self._switch_recognizer_thread,
                           args=("face_recognition",), daemon=True).start()
        else:
            # MobileFaceNet 模型目录
            import threading
            threading.Thread(target=self._load_model_thread,
                           args=(model_data,), daemon=True).start()

    def _switch_recognizer_thread(self, backend):
        """后台线程切换识别器后端"""
        try:
            from common.recognizers import RecognizerFactory
            new_recognizer = RecognizerFactory.create(backend)
            self.access_control.recognizer = new_recognizer
            self.access_control._load_database_to_recognizer()
            success = True
        except Exception as e:
            print(f"Failed to switch to {backend}: {e}")
            success = False
        self.model_loaded.emit(backend, success)

    def _load_model_thread(self, model_path):
        """后台线程加载 MobileFaceNet 模型"""
        try:
            success = self.access_control.load_model_from_dir(model_path)
        except Exception:
            success = False
        self.model_loaded.emit(model_path, success)

    def _on_model_loaded(self, model_path, success):
        """模型加载完成回调"""
        self.model_combo.blockSignals(False)
        self.model_combo.setEnabled(True)
        if success:
            self.update_ui_info()
            name = os.path.basename(model_path) if os.path.sep in model_path else model_path
            QMessageBox.information(self, "成功", f"已切换到: {name}")
        else:
            QMessageBox.warning(self, "错误", f"加载模型失败: {model_path}")
            # 恢复下拉框到之前的选项
            self._populate_model_list()

    def reload_faces(self):
        """重新加载人脸库"""
        self.access_control.reload_known_faces()
        self.update_ui_info()
        count = self.access_control.get_known_faces_count()
        QMessageBox.information(self, "成功", f"已重新加载 {count} 个人脸编码")

    def _toggle_detection_method(self):
        """切换人脸检测方式：YOLO ↔ face_recognition"""
        det_method = self.access_control.get_detection_method()
        current_is_yolo = "YOLO" in det_method
        new_state = not current_is_yolo

        try:
            self.access_control.set_yolo_enabled(new_state)
        except Exception as e:
            QMessageBox.warning(self, "切换失败", f"切换检测方式失败：{e}")
            return

        self._update_detection_toggle_style()
        method = self.access_control.get_detection_method()
        print(f"Detection method switched to: {method}")

    def _update_detection_toggle_style(self):
        """更新检测方式切换按钮的样式和文字"""
        det_method = self.access_control.get_detection_method()
        is_yolo = "YOLO" in det_method

        if is_yolo:
            self.detection_toggle_btn.setText("YOLO  ⚡ 已启用")
            self.detection_toggle_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(0, 230, 118, 0.15);
                    color: {KIOSK_PASS};
                    border: 1px solid rgba(0, 230, 118, 0.3);
                    border-radius: 10px;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 8px 16px;
                }}
                QPushButton:hover {{
                    background-color: rgba(0, 230, 118, 0.25);
                    border-color: rgba(0, 230, 118, 0.5);
                }}
            """)
            self.detection_hint.setText("YOLO 检测速度更快，适合实时场景。点击切换为 face_recognition。")
        else:
            self.detection_toggle_btn.setText("face_recognition  🔧 已启用")
            self.detection_toggle_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(79, 195, 247, 0.12);
                    color: {KIOSK_ACCENT};
                    border: 1px solid rgba(79, 195, 247, 0.25);
                    border-radius: 10px;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 8px 16px;
                }}
                QPushButton:hover {{
                    background-color: rgba(79, 195, 247, 0.2);
                    border-color: rgba(79, 195, 247, 0.4);
                }}
            """)
            self.detection_hint.setText("face_recognition 基于 dlib，精度高但速度较慢。点击切换为 YOLO。")

    # ── 关闭 ────────────────────────────────────────────────

    def closeEvent(self, event):
        """关闭窗口事件"""
        self.timer.stop()
        self.clock_timer.stop()
        self.status_timer.stop()
        self.access_control.stop_processing()
        self.cap.release()
        event.accept()

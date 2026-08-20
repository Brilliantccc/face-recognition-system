"""
门禁系统窗口模块
显示摄像头预览和门禁状态
"""

import cv2
import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QMessageBox, QListWidget,
                             QListWidgetItem, QGroupBox, QComboBox, QFrame,
                             QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gate.access_control import AccessControl
from common.user_manager import UserManager
from common.database import Database
from common.theme import (
    GLOBAL_STYLESHEET, btn_primary, btn_success, btn_danger, btn_outline,
    PRIMARY, PRIMARY_LIGHT, SUCCESS, SUCCESS_LIGHT, DANGER, DANGER_LIGHT,
    WARNING, WARNING_LIGHT, BG_MAIN, BG_CARD, BG_SIDEBAR,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_WHITE, BORDER, DIVIDER,
    GATE_PASS_BG, GATE_PASS_BORDER, GATE_PASS_TEXT,
    GATE_REJECT_BG, GATE_REJECT_BORDER, GATE_REJECT_TEXT,
    GATE_STANDBY_BG, GATE_STANDBY_BORDER, GATE_STANDBY_TEXT
)


class GateWindow(QMainWindow):
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

        # 设置定时器 (降低帧率以减少CPU使用)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(150)  # 约7 FPS

        # 初始化UI
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("🚪 人脸识别门禁系统")
        self.setGeometry(100, 100, 1300, 850)
        self.setMinimumSize(1000, 700)

        # 应用全局样式
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # ── 左侧：摄像头预览 ──
        left_panel = QWidget()
        left_panel.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 12px; border: 1px solid {BORDER};"
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        # 摄像头标题
        camera_title = QLabel("📹 实时监控")
        camera_title.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {TEXT_PRIMARY}; padding: 4px 0;"
        )
        left_layout.addWidget(camera_title)

        # 摄像头预览标签
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet(
            f"border: 2px solid #333; background-color: #1a1a2e; border-radius: 10px;"
        )
        left_layout.addWidget(self.camera_label, 1)

        # 状态显示（大卡片）
        self.status_card = QFrame()
        self.status_card.setMinimumHeight(70)
        self.status_card.setStyleSheet(
            f"background-color: {GATE_STANDBY_BG}; border: 2px solid {GATE_STANDBY_BORDER}; "
            f"border-radius: 12px;"
        )
        status_card_layout = QVBoxLayout(self.status_card)
        status_card_layout.setContentsMargins(16, 8, 16, 8)

        self.status_label = QLabel("📹 请正对摄像头")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {GATE_STANDBY_TEXT}; "
            f"background: transparent; border: none;"
        )
        status_card_layout.addWidget(self.status_label)

        left_layout.addWidget(self.status_card)

        main_layout.addWidget(left_panel, 2)

        # ── 右侧：控制面板 ──
        right_panel = QWidget()
        right_panel.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 12px; border: 1px solid {BORDER};"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        # ── 门禁状态信息 ──
        status_group = QGroupBox("📊 门禁状态")
        status_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px; font-weight: bold;
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 10px;
                margin-top: 14px;
                padding: 20px 14px 14px 14px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                color: {TEXT_PRIMARY};
            }}
        """)
        status_layout = QVBoxLayout()
        status_layout.setSpacing(10)

        # 用户信息卡片
        user_card = QFrame()
        user_card.setStyleSheet(
            f"background-color: {BG_MAIN}; border-radius: 8px; padding: 8px;"
        )
        user_card_layout = QVBoxLayout(user_card)
        user_card_layout.setContentsMargins(12, 8, 12, 8)
        user_card_layout.setSpacing(4)

        self.user_label = QLabel("👤 用户: 未知")
        self.user_label.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {TEXT_PRIMARY}; background: transparent;")
        user_card_layout.addWidget(self.user_label)

        self.confidence_label = QLabel("🎯 置信度: 0%")
        self.confidence_label.setStyleSheet(f"font-size: 13px; color: {TEXT_SECONDARY}; background: transparent;")
        user_card_layout.addWidget(self.confidence_label)

        status_layout.addWidget(user_card)

        # 系统信息
        info_card = QFrame()
        info_card.setStyleSheet(
            f"background-color: {BG_MAIN}; border-radius: 8px; padding: 8px;"
        )
        info_card_layout = QVBoxLayout(info_card)
        info_card_layout.setContentsMargins(12, 8, 12, 8)
        info_card_layout.setSpacing(4)

        self.faces_count_label = QLabel("📋 注册用户: 0")
        self.faces_count_label.setStyleSheet(f"font-size: 13px; color: {TEXT_SECONDARY}; background: transparent;")
        info_card_layout.addWidget(self.faces_count_label)

        self.model_label = QLabel("🤖 模型: 加载中...")
        self.model_label.setStyleSheet(f"font-size: 12px; color: {TEXT_MUTED}; background: transparent;")
        info_card_layout.addWidget(self.model_label)

        self.security_label = QLabel("🔒 安全模式: 加载中...")
        self.security_label.setStyleSheet(f"font-size: 12px; color: {TEXT_MUTED}; background: transparent;")
        info_card_layout.addWidget(self.security_label)

        status_layout.addWidget(info_card)
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        # ── 操作面板 ──
        buttons_group = QGroupBox("⚙️ 操作面板")
        buttons_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px; font-weight: bold;
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 10px;
                margin-top: 14px;
                padding: 20px 14px 14px 14px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                color: {TEXT_PRIMARY};
            }}
        """)
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)

        # 模型选择
        model_label = QLabel("识别模型:")
        model_label.setStyleSheet(f"font-size: 13px; color: {TEXT_SECONDARY};")
        buttons_layout.addWidget(model_label)

        model_select_layout = QHBoxLayout()
        model_select_layout.setSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(38)
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_select_layout.addWidget(self.model_combo, 1)

        buttons_layout.addLayout(model_select_layout)

        # 刷新按钮
        self.reload_btn = QPushButton("🔄 刷新人脸库")
        self.reload_btn.setStyleSheet(btn_primary("min-height: 40px; font-size: 14px;"))
        self.reload_btn.clicked.connect(self.reload_faces)
        buttons_layout.addWidget(self.reload_btn)

        buttons_group.setLayout(buttons_layout)
        right_layout.addWidget(buttons_group)

        # ── 通行日志 ──
        log_group = QGroupBox("📋 近期通行记录")
        log_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px; font-weight: bold;
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 10px;
                margin-top: 14px;
                padding: 20px 14px 14px 14px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                color: {TEXT_PRIMARY};
            }}
        """)
        log_layout = QVBoxLayout()

        self.log_list = QListWidget()
        self.log_list.setMaximumHeight(220)
        self.log_list.setStyleSheet(f"""
            QListWidget {{
                border: 1px solid {BORDER};
                border-radius: 8px;
                background-color: {BG_MAIN};
                font-size: 12px;
            }}
            QListWidget::item {{
                padding: 8px 12px;
                border-bottom: 1px solid {DIVIDER};
            }}
            QListWidget::item:selected {{
                background-color: {PRIMARY_LIGHT};
            }}
        """)
        log_layout.addWidget(self.log_list)

        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)

        right_layout.addStretch()
        main_layout.addWidget(right_panel, 1)

        # 状态栏
        self.statusBar().showMessage("✅ 系统就绪")
        self.statusBar().setStyleSheet(
            f"background-color: {BG_CARD}; color: {TEXT_SECONDARY}; "
            f"border-top: 1px solid {BORDER}; padding: 6px 16px; font-size: 12px;"
        )

        # 填充模型列表
        self._populate_model_list()

        # 更新界面信息
        self.update_ui_info()

    def update_frame(self):
        """更新摄像头画面"""
        ret, frame = self.cap.read()
        if ret:
            # 处理帧
            frame, status, user_name, confidence = self.access_control.process_frame(frame)

            # 更新状态显示
            if status == "Pass":
                self.status_label.setText(f"✅ 欢迎: {user_name}")
                self.status_label.setStyleSheet(
                    f"font-size: 20px; font-weight: bold; color: {GATE_PASS_TEXT}; "
                    f"background: transparent; border: none;"
                )
                self.status_card.setStyleSheet(
                    f"background-color: {GATE_PASS_BG}; border: 2px solid {GATE_PASS_BORDER}; "
                    f"border-radius: 12px;"
                )
                self.user_label.setText(f"👤 用户: {user_name}")
                self.user_label.setStyleSheet(
                    f"font-size: 15px; font-weight: bold; color: {GATE_PASS_TEXT}; background: transparent;"
                )
                self.confidence_label.setText(f"🎯 置信度: {confidence:.2%}")
                self.confidence_label.setStyleSheet(
                    f"font-size: 13px; color: {GATE_PASS_TEXT}; background: transparent;"
                )
            elif status == "Reject":
                self.status_label.setText("❌ 未授权 — 未注册或未激活")
                self.status_label.setStyleSheet(
                    f"font-size: 20px; font-weight: bold; color: {GATE_REJECT_TEXT}; "
                    f"background: transparent; border: none;"
                )
                self.status_card.setStyleSheet(
                    f"background-color: {GATE_REJECT_BG}; border: 2px solid {GATE_REJECT_BORDER}; "
                    f"border-radius: 12px;"
                )
                self.user_label.setText("👤 用户: 未知")
                self.user_label.setStyleSheet(
                    f"font-size: 15px; font-weight: bold; color: {GATE_REJECT_TEXT}; background: transparent;"
                )
                self.confidence_label.setText("🎯 置信度: 0%")
                self.confidence_label.setStyleSheet(
                    f"font-size: 13px; color: {GATE_REJECT_TEXT}; background: transparent;"
                )
            else:
                self.status_label.setText("📹 请正对摄像头")
                self.status_label.setStyleSheet(
                    f"font-size: 20px; font-weight: bold; color: {GATE_STANDBY_TEXT}; "
                    f"background: transparent; border: none;"
                )
                self.status_card.setStyleSheet(
                    f"background-color: {GATE_STANDBY_BG}; border: 2px solid {GATE_STANDBY_BORDER}; "
                    f"border-radius: 12px;"
                )
                self.user_label.setText("👤 用户: 未知")
                self.user_label.setStyleSheet(
                    f"font-size: 15px; font-weight: bold; color: {TEXT_PRIMARY}; background: transparent;"
                )
                self.confidence_label.setText("🎯 置信度: 0%")
                self.confidence_label.setStyleSheet(
                    f"font-size: 13px; color: {TEXT_SECONDARY}; background: transparent;"
                )

            # 转换为Qt图像
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 显示图像
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)

    def update_ui_info(self):
        """更新界面信息"""
        # 显示注册用户数
        active_count = self.access_control.get_active_users_count()
        known_faces = self.access_control.get_known_faces_count()
        self.faces_count_label.setText(
            f"📋 注册用户: {active_count} (数据库) | {known_faces} (人脸编码)"
        )

        # 显示识别模型信息
        detect_method = self.access_control.get_detection_method()
        recog_method = self.access_control.get_recognition_method()
        self.model_label.setText(f"🤖 检测: {detect_method} | 识别: {recog_method}")

        # 显示安全模式
        if self.access_control.is_strict_mode():
            self.security_label.setText("🔒 安全模式: 严格 — 仅数据库注册用户可通过")
            self.security_label.setStyleSheet(f"color: {SUCCESS}; font-weight: bold; background: transparent;")
        else:
            self.security_label.setText("⚠️ 安全模式: 普通 — 模型类别可自动注册")
            self.security_label.setStyleSheet(f"color: {WARNING}; font-weight: bold; background: transparent;")

        # 更新门禁日志
        self.log_list.clear()
        logs = self.db.get_access_log(limit=10)
        for log in logs:
            is_pass = log["result"] == "pass"
            result_text = "✅ 通过" if is_pass else "❌ 拒绝"
            icon = "🟢" if is_pass else "🔴"

            item_text = f"{icon} {log['timestamp']}  |  {log['user_name']}  |  {result_text}"
            item = QListWidgetItem(item_text)

            if is_pass:
                item.setForeground(QColor(SUCCESS))
            else:
                item.setForeground(QColor(DANGER))

            self.log_list.addItem(item)

    def _populate_model_list(self):
        """填充模型选择下拉框"""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        models = self.access_control.get_available_models()
        current_dir = getattr(self.access_control, 'trained_model_dir', '')

        for i, m in enumerate(models):
            display = f"{m['name']} ({m['num_classes']} 类)"
            self.model_combo.addItem(display, m['path'])
            if m['path'] == current_dir:
                self.model_combo.setCurrentIndex(i)

        if self.model_combo.count() == 0:
            self.model_combo.addItem("无可用模型", "")
            self.model_combo.setEnabled(False)

        self.model_combo.blockSignals(False)

    def on_model_changed(self, index):
        """切换识别模型"""
        model_path = self.model_combo.currentData()
        if not model_path:
            return

        self.statusBar().showMessage(f"⏳ 正在加载模型: {os.path.basename(model_path)}...")

        success = self.access_control.load_model_from_dir(model_path)
        if success:
            self.update_ui_info()
            self.statusBar().showMessage(f"✅ 模型已切换: {os.path.basename(model_path)}")
        else:
            QMessageBox.warning(self, "错误", f"加载模型失败: {model_path}")

    def reload_faces(self):
        """重新加载人脸库"""
        self.access_control.reload_known_faces()
        self.update_ui_info()
        count = self.access_control.get_known_faces_count()
        QMessageBox.information(self, "成功", f"已重新加载 {count} 个人脸编码")

    def closeEvent(self, event):
        """关闭窗口事件"""
        self.timer.stop()
        self.access_control.stop_processing()
        self.cap.release()
        event.accept()

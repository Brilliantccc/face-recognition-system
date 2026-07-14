"""
门禁系统窗口模块
显示摄像头预览和门禁状态
"""

import cv2
import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QMessageBox, QListWidget,
                             QListWidgetItem, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gate.access_control import AccessControl
from common.user_manager import UserManager
from common.database import Database


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
                QMessageBox.critical(self, "Error", "Cannot open camera!")
                sys.exit(1)

        # 启动人脸识别后台线程
        self.access_control.start_processing()

        # 设置定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

        # 初始化UI
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("Face Recognition Access Control System")
        self.setGeometry(100, 100, 1200, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧：摄像头预览
        left_layout = QVBoxLayout()

        # 摄像头预览标签
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        left_layout.addWidget(self.camera_label)

        # 状态显示
        self.status_label = QLabel("Status: Standby")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.status_label.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        left_layout.addWidget(self.status_label)

        main_layout.addLayout(left_layout, 2)

        # 右侧：控制面板
        right_layout = QVBoxLayout()

        # 门禁状态
        status_group = QGroupBox("Access Status")
        status_layout = QVBoxLayout()

        self.user_label = QLabel("User: None")
        self.user_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.user_label)

        self.confidence_label = QLabel("Confidence: 0%")
        self.confidence_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.confidence_label)

        self.faces_count_label = QLabel("Registered Users: 0")
        self.faces_count_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.faces_count_label)

        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        # 操作按钮
        buttons_group = QGroupBox("Operations")
        buttons_layout = QVBoxLayout()

        self.reload_btn = QPushButton("Reload Face Database")
        self.reload_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        self.reload_btn.clicked.connect(self.reload_faces)
        buttons_layout.addWidget(self.reload_btn)

        buttons_group.setLayout(buttons_layout)
        right_layout.addWidget(buttons_group)

        # 门禁日志
        log_group = QGroupBox("Recent Access Log")
        log_layout = QVBoxLayout()

        self.log_list = QListWidget()
        self.log_list.setMaximumHeight(200)
        log_layout.addWidget(self.log_list)

        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, 1)

        # 状态栏
        self.statusBar().showMessage("System Ready")

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
                self.status_label.setText("Status: PASS")
                self.status_label.setStyleSheet("padding: 10px; background-color: #90EE90; font-size: 16px; font-weight: bold;")
            elif status == "Reject":
                self.status_label.setText("Status: REJECT")
                self.status_label.setStyleSheet("padding: 10px; background-color: #FFB6C1; font-size: 16px; font-weight: bold;")
            else:
                self.status_label.setText("Status: Standby")
                self.status_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; font-size: 16px;")

            # 更新用户信息
            if user_name:
                self.user_label.setText(f"User: {user_name}")
                self.confidence_label.setText(f"Confidence: {confidence:.2%}")
            else:
                self.user_label.setText("User: None")
                self.confidence_label.setText("Confidence: 0%")

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
        users = self.user_manager.get_all_users(active_only=True)
        self.faces_count_label.setText(f"Registered Users: {len(users)}")

        # 更新门禁日志
        self.log_list.clear()
        logs = self.db.get_access_log(limit=10)
        for log in logs:
            result_text = "PASS" if log["result"] == "pass" else "REJECT"
            item_text = f"{log['timestamp']} - {log['user_name']} - {result_text}"
            item = QListWidgetItem(item_text)
            if log["result"] == "pass":
                item.setForeground(Qt.darkGreen)
            else:
                item.setForeground(Qt.darkRed)
            self.log_list.addItem(item)

    def reload_faces(self):
        """重新加载人脸库"""
        self.access_control.reload_known_faces()
        self.update_ui_info()
        QMessageBox.information(self, "Success",
            f"Reloaded {self.access_control.get_known_faces_count()} face encodings")

    def closeEvent(self, event):
        """关闭窗口事件"""
        self.timer.stop()
        self.access_control.stop_processing()
        self.cap.release()
        event.accept()

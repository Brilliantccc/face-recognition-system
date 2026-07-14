"""
人事管理窗口模块
提供员工管理和人脸数据录入功能
"""

import cv2
import os
import sys
import face_recognition
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QMessageBox, QDialog,
                             QFormLayout, QLineEdit, QSpinBox, QFileDialog,
                             QListWidget, QListWidgetItem, QGroupBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QComboBox, QTabWidget,
                             QAbstractItemView)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.database import Database
from common.user_manager import UserManager


class AdminWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化数据库和用户管理器
        self.db = Database()
        self.user_manager = UserManager(self.db)

        # 初始化UI
        self.init_ui()

        # 加载用户列表
        self.load_users()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("人脸识别人事管理系统")
        self.setGeometry(100, 100, 1200, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧：用户列表
        left_layout = QVBoxLayout()

        # 搜索框
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索姓名/工号/部门...")
        self.search_input.textChanged.connect(self.search_users)
        search_layout.addWidget(self.search_input)

        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.search_users)
        search_layout.addWidget(search_btn)

        left_layout.addLayout(search_layout)

        # 用户表格
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(6)
        self.user_table.setHorizontalHeaderLabels(["ID", "工号", "姓名", "部门", "人脸数", "状态"])
        self.user_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.user_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.user_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.user_table.itemSelectionChanged.connect(self.on_user_select)
        left_layout.addWidget(self.user_table)

        # 统计信息
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        left_layout.addWidget(self.stats_label)

        main_layout.addLayout(left_layout, 3)

        # 右侧：操作面板
        right_layout = QVBoxLayout()

        # 用户详情
        detail_group = QGroupBox("用户详情")
        detail_layout = QVBoxLayout()

        self.detail_label = QLabel("请选择一个用户")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setMinimumHeight(100)
        detail_layout.addWidget(self.detail_label)

        detail_group.setLayout(detail_layout)
        right_layout.addWidget(detail_group)

        # 人脸照片预览
        photo_group = QGroupBox("人脸照片")
        photo_layout = QVBoxLayout()

        self.photo_list = QListWidget()
        self.photo_list.setMaximumHeight(150)
        photo_layout.addWidget(self.photo_list)

        photo_group.setLayout(photo_layout)
        right_layout.addWidget(photo_group)

        # 操作按钮
        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout()

        add_user_btn = QPushButton("➕ 添加新员工")
        add_user_btn.setStyleSheet("padding: 12px; font-size: 14px; background-color: #4CAF50; color: white;")
        add_user_btn.clicked.connect(self.show_add_user_dialog)
        btn_layout.addWidget(add_user_btn)

        add_face_btn = QPushButton("📷 为选中员工添加人脸")
        add_face_btn.setStyleSheet("padding: 12px; font-size: 14px; background-color: #2196F3; color: white;")
        add_face_btn.clicked.connect(self.show_add_face_dialog)
        btn_layout.addWidget(add_face_btn)

        delete_btn = QPushButton("🗑️ 删除选中员工")
        delete_btn.setStyleSheet("padding: 12px; font-size: 14px; background-color: #f44336; color: white;")
        delete_btn.clicked.connect(self.delete_user)
        btn_layout.addWidget(delete_btn)

        refresh_btn = QPushButton("🔄 刷新列表")
        refresh_btn.clicked.connect(self.load_users)
        btn_layout.addWidget(refresh_btn)

        btn_group.setLayout(btn_layout)
        right_layout.addWidget(btn_group)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, 2)

        # 状态栏
        self.statusBar().showMessage("系统就绪")

    def load_users(self):
        """加载用户列表"""
        users = self.user_manager.get_all_users()

        self.user_table.setRowCount(len(users))
        for i, user in enumerate(users):
            face_count = self.db.get_user_face_count(user["id"])
            status = "启用" if user["is_active"] else "禁用"

            self.user_table.setItem(i, 0, QTableWidgetItem(str(user["id"])))
            self.user_table.setItem(i, 1, QTableWidgetItem(user.get("employee_id", "-")))
            self.user_table.setItem(i, 2, QTableWidgetItem(user["name"]))
            self.user_table.setItem(i, 3, QTableWidgetItem(user.get("department", "-")))
            self.user_table.setItem(i, 4, QTableWidgetItem(str(face_count)))
            self.user_table.setItem(i, 5, QTableWidgetItem(status))

        # 更新统计信息
        stats = self.db.get_statistics()
        self.stats_label.setText(
            f"📊 统计: 总员工 {stats['total_users']} | 启用 {stats['active_users']} | "
            f"人脸照片 {stats['total_faces']}"
        )

    def search_users(self):
        """搜索用户"""
        keyword = self.search_input.text().strip()
        if keyword:
            users = self.user_manager.search_users(keyword)
        else:
            users = self.user_manager.get_all_users()

        self.user_table.setRowCount(len(users))
        for i, user in enumerate(users):
            face_count = self.db.get_user_face_count(user["id"])
            status = "启用" if user["is_active"] else "禁用"

            self.user_table.setItem(i, 0, QTableWidgetItem(str(user["id"])))
            self.user_table.setItem(i, 1, QTableWidgetItem(user.get("employee_id", "-")))
            self.user_table.setItem(i, 2, QTableWidgetItem(user["name"]))
            self.user_table.setItem(i, 3, QTableWidgetItem(user.get("department", "-")))
            self.user_table.setItem(i, 4, QTableWidgetItem(str(face_count)))
            self.user_table.setItem(i, 5, QTableWidgetItem(status))

    def on_user_select(self):
        """用户选中事件"""
        selected_rows = self.user_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        user_id = int(self.user_table.item(row, 0).text())
        user = self.user_manager.get_user(user_id)

        if user:
            # 显示用户详情
            face_count = self.db.get_user_face_count(user_id)
            detail_text = f"""
            <h3>{user['name']}</h3>
            <p><b>工号:</b> {user.get('employee_id', '-')}</p>
            <p><b>部门:</b> {user.get('department', '-')}</p>
            <p><b>电话:</b> {user.get('phone', '-')}</p>
            <p><b>人脸照片:</b> {face_count} 张</p>
            <p><b>注册时间:</b> {user.get('created_at', '-')}</p>
            """
            self.detail_label.setText(detail_text)

            # 显示人脸照片列表
            self.photo_list.clear()
            user_dir = self.user_manager._get_user_dir(user_id, user['name'])
            if os.path.exists(user_dir):
                for file_name in os.listdir(user_dir):
                    if file_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.photo_list.addItem(file_name)

    def show_add_user_dialog(self):
        """显示添加用户对话框"""
        dialog = AddUserDialog(self.user_manager, self)
        if dialog.exec_():
            self.load_users()

    def show_add_face_dialog(self):
        """显示添加人脸对话框"""
        selected_rows = self.user_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择一个员工！")
            return

        row = selected_rows[0].row()
        user_id = int(self.user_table.item(row, 0).text())
        user_name = self.user_table.item(row, 2).text()

        dialog = AddFaceDialog(self.user_manager, user_id, user_name, self)
        if dialog.exec_():
            self.load_users()

    def delete_user(self):
        """删除选中的用户"""
        selected_rows = self.user_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择要删除的员工！")
            return

        row = selected_rows[0].row()
        user_id = int(self.user_table.item(row, 0).text())
        user_name = self.user_table.item(row, 2).text()

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除员工 '{user_name}' 吗？\n该操作将删除该员工的所有人脸数据。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.user_manager.delete_user(user_id):
                QMessageBox.information(self, "成功", f"员工 '{user_name}' 已删除")
                self.load_users()
            else:
                QMessageBox.critical(self, "错误", "删除员工失败")


class AddUserDialog(QDialog):
    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("添加新员工")
        self.setMinimumWidth(400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 表单布局
        form_layout = QFormLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入员工姓名")
        form_layout.addRow("姓名*:", self.name_input)

        self.employee_id_input = QLineEdit()
        self.employee_id_input.setPlaceholderText("请输入工号")
        form_layout.addRow("工号:", self.employee_id_input)

        self.department_input = QLineEdit()
        self.department_input.setPlaceholderText("请输入部门")
        form_layout.addRow("部门:", self.department_input)

        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("请输入联系电话")
        form_layout.addRow("电话:", self.phone_input)

        layout.addLayout(form_layout)

        # 按钮
        btn_layout = QHBoxLayout()

        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_user)
        btn_layout.addWidget(save_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def save_user(self):
        """保存用户"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入员工姓名！")
            return

        employee_id = self.employee_id_input.text().strip() or None
        department = self.department_input.text().strip() or None
        phone = self.phone_input.text().strip() or None

        try:
            self.user_manager.db.add_user(name, employee_id, department, phone)
            QMessageBox.information(self, "成功", f"员工 '{name}' 添加成功！")
            self.accept()
        except ValueError as e:
            QMessageBox.critical(self, "错误", str(e))


class AddFaceDialog(QDialog):
    def __init__(self, user_manager: UserManager, user_id: int, user_name: str, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.user_id = user_id
        self.user_name = user_name
        self.setWindowTitle(f"为 {user_name} 添加人脸")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.captured_images = []

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(30)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 摄像头预览
        preview_group = QGroupBox("摄像头预览")
        preview_layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        preview_layout.addWidget(self.camera_label)

        self.status_label = QLabel(f"已拍摄: 0 张")
        self.status_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.status_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # 按钮
        btn_layout = QHBoxLayout()

        capture_btn = QPushButton("📸 拍摄一张")
        capture_btn.setStyleSheet("padding: 12px; font-size: 14px; background-color: #2196F3; color: white;")
        capture_btn.clicked.connect(self.capture_one)
        btn_layout.addWidget(capture_btn)

        upload_btn = QPushButton("📁 上传照片")
        upload_btn.setStyleSheet("padding: 12px; font-size: 14px;")
        upload_btn.clicked.connect(self.upload_photos)
        btn_layout.addWidget(upload_btn)

        layout.addLayout(btn_layout)

        # 照片列表
        self.photo_list = QListWidget()
        self.photo_list.setMaximumHeight(80)
        layout.addWidget(self.photo_list)

        # 底部按钮
        bottom_layout = QHBoxLayout()

        clear_btn = QPushButton("🗑️ 清空")
        clear_btn.clicked.connect(self.clear_photos)
        bottom_layout.addWidget(clear_btn)

        bottom_layout.addStretch()

        save_btn = QPushButton("✓ 保存")
        save_btn.setStyleSheet("padding: 12px 24px; font-size: 14px; background-color: #4CAF50; color: white;")
        save_btn.clicked.connect(self.save_faces)
        bottom_layout.addWidget(save_btn)

        cancel_btn = QPushButton("✗ 取消")
        cancel_btn.clicked.connect(self.close_dialog)
        bottom_layout.addWidget(cancel_btn)

        layout.addLayout(bottom_layout)

    def update_preview(self):
        """更新摄像头预览"""
        ret, frame = self.cap.read()
        if ret:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small, model="hog")

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left*2, top*2), (right*2, bottom*2), (0, 255, 0), 2)

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)

    def capture_one(self):
        """拍摄一张照片"""
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "错误", "无法读取摄像头！")
            return

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        if len(face_locations) == 0:
            QMessageBox.warning(self, "警告", "未检测到人脸！")
            return

        self.captured_images.append(frame.copy())
        self.photo_list.addItem(f"📸 第 {len(self.captured_images)} 张 ✓")
        self.status_label.setText(f"已拍摄: {len(self.captured_images)} 张")

    def upload_photos(self):
        """上传照片"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择人脸照片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )

        if file_paths:
            valid_count = 0
            for file_path in file_paths:
                image = cv2.imread(file_path)
                if image is not None:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_image, model="hog")
                    if len(face_locations) > 0:
                        self.captured_images.append(image)
                        self.photo_list.addItem(f"📁 {os.path.basename(file_path)} ✓")
                        valid_count += 1

            if valid_count > 0:
                self.status_label.setText(f"已选择: {len(self.captured_images)} 张")
            else:
                QMessageBox.warning(self, "警告", "所有照片都未检测到人脸！")

    def clear_photos(self):
        """清空照片"""
        self.captured_images = []
        self.photo_list.clear()
        self.status_label.setText("已拍摄: 0 张")

    def save_faces(self):
        """保存人脸照片"""
        if len(self.captured_images) == 0:
            QMessageBox.warning(self, "警告", "请先拍摄或上传照片！")
            return

        try:
            count = self.user_manager.add_faces_to_user(self.user_id, self.captured_images)
            QMessageBox.information(self, "成功", f"成功添加 {count} 张人脸照片！")
            self.close_dialog()
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def close_dialog(self):
        """关闭对话框"""
        self.timer.stop()
        self.cap.release()
        self.reject()

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.timer.stop()
        self.cap.release()
        event.accept()

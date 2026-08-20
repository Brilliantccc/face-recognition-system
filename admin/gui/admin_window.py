"""
人事管理窗口模块
提供员工管理和人脸数据录入功能
"""

import cv2
import os
import sys
import shutil
import subprocess

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QMessageBox, QDialog,
                             QFormLayout, QLineEdit, QFileDialog,
                             QListWidget, QListWidgetItem, QGroupBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QCheckBox, QFrame, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.database import Database
from common.user_manager import UserManager
from common.config import WORK_PHOTOS_DIR, FACES_DIR, MIN_FACE_PHOTOS, DATA_DIR
from admin.batch_register import scan_uploads, batch_register_async
from common.theme import (
    GLOBAL_STYLESHEET, btn_primary, btn_success, btn_danger, btn_warning,
    btn_info, btn_outline, CARD_STYLE,
    PRIMARY, PRIMARY_LIGHT, SUCCESS, DANGER, WARNING, BG_MAIN, BG_CARD,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, BORDER, DIVIDER
)

import numpy as np


_face_recognition_module = None
_face_recognition_loaded = False


def _get_face_recognition():
    """延迟加载 face_recognition（加载失败返回 None，不崩溃）"""
    global _face_recognition_module, _face_recognition_loaded
    if _face_recognition_loaded:
        return _face_recognition_module
    _face_recognition_loaded = True
    try:
        import face_recognition
        _face_recognition_module = face_recognition
    except Exception:
        _face_recognition_module = None
    return _face_recognition_module


def _safe_face_locations(image, model="hog"):
    """安全的人脸检测（face_recognition 不可用时返回空列表）"""
    fr = _get_face_recognition()
    if fr is None:
        return []
    return fr.face_locations(image, model=model)


def _safe_face_encodings(image, face_locations):
    """安全的人脸编码（face_recognition 不可用时返回空列表）"""
    fr = _get_face_recognition()
    if fr is None:
        return []
    return fr.face_encodings(image, face_locations)


def imread_safe(filepath):
    """读取图片，支持中文路径（Windows）"""
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ============================================================
# 辅助函数
# ============================================================

def create_separator():
    """创建水平分隔线"""
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet(f"background-color: {DIVIDER}; max-height: 1px; margin: 4px 0;")
    return sep


def create_title_label(text, size=20):
    """创建标题标签"""
    label = QLabel(text)
    label.setStyleSheet(f"font-size: {size}px; font-weight: bold; color: {PRIMARY}; padding: 8px 0;")
    return label


# ============================================================
# 主窗口
# ============================================================

class AdminWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化数据库和用户管理器
        self.db = Database()
        self.user_manager = UserManager(self.db)

        # 多进程导入相关
        self._import_proc = None
        self._import_queue = None
        self._import_timer = QTimer(self)
        self._import_timer.timeout.connect(self._poll_import_result)

        # 初始化UI
        self.init_ui()

        # 加载用户列表
        self.load_users()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("👤 人脸识别人事管理系统")
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

        # ── 左侧：用户列表区 ──
        left_panel = QWidget()
        left_panel.setStyleSheet(f"background-color: {BG_CARD}; border-radius: 12px; border: 1px solid {BORDER};")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        # 标题
        left_title = create_title_label("📋 员工列表")
        left_layout.addWidget(left_title)

        # 搜索框
        search_layout = QHBoxLayout()
        search_layout.setSpacing(8)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 搜索姓名 / 工号 / 部门...")
        self.search_input.setMinimumHeight(36)
        self.search_input.textChanged.connect(self.search_users)
        search_layout.addWidget(self.search_input)

        search_btn = QPushButton("搜索")
        search_btn.setFixedWidth(70)
        search_btn.setStyleSheet(btn_primary("min-height: 36px;"))
        search_btn.clicked.connect(self.search_users)
        search_layout.addWidget(search_btn)

        left_layout.addLayout(search_layout)

        # 用户表格
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(5)
        self.user_table.setHorizontalHeaderLabels(["工号", "姓名", "部门", "人脸数", "状态"])
        self.user_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.user_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.user_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.user_table.setAlternatingRowColors(True)
        self.user_table.verticalHeader().setVisible(False)
        self.user_table.setShowGrid(False)
        self.user_table.itemSelectionChanged.connect(self.on_user_select)
        left_layout.addWidget(self.user_table)

        # 统计信息
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet(
            f"padding: 10px 14px; background-color: {PRIMARY_LIGHT}; "
            f"border-radius: 8px; font-size: 13px; color: {TEXT_SECONDARY};"
        )
        left_layout.addWidget(self.stats_label)

        main_layout.addWidget(left_panel, 3)

        # ── 右侧：操作面板 ──
        right_panel = QWidget()
        right_panel.setStyleSheet(f"background-color: {BG_CARD}; border-radius: 12px; border: 1px solid {BORDER};")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        # 工作照片区域
        photo_card = QWidget()
        photo_card.setStyleSheet(f"QGroupBox {{ {CARD_STYLE} }}")
        photo_layout = QVBoxLayout(photo_card)
        photo_layout.setContentsMargins(12, 20, 12, 12)

        photo_title = QLabel("👤 工作照片")
        photo_title.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TEXT_PRIMARY}; margin-bottom: 8px;")
        photo_layout.addWidget(photo_title)

        # 照片预览
        self.work_photo_preview = QLabel()
        self.work_photo_preview.setAlignment(Qt.AlignCenter)
        self.work_photo_preview.setMinimumSize(200, 200)
        self.work_photo_preview.setStyleSheet(
            f"border: 2px dashed {BORDER}; background-color: {BG_MAIN}; "
            f"color: {TEXT_MUTED}; font-size: 14px; border-radius: 12px;"
        )
        self.work_photo_preview.setText("📋 暂无工作照片\n请上传一张")
        photo_layout.addWidget(self.work_photo_preview)

        # 照片操作按钮
        photo_btn_layout = QHBoxLayout()
        photo_btn_layout.setSpacing(8)

        upload_work_btn = QPushButton("📁 上传工作照片")
        upload_work_btn.setStyleSheet(btn_info("min-height: 34px;"))
        upload_work_btn.clicked.connect(self.upload_work_photo)
        photo_btn_layout.addWidget(upload_work_btn)

        clear_work_btn = QPushButton("🗑️ 清除")
        clear_work_btn.setStyleSheet(btn_danger("min-height: 34px;"))
        clear_work_btn.clicked.connect(self.clear_work_photo)
        photo_btn_layout.addWidget(clear_work_btn)

        photo_layout.addLayout(photo_btn_layout)
        right_layout.addWidget(photo_card)

        # 用户详情
        detail_card = QWidget()
        detail_card.setStyleSheet(f"QGroupBox {{ {CARD_STYLE} }}")
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(12, 20, 12, 12)

        detail_title = QLabel("📄 用户详情")
        detail_title.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TEXT_PRIMARY}; margin-bottom: 8px;")
        detail_layout.addWidget(detail_title)

        self.detail_label = QLabel("请选择一个用户查看详情")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setMinimumHeight(100)
        self.detail_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 13px; background-color: {BG_MAIN}; "
            f"border-radius: 8px; padding: 16px;"
        )
        detail_layout.addWidget(self.detail_label)
        right_layout.addWidget(detail_card)

        # 操作按钮区
        op_card = QWidget()
        op_card.setStyleSheet(f"QGroupBox {{ {CARD_STYLE} }}")
        op_layout = QVBoxLayout(op_card)
        op_layout.setContentsMargins(12, 20, 12, 12)
        op_layout.setSpacing(8)

        op_title = QLabel("⚡ 快捷操作")
        op_title.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TEXT_PRIMARY}; margin-bottom: 8px;")
        op_layout.addWidget(op_title)

        # 第一行：主要操作
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        add_user_btn = QPushButton("➕ 添加新员工")
        add_user_btn.setStyleSheet(btn_success("min-height: 42px; font-size: 14px;"))
        add_user_btn.clicked.connect(self.show_add_user_dialog)
        row1.addWidget(add_user_btn)

        quick_register_btn = QPushButton("⚡ 快速注册")
        quick_register_btn.setToolTip("一步完成：输入信息 + 摄像头拍照 + 自动注册")
        quick_register_btn.setStyleSheet(btn_primary("min-height: 42px; font-size: 14px;"))
        quick_register_btn.clicked.connect(self.show_quick_register_dialog)
        row1.addWidget(quick_register_btn)

        op_layout.addLayout(row1)

        # 第二行：导入操作
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        batch_import_btn = QPushButton("📂 批量导入")
        batch_import_btn.setToolTip("从文件夹批量导入人员（格式：文件夹/姓名/照片）")
        batch_import_btn.setStyleSheet(btn_info("min-height: 42px; font-size: 14px;"))
        batch_import_btn.clicked.connect(self.show_batch_import_dialog)
        row2.addWidget(batch_import_btn)

        import_uploads_btn = QPushButton("📥 导入 uploads")
        import_uploads_btn.setToolTip("一键导入 data/uploads 目录下已准备好的人员照片")
        import_uploads_btn.setStyleSheet(btn_warning("min-height: 42px; font-size: 14px;"))
        import_uploads_btn.clicked.connect(self.import_from_uploads)
        row2.addWidget(import_uploads_btn)

        op_layout.addLayout(row2)

        # 第三行：编辑操作
        row3 = QHBoxLayout()
        row3.setSpacing(8)

        edit_btn = QPushButton("✏️ 编辑员工")
        edit_btn.setStyleSheet(btn_info("min-height: 42px; font-size: 14px;"))
        edit_btn.clicked.connect(self.show_edit_user_dialog)
        row3.addWidget(edit_btn)

        add_face_btn = QPushButton("📷 补充人脸")
        add_face_btn.setToolTip("为已有员工添加更多人脸照片，提升识别准确率")
        add_face_btn.setStyleSheet(btn_primary("min-height: 42px; font-size: 14px;"))
        add_face_btn.clicked.connect(self.show_add_face_dialog)
        row3.addWidget(add_face_btn)

        op_layout.addLayout(row3)

        # 分隔线
        op_layout.addWidget(create_separator())

        # 底部行：删除和刷新
        row4 = QHBoxLayout()
        row4.setSpacing(8)

        refresh_btn = QPushButton("🔄 刷新列表")
        refresh_btn.setStyleSheet(btn_outline("min-height: 38px;"))
        refresh_btn.clicked.connect(self.load_users)
        row4.addWidget(refresh_btn)

        row4.addStretch()

        delete_btn = QPushButton("🗑️ 删除员工")
        delete_btn.setStyleSheet(btn_danger("min-height: 38px;"))
        delete_btn.clicked.connect(self.delete_user)
        row4.addWidget(delete_btn)

        op_layout.addLayout(row4)
        right_layout.addWidget(op_card)

        main_layout.addWidget(right_panel, 2)

        # 状态栏
        self.statusBar().showMessage("✅ 系统就绪")
        self.statusBar().setStyleSheet(
            f"background-color: {BG_CARD}; color: {TEXT_SECONDARY}; "
            f"border-top: 1px solid {BORDER}; padding: 6px 16px; font-size: 12px;"
        )

    def _populate_table(self, users):
        """填充用户表格（共用方法）"""
        self.user_table.setRowCount(len(users))
        for i, user in enumerate(users):
            face_count = self.db.get_user_face_count(user["id"])
            is_active = user["is_active"]

            self.user_table.setItem(i, 0, QTableWidgetItem(user.get("employee_id", "-")))
            self.user_table.setItem(i, 1, QTableWidgetItem(user["name"]))
            self.user_table.setItem(i, 2, QTableWidgetItem(user.get("department", "-")))

            # 人脸数
            face_item = QTableWidgetItem(str(face_count))
            face_item.setTextAlignment(Qt.AlignCenter)
            if face_count < MIN_FACE_PHOTOS:
                face_item.setForeground(QColor(WARNING))
            else:
                face_item.setForeground(QColor(SUCCESS))
            self.user_table.setItem(i, 3, face_item)

            # 状态
            status = "✅ 启用" if is_active else "❌ 禁用"
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            if not is_active:
                status_item.setForeground(QColor(DANGER))
            self.user_table.setItem(i, 4, status_item)

            # 将用户ID存储到工号列的UserRole中
            self.user_table.item(i, 0).setData(Qt.UserRole, user["id"])

        # 设置行高
        for row in range(self.user_table.rowCount()):
            self.user_table.setRowHeight(row, 42)

    def _get_selected_user_id(self):
        """获取当前选中的用户ID"""
        selected_rows = self.user_table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        row = selected_rows[0].row()
        return self.user_table.item(row, 0).data(Qt.UserRole)

    def load_users(self):
        """加载用户列表"""
        users = self.user_manager.get_all_users()
        self._populate_table(users)

        # 更新统计信息
        stats = self.db.get_statistics()
        self.stats_label.setText(
            f"📊 总员工 <b>{stats['total_users']}</b>  |  "
            f"启用 <b style='color:{SUCCESS}'>{stats['active_users']}</b>  |  "
            f"人脸照片 <b>{stats['total_faces']}</b> 张"
        )

    def search_users(self):
        """搜索用户"""
        keyword = self.search_input.text().strip()
        if keyword:
            users = self.user_manager.search_users(keyword)
        else:
            users = self.user_manager.get_all_users()

        self._populate_table(users)

    def on_user_select(self):
        """用户选中事件"""
        user_id = self._get_selected_user_id()
        if user_id is None:
            return

        user = self.user_manager.get_user(user_id)

        if user:
            face_count = self.db.get_user_face_count(user_id)
            detail_text = (
                f"<div style='font-size:16px; font-weight:bold; color:{PRIMARY}; margin-bottom:8px;'>"
                f"👤 {user['name']}</div>"
                f"<table style='font-size:13px; line-height:1.8;'>"
                f"<tr><td style='color:{TEXT_MUTED}; padding-right:12px;'>工号</td>"
                f"<td><b>{user.get('employee_id') or '-'}</b></td></tr>"
                f"<tr><td style='color:{TEXT_MUTED}; padding-right:12px;'>部门</td>"
                f"<td><b>{user.get('department') or '-'}</b></td></tr>"
                f"<tr><td style='color:{TEXT_MUTED}; padding-right:12px;'>电话</td>"
                f"<td><b>{user.get('phone') or '-'}</b></td></tr>"
                f"<tr><td style='color:{TEXT_MUTED}; padding-right:12px;'>人脸照片</td>"
                f"<td><b style='color:{SUCCESS if face_count >= MIN_FACE_PHOTOS else WARNING}'>"
                f"{face_count} 张</b></td></tr>"
                f"<tr><td style='color:{TEXT_MUTED}; padding-right:12px;'>注册时间</td>"
                f"<td><b>{user.get('created_at', '-')}</b></td></tr>"
                f"</table>"
            )
            self.detail_label.setText(detail_text)
            self.detail_label.setStyleSheet(
                f"background-color: {BG_MAIN}; border-radius: 8px; padding: 16px; "
                f"font-size: 13px; color: {TEXT_PRIMARY}; text-align: left;"
            )

            # 加载工作照片
            self._load_work_photo(user)

    def _set_work_photo_placeholder(self):
        """设置工作照片预览区域的占位提示"""
        self.work_photo_preview.setText("📋 暂无工作照片\n请上传一张")
        self.work_photo_preview.setStyleSheet(
            f"border: 2px dashed {BORDER}; background-color: {BG_MAIN}; "
            f"color: {TEXT_MUTED}; font-size: 14px; border-radius: 12px;"
        )
        self.work_photo_preview.setPixmap(QPixmap())

    def _load_work_photo(self, user):
        """加载并显示用户的工作照片"""
        work_photo_path = user.get('work_photo')
        if work_photo_path and os.path.exists(work_photo_path):
            image = imread_safe(work_photo_path)
            if image is not None:
                # 人脸检测并绘制方框
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = _safe_face_locations(rgb_image, "hog")
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

                h, w, ch = image.shape
                bytes_per_line = ch * w
                qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qt_image)
                if not pixmap.isNull():
                    self.work_photo_preview.setStyleSheet(
                        f"border: 2px solid {SUCCESS}; background-color: #fff; border-radius: 12px;"
                    )
                    scaled_pixmap = pixmap.scaled(
                        self.work_photo_preview.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.work_photo_preview.setPixmap(scaled_pixmap)
                    return
        self._set_work_photo_placeholder()

    def upload_work_photo(self):
        """上传工作照片"""
        user_id = self._get_selected_user_id()
        if user_id is None:
            QMessageBox.warning(self, "提示", "请先选择一个员工！")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择工作照片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if not file_path:
            return

        image = imread_safe(file_path)
        if image is None:
            QMessageBox.warning(self, "提示", "无法读取图片文件！")
            return

        # 检测人脸
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = _safe_face_locations(rgb_image, "hog")
        if len(face_locations) == 0:
            QMessageBox.warning(self, "提示", "照片中未检测到人脸，请选择包含人脸的照片！")
            return

        # 保存到工作照片目录
        os.makedirs(WORK_PHOTOS_DIR, exist_ok=True)
        save_path = os.path.join(WORK_PHOTOS_DIR, f"{user_id}.jpg")
        cv2.imwrite(save_path, image)

        # 更新数据库
        if self.db.set_work_photo(user_id, save_path):
            QMessageBox.information(self, "成功", "工作照片上传成功！")
            user = self.user_manager.get_user(user_id)
            if user:
                self._load_work_photo(user)
        else:
            QMessageBox.critical(self, "错误", "保存失败，请稍后重试")

    def clear_work_photo(self):
        """清除工作照片"""
        user_id = self._get_selected_user_id()
        if user_id is None:
            QMessageBox.warning(self, "提示", "请先选择一个员工！")
            return

        user = self.user_manager.get_user(user_id)
        work_photo_path = user.get('work_photo') if user else None

        if not work_photo_path:
            QMessageBox.information(self, "提示", "该员工没有设置工作照片")
            return

        reply = QMessageBox.question(
            self, "确认清除",
            "确定要清除该员工的工作照片吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if work_photo_path and os.path.exists(work_photo_path):
                os.remove(work_photo_path)
            self.db.clear_work_photo(user_id)
            self._set_work_photo_placeholder()
            QMessageBox.information(self, "成功", "工作照片已清除")

    def show_add_user_dialog(self):
        """显示添加用户对话框"""
        dialog = AddUserDialog(self.user_manager, self)
        if dialog.exec_():
            self.load_users()

    def show_edit_user_dialog(self):
        """显示编辑用户对话框"""
        user_id = self._get_selected_user_id()
        if user_id is None:
            QMessageBox.warning(self, "提示", "请先选择要编辑的员工！")
            return

        user = self.user_manager.get_user(user_id)
        if user:
            dialog = EditUserDialog(self.user_manager, user, self)
            if dialog.exec_():
                self.load_users()

    def show_add_face_dialog(self):
        """显示添加人脸对话框"""
        user_id = self._get_selected_user_id()
        if user_id is None:
            QMessageBox.warning(self, "提示", "请先选择一个员工！")
            return

        user = self.user_manager.get_user(user_id)
        user_name = user["name"] if user else ""

        dialog = AddFaceDialog(self.user_manager, user_id, user_name, self)
        if dialog.exec_():
            self.load_users()

    def show_quick_register_dialog(self):
        """显示快速注册对话框"""
        dialog = QuickRegisterDialog(self.user_manager, self)
        if dialog.exec_():
            self.load_users()

    def show_batch_import_dialog(self):
        """显示批量导入对话框"""
        dialog = BatchImportDialog(self.user_manager, self)
        if dialog.exec_():
            self.load_users()

    def import_from_uploads(self):
        """一键从 data/uploads 目录导入人员"""
        try:
            self._do_import_from_uploads()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败：{str(e)}")

    def _do_import_from_uploads(self):
        """一键从 data/uploads 目录导入人员（多进程，dlib 崩溃不影响 GUI）"""
        uploads_dir = os.path.join(DATA_DIR, "uploads")
        try:
            persons = scan_uploads(uploads_dir)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"扫描 uploads 目录失败：{str(e)}")
            return

        if not persons:
            QMessageBox.warning(self, "提示",
                "data/uploads 目录下没有找到有效的人员照片。\n\n"
                "请按以下格式放置：\n"
                "  data/uploads/张三/001.jpg\n"
                "  data/uploads/李四/001.jpg")
            return

        total = sum(p['count'] for p in persons)
        person_list = "\n".join([f"  • {p['name']}（{p['count']} 张）" for p in persons])

        reply = QMessageBox.question(
            self, "确认导入",
            f"检测到 {len(persons)} 个人员，共 {total} 张照片：\n\n{person_list}\n\n"
            f"是否导入？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        # 启动子进程执行导入（dlib 在子进程中加载，崩溃不影响 GUI）
        proc, queue, _ = batch_register_async(uploads_dir, None)
        self._import_proc = proc
        self._import_queue = queue
        self._import_timer.start(200)  # 每 200ms 轮询一次

    def _poll_import_result(self):
        """轮询子进程结果"""
        if self._import_queue is None or not self._import_queue.empty():
            # 取出结果
            status, data = self._import_queue.get()
            self._import_timer.stop()
            self._import_proc = None
            self._import_queue = None

            if status == 'error':
                QMessageBox.critical(self, "导入失败", f"子进程错误：{data}")
            else:
                result = data
                msg = f"✅ 导入完成！\n\n新增: {result['success']} 人"
                if result.get('reactivate', 0) > 0:
                    msg += f"\n恢复（之前删除的）: {result['reactivate']} 人"
                if result['skip'] > 0:
                    msg += f"\n跳过（已存在）: {result['skip']} 人"
                if result['errors']:
                    msg += f"\n失败: {result['fail']} 人\n\n" + "\n".join(result['errors'][:5])
                msg += "\n\n⚠️ 人脸编码需要启动门禁系统时自动生成"
                QMessageBox.information(self, "导入结果", msg)
                self.load_users()
            return

        # 子进程还在运行，检查是否已退出
        if self._import_proc and not self._import_proc.is_alive():
            # 进程已退出但队列为空 = 崩溃
            self._import_timer.stop()
            exit_code = self._import_proc.exitcode
            self._import_proc = None
            self._import_queue = None
            QMessageBox.critical(self, "导入失败",
                f"子进程异常退出（exit code: {exit_code}）\n\n"
                "可能是 dlib/face_recognition 加载失败。\n"
                "请检查：\n"
                "1. 是否安装了 dlib：pip install dlib\n"
                "2. 是否安装了 Visual C++ Build Tools")

    def delete_user(self):
        """删除选中的用户"""
        user_id = self._get_selected_user_id()
        if user_id is None:
            QMessageBox.warning(self, "提示", "请先选择要删除的员工！")
            return

        user = self.user_manager.get_user(user_id)
        user_name = user["name"] if user else ""

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除员工 '{user_name}' 吗？\n该操作将删除该员工的所有人脸数据。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            work_photo = user.get('work_photo') if user else None
            if work_photo and os.path.exists(work_photo):
                os.remove(work_photo)

            if self.user_manager.delete_user(user_id):
                QMessageBox.information(self, "成功", f"员工 '{user_name}' 已删除")
                self.load_users()
            else:
                QMessageBox.critical(self, "错误", "删除员工失败")


# ============================================================
# 添加用户对话框
# ============================================================

class AddUserDialog(QDialog):
    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("➕ 添加新员工")
        self.setMinimumWidth(420)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # 标题
        title = QLabel("添加新员工")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {PRIMARY};")
        layout.addWidget(title)

        # 表单
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入员工姓名")
        self.name_input.setMinimumHeight(36)
        form_layout.addRow("姓名 *:", self.name_input)

        self.employee_id_input = QLineEdit()
        self.employee_id_input.setPlaceholderText("请输入工号")
        self.employee_id_input.setMinimumHeight(36)
        form_layout.addRow("工号:", self.employee_id_input)

        self.department_input = QLineEdit()
        self.department_input.setPlaceholderText("请输入部门")
        self.department_input.setMinimumHeight(36)
        form_layout.addRow("部门:", self.department_input)

        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("请输入联系电话")
        self.phone_input.setMinimumHeight(36)
        form_layout.addRow("电话:", self.phone_input)

        layout.addLayout(form_layout)
        layout.addStretch()

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(btn_outline())
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        btn_layout.addStretch()

        save_btn = QPushButton("✓ 保存")
        save_btn.setStyleSheet(btn_success("min-height: 38px; min-width: 100px;"))
        save_btn.clicked.connect(self.save_user)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def save_user(self):
        """保存用户"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入员工姓名！")
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


# ============================================================
# 编辑用户对话框
# ============================================================

class EditUserDialog(QDialog):
    """编辑用户对话框"""
    def __init__(self, user_manager: UserManager, user: dict, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.user = user
        self.setWindowTitle("✏️ 编辑员工信息")
        self.setMinimumWidth(420)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # 标题
        title = QLabel(f"编辑 — {self.user.get('name', '')}")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {PRIMARY};")
        layout.addWidget(title)

        # 表单
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.name_input = QLineEdit(self.user.get('name') or '')
        self.name_input.setMinimumHeight(36)
        form_layout.addRow("姓名 *:", self.name_input)

        self.employee_id_input = QLineEdit(self.user.get('employee_id') or '')
        self.employee_id_input.setMinimumHeight(36)
        form_layout.addRow("工号:", self.employee_id_input)

        self.department_input = QLineEdit(self.user.get('department') or '')
        self.department_input.setMinimumHeight(36)
        form_layout.addRow("部门:", self.department_input)

        self.phone_input = QLineEdit(self.user.get('phone') or '')
        self.phone_input.setMinimumHeight(36)
        form_layout.addRow("电话:", self.phone_input)

        layout.addLayout(form_layout)
        layout.addStretch()

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(btn_outline())
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        btn_layout.addStretch()

        save_btn = QPushButton("✓ 保存修改")
        save_btn.setStyleSheet(btn_primary("min-height: 38px; min-width: 120px;"))
        save_btn.clicked.connect(self.save_user)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def save_user(self):
        """保存修改后的用户信息"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入员工姓名！")
            return

        employee_id = self.employee_id_input.text().strip() or None
        department = self.department_input.text().strip() or None
        phone = self.phone_input.text().strip() or None

        try:
            success = self.user_manager.db.update_user(
                self.user['id'],
                name=name,
                employee_id=employee_id,
                department=department,
                phone=phone
            )
            if success:
                QMessageBox.information(self, "成功", f"员工 '{name}' 信息已更新！")
                self.accept()
            else:
                QMessageBox.critical(self, "错误", "更新失败，请稍后重试")
        except ValueError as e:
            QMessageBox.critical(self, "错误", str(e))


# ============================================================
# 补充人脸对话框
# ============================================================

class AddFaceDialog(QDialog):
    def __init__(self, user_manager: UserManager, user_id: int, user_name: str, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.user_id = user_id
        self.user_name = user_name
        self.setWindowTitle(f"📷 补充人脸照片 — {user_name}")
        self.setMinimumWidth(620)
        self.setMinimumHeight(520)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # 获取当前人脸照片数量
        self.current_face_count = user_manager.db.get_user_face_count(user_id)

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
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # 提示信息
        tip = QLabel(
            f"📋 当前已有 <b>{self.current_face_count}</b> 张  |  "
            f"💡 建议每人至少 20 张，识别更准确"
        )
        tip.setStyleSheet(
            f"font-size: 13px; color: {TEXT_SECONDARY}; padding: 10px; "
            f"background: {PRIMARY_LIGHT}; border-radius: 8px;"
        )
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)

        # 摄像头预览
        preview_card = QGroupBox("📷 摄像头预览")
        preview_layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setStyleSheet(
            f"border: 2px solid #333; background-color: #000; border-radius: 10px;"
        )
        preview_layout.addWidget(self.camera_label)

        self.status_label = QLabel(f"已拍摄: 0 张")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"font-size: 14px; padding: 8px; color: {TEXT_SECONDARY};")
        preview_layout.addWidget(self.status_label)

        preview_card.setLayout(preview_layout)
        layout.addWidget(preview_card)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        capture_btn = QPushButton("📸 拍摄一张")
        capture_btn.setStyleSheet(btn_primary("min-height: 40px; font-size: 14px;"))
        capture_btn.clicked.connect(self.capture_one)
        btn_layout.addWidget(capture_btn)

        upload_btn = QPushButton("📁 上传照片")
        upload_btn.setStyleSheet(btn_info("min-height: 40px; font-size: 14px;"))
        upload_btn.clicked.connect(self.upload_photos)
        btn_layout.addWidget(upload_btn)

        layout.addLayout(btn_layout)

        # 照片列表
        self.photo_list = QListWidget()
        self.photo_list.setMaximumHeight(80)
        layout.addWidget(self.photo_list)

        # 底部按钮
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        clear_btn = QPushButton("🗑️ 清空")
        clear_btn.setStyleSheet(btn_danger())
        clear_btn.clicked.connect(self.clear_photos)
        bottom_layout.addWidget(clear_btn)

        bottom_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(btn_outline())
        cancel_btn.clicked.connect(self.close_dialog)
        bottom_layout.addWidget(cancel_btn)

        save_btn = QPushButton("✓ 保存")
        save_btn.setStyleSheet(btn_success("min-height: 38px; min-width: 100px;"))
        save_btn.clicked.connect(self.save_faces)
        bottom_layout.addWidget(save_btn)

        layout.addLayout(bottom_layout)

    def update_preview(self):
        """更新摄像头预览"""
        ret, frame = self.cap.read()
        if ret:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = _safe_face_locations(rgb_small, "hog")

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
        face_locations = _safe_face_locations(rgb_frame, "hog")

        if len(face_locations) == 0:
            QMessageBox.warning(self, "提示", "未检测到人脸！")
            return

        self.captured_images.append(frame.copy())
        self.photo_list.addItem(f"📸 第 {len(self.captured_images)} 张 ✓")
        self.status_label.setText(f"已拍摄: {len(self.captured_images)} 张")
        self.status_label.setStyleSheet(
            f"font-size: 14px; padding: 8px; color: {SUCCESS}; font-weight: bold;"
        )

    def upload_photos(self):
        """上传照片"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择人脸照片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )

        if file_paths:
            valid_count = 0
            for file_path in file_paths:
                image = imread_safe(file_path)
                if image is not None:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = _safe_face_locations(rgb_image, "hog")
                    if len(face_locations) > 0:
                        self.captured_images.append(image)
                        self.photo_list.addItem(f"📁 {os.path.basename(file_path)} ✓")
                        valid_count += 1

            if valid_count > 0:
                self.status_label.setText(f"已选择: {len(self.captured_images)} 张")
                self.status_label.setStyleSheet(
                    f"font-size: 14px; padding: 8px; color: {SUCCESS}; font-weight: bold;"
                )
            else:
                QMessageBox.warning(self, "提示", "所有照片都未检测到人脸！")

    def clear_photos(self):
        """清空照片"""
        self.captured_images = []
        self.photo_list.clear()
        self.status_label.setText("已拍摄: 0 张")
        self.status_label.setStyleSheet(f"font-size: 14px; padding: 8px; color: {TEXT_SECONDARY};")

    def save_faces(self):
        """保存人脸照片"""
        if len(self.captured_images) == 0:
            QMessageBox.warning(self, "提示", "请先拍摄或上传照片！")
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


# ============================================================
# 快速注册对话框
# ============================================================

class QuickRegisterDialog(QDialog):
    """快速注册对话框 — 一步完成：输入信息 + 摄像头拍照"""
    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("⚡ 快速注册员工")
        self.setMinimumWidth(720)
        self.setMinimumHeight(620)
        self.setStyleSheet(GLOBAL_STYLESHEET)

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
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # 标题提示
        tip = QLabel("💡 填写信息 → 对准摄像头拍照（至少5张）→ 点击注册")
        tip.setStyleSheet(
            f"font-size: 13px; color: {TEXT_SECONDARY}; padding: 10px; "
            f"background: {PRIMARY_LIGHT}; border-radius: 8px;"
        )
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)

        # 信息表单
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("必填")
        self.name_input.setMinimumHeight(34)
        form_layout.addRow("姓名 *:", self.name_input)

        self.employee_id_input = QLineEdit()
        self.employee_id_input.setPlaceholderText("可选")
        self.employee_id_input.setMinimumHeight(34)
        form_layout.addRow("工号:", self.employee_id_input)

        self.department_input = QLineEdit()
        self.department_input.setPlaceholderText("可选")
        self.department_input.setMinimumHeight(34)
        form_layout.addRow("部门:", self.department_input)

        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("可选")
        self.phone_input.setMinimumHeight(34)
        form_layout.addRow("电话:", self.phone_input)

        layout.addLayout(form_layout)

        # 摄像头预览
        preview_card = QGroupBox("📷 摄像头预览")
        preview_layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setStyleSheet(
            f"border: 2px solid #333; background-color: #000; border-radius: 10px;"
        )
        preview_layout.addWidget(self.camera_label)

        self.status_label = QLabel(f"已拍摄: 0 张（最少需要 {MIN_FACE_PHOTOS} 张）")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"font-size: 13px; padding: 8px; color: {TEXT_SECONDARY};")
        preview_layout.addWidget(self.status_label)

        preview_card.setLayout(preview_layout)
        layout.addWidget(preview_card)

        # 拍照按钮
        capture_layout = QHBoxLayout()
        capture_layout.setSpacing(10)

        capture_btn = QPushButton("📸 拍摄一张")
        capture_btn.setStyleSheet(btn_primary("min-height: 40px; font-size: 14px;"))
        capture_btn.clicked.connect(self.capture_one)
        capture_layout.addWidget(capture_btn)

        upload_btn = QPushButton("📁 批量上传")
        upload_btn.setStyleSheet(btn_info("min-height: 40px; font-size: 14px;"))
        upload_btn.clicked.connect(self.upload_photos)
        capture_layout.addWidget(upload_btn)

        clear_btn = QPushButton("🗑️ 清空重拍")
        clear_btn.setStyleSheet(btn_danger("min-height: 40px; font-size: 14px;"))
        clear_btn.clicked.connect(self.clear_photos)
        capture_layout.addWidget(clear_btn)

        layout.addLayout(capture_layout)

        # 底部按钮
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(12)
        bottom_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(btn_outline("min-height: 38px;"))
        cancel_btn.clicked.connect(self.close_dialog)
        bottom_layout.addWidget(cancel_btn)

        register_btn = QPushButton("✓ 一键注册")
        register_btn.setStyleSheet(btn_success("min-height: 44px; min-width: 140px; font-size: 16px;"))
        register_btn.clicked.connect(self.register_user)
        bottom_layout.addWidget(register_btn)

        layout.addLayout(bottom_layout)

    def update_preview(self):
        """更新摄像头预览"""
        ret, frame = self.cap.read()
        if ret:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = _safe_face_locations(rgb_small, "hog")

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
        face_locations = _safe_face_locations(rgb_frame, "hog")

        if len(face_locations) == 0:
            QMessageBox.warning(self, "提示", "未检测到人脸！请正对摄像头，保持光线充足。")
            return

        self.captured_images.append(frame.copy())
        count = len(self.captured_images)
        self.status_label.setText(f"已拍摄: {count} 张（最少需要 {MIN_FACE_PHOTOS} 张）")
        if count >= MIN_FACE_PHOTOS:
            self.status_label.setStyleSheet(
                f"font-size: 13px; padding: 8px; color: {SUCCESS}; font-weight: bold;"
            )
        else:
            self.status_label.setStyleSheet(
                f"font-size: 13px; padding: 8px; color: {WARNING};"
            )

    def upload_photos(self):
        """上传照片"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择人脸照片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )

        if file_paths:
            valid_count = 0
            for file_path in file_paths:
                image = imread_safe(file_path)
                if image is not None:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = _safe_face_locations(rgb_image, "hog")
                    if len(face_locations) > 0:
                        self.captured_images.append(image)
                        valid_count += 1

            count = len(self.captured_images)
            self.status_label.setText(f"已选择: {count} 张（最少需要 {MIN_FACE_PHOTOS} 张）")
            if count >= MIN_FACE_PHOTOS:
                self.status_label.setStyleSheet(
                    f"font-size: 13px; padding: 8px; color: {SUCCESS}; font-weight: bold;"
                )
            else:
                self.status_label.setStyleSheet(
                    f"font-size: 13px; padding: 8px; color: {WARNING};"
                )

            if valid_count > 0:
                QMessageBox.information(self, "成功", f"成功添加 {valid_count} 张照片")
            else:
                QMessageBox.warning(self, "提示", "所有照片都未检测到人脸！")

    def clear_photos(self):
        """清空照片"""
        self.captured_images = []
        self.status_label.setText(f"已拍摄: 0 张（最少需要 {MIN_FACE_PHOTOS} 张）")
        self.status_label.setStyleSheet(f"font-size: 13px; padding: 8px; color: {TEXT_SECONDARY};")

    def register_user(self):
        """一键注册"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入员工姓名！")
            return

        if len(self.captured_images) < MIN_FACE_PHOTOS:
            QMessageBox.warning(self, "提示",
                f"至少需要拍摄 {MIN_FACE_PHOTOS} 张人脸照片！\n当前已拍摄: {len(self.captured_images)} 张")
            return

        employee_id = self.employee_id_input.text().strip() or None
        department = self.department_input.text().strip() or None
        phone = self.phone_input.text().strip() or None

        try:
            user_id = self.user_manager.register_user(
                name=name,
                images=self.captured_images,
                employee_id=employee_id,
                department=department,
                phone=phone
            )
            QMessageBox.information(self, "注册成功",
                f"✅ 员工 '{name}' 注册成功！\n\n"
                f"用户ID: {user_id}\n"
                f"人脸照片: {len(self.captured_images)} 张\n\n"
                f"如需门禁识别，请在训练平台中训练模型。")
            self.accept()
        except ValueError as e:
            QMessageBox.critical(self, "注册失败", str(e))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"注册失败: {str(e)}")

    def close_dialog(self):
        """关闭对话框"""
        self.timer.stop()
        self.cap.release()
        self.reject()

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


# ============================================================
# 批量导入对话框
# ============================================================

class BatchImportDialog(QDialog):
    """批量导入对话框 — 从文件夹批量导入人员"""
    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("📂 批量导入人员")
        self.setMinimumWidth(620)
        self.setMinimumHeight(520)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        self.import_dir = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # 说明
        info = QLabel(
            "📂 选择包含人员照片的文件夹\n\n"
            "目录结构要求:\n"
            "  选择的文件夹/\n"
            "  ├── 张三/\n"
            "  │   ├── 001.jpg\n"
            "  │   └── ...\n"
            "  └── 李四/\n"
            "      └── ..."
        )
        info.setStyleSheet(
            f"padding: 14px; background: {PRIMARY_LIGHT}; border-radius: 8px; "
            f"font-size: 12px; color: {TEXT_SECONDARY};"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # 选择文件夹
        folder_layout = QHBoxLayout()
        folder_layout.setSpacing(10)

        self.folder_label = QLabel("未选择文件夹")
        self.folder_label.setStyleSheet(
            f"padding: 10px 14px; background: {BG_MAIN}; border: 1px solid {BORDER}; "
            f"border-radius: 8px; color: {TEXT_MUTED};"
        )
        folder_layout.addWidget(self.folder_label, 1)

        browse_btn = QPushButton("📁 选择文件夹")
        browse_btn.setStyleSheet(btn_primary("min-height: 38px;"))
        browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_btn)

        layout.addLayout(folder_layout)

        # 预览列表
        preview_card = QGroupBox("预览人员")
        preview_layout = QVBoxLayout()

        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(3)
        self.preview_table.setHorizontalHeaderLabels(["姓名", "照片数量", "状态"])
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setShowGrid(False)
        preview_layout.addWidget(self.preview_table)

        preview_card.setLayout(preview_layout)
        layout.addWidget(preview_card)

        # 统计信息
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet(
            f"padding: 10px 14px; background: {PRIMARY_LIGHT}; border-radius: 8px; "
            f"font-size: 13px; color: {TEXT_SECONDARY};"
        )
        layout.addWidget(self.stats_label)

        layout.addStretch()

        # 底部按钮
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(12)

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(btn_outline("min-height: 38px;"))
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(cancel_btn)

        bottom_layout.addStretch()

        import_btn = QPushButton("✓ 开始导入")
        import_btn.setStyleSheet(btn_success("min-height: 42px; min-width: 140px; font-size: 15px;"))
        import_btn.clicked.connect(self.do_import)
        bottom_layout.addWidget(import_btn)

        layout.addLayout(bottom_layout)

    def browse_folder(self):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择人员照片文件夹")
        if folder:
            self.import_dir = folder
            self.folder_label.setText(folder)
            self.folder_label.setStyleSheet(
                f"padding: 10px 14px; background: {BG_MAIN}; border: 1px solid {SUCCESS}; "
                f"border-radius: 8px; color: {TEXT_PRIMARY};"
            )
            self.load_preview()

    def load_preview(self):
        """加载预览"""
        if not self.import_dir or not os.path.exists(self.import_dir):
            return

        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        persons = []

        for person_name in sorted(os.listdir(self.import_dir)):
            person_dir = os.path.join(self.import_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            img_count = len([f for f in os.listdir(person_dir)
                           if os.path.splitext(f)[1].lower() in valid_exts])
            persons.append({'name': person_name, 'count': img_count})

        self.preview_table.setRowCount(len(persons))
        total_images = 0
        for i, person in enumerate(persons):
            self.preview_table.setItem(i, 0, QTableWidgetItem(person['name']))
            count_item = QTableWidgetItem(str(person['count']))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.preview_table.setItem(i, 1, count_item)

            is_ready = person['count'] >= MIN_FACE_PHOTOS
            status = "✅ 就绪" if is_ready else "⚠️ 图片不足"
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            if not is_ready:
                status_item.setForeground(QColor(WARNING))
            else:
                status_item.setForeground(QColor(SUCCESS))
            self.preview_table.setItem(i, 2, status_item)

            # 设置行高
            self.preview_table.setRowHeight(i, 38)
            total_images += person['count']

        valid_count = len([p for p in persons if p['count'] >= MIN_FACE_PHOTOS])
        self.stats_label.setText(
            f"📊 共 <b>{len(persons)}</b> 人  |  "
            f"<b>{total_images}</b> 张照片  |  "
            f"<b style='color:{SUCCESS}'>{valid_count}</b> 人可导入"
        )

    def do_import(self):
        """执行导入"""
        if not self.import_dir:
            QMessageBox.warning(self, "提示", "请先选择文件夹！")
            return

        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        persons = []
        for person_name in sorted(os.listdir(self.import_dir)):
            person_dir = os.path.join(self.import_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            img_count = len([f for f in os.listdir(person_dir)
                           if os.path.splitext(f)[1].lower() in valid_exts])
            if img_count >= MIN_FACE_PHOTOS:
                persons.append({'name': person_name, 'dir': person_dir})

        if not persons:
            QMessageBox.warning(self, "提示", f"没有找到至少 {MIN_FACE_PHOTOS} 张照片的人员！")
            return

        reply = QMessageBox.question(
            self, "确认导入",
            f"将导入 {len(persons)} 个人员，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        success_count = 0
        fail_count = 0
        for person in persons:
            try:
                images = []
                for img_name in os.listdir(person['dir']):
                    if os.path.splitext(img_name)[1].lower() in valid_exts:
                        img_path = os.path.join(person['dir'], img_name)
                        img = imread_safe(img_path)
                        if img is not None:
                            images.append(img)

                if images:
                    self.user_manager.register_user(
                        name=person['name'],
                        images=images
                    )
                    success_count += 1
            except Exception as e:
                fail_count += 1

        QMessageBox.information(self, "导入完成",
            f"✅ 导入完成！\n\n成功: {success_count} 人\n失败: {fail_count} 人")
        self.accept()

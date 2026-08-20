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
                             QCheckBox, QFrame, QSpacerItem, QSizePolicy,
                             QProgressBar, QDialogButtonBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QIcon

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.database import Database
from common.user_manager import UserManager
from common.config import WORK_PHOTOS_DIR, FACES_DIR, MIN_FACE_PHOTOS, DATA_DIR
from admin.batch_register import scan_uploads, batch_register_async
from admin.generate_encodings import generate_encodings_async
from common.theme import (
    GLOBAL_STYLESHEET, btn_primary, btn_success, btn_danger, btn_warning,
    btn_info, btn_outline, CARD_STYLE,
    PRIMARY, PRIMARY_LIGHT, SUCCESS, DANGER, WARNING, BG_MAIN, BG_CARD,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, BORDER, DIVIDER,
    ADMIN_HEADER_BG, ADMIN_HEADER_TEXT, ADMIN_ACCENT, ADMIN_ACCENT_HOVER,
    ADMIN_TOOLBAR_BG, ADMIN_TOOLBAR_BORDER, ADMIN_ROW_HOVER, ADMIN_ROW_ALT,
    ADMIN_DETAIL_BG, ADMIN_STAT_BG,
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
# 样式常量
# ============================================================

ADMIN_BASE = f"""
    QMainWindow {{
        background-color: #f0f2f5;
    }}
    QWidget {{
        color: {TEXT_PRIMARY};
        font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
    }}
"""

HEADER_STYLE = f"""
    QFrame {{
        background-color: {ADMIN_HEADER_BG};
        border: none;
    }}
"""

TOOLBAR_STYLE = f"""
    QFrame {{
        background-color: {ADMIN_TOOLBAR_BG};
        border-bottom: 1px solid {ADMIN_TOOLBAR_BORDER};
    }}
"""

TOOLBAR_BTN = """
    QPushButton {
        background-color: transparent;
        color: #475569;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 500;
    }
    QPushButton:hover {
        background-color: #f1f5f9;
        border-color: #cbd5e1;
        color: #1e293b;
    }
    QPushButton:pressed {
        background-color: #e2e8f0;
    }
"""

TOOLBAR_BTN_PRIMARY = f"""
    QPushButton {{
        background-color: {ADMIN_ACCENT};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {ADMIN_ACCENT_HOVER};
    }}
    QPushButton:pressed {{
        background-color: #1d4ed8;
    }}
"""

TABLE_STYLE = f"""
    QTableWidget {{
        border: 1px solid {ADMIN_TOOLBAR_BORDER};
        border-radius: 10px;
        background-color: white;
        gridline-color: #f1f5f9;
        font-size: 13px;
        selection-background-color: #eff6ff;
        selection-color: {TEXT_PRIMARY};
        outline: none;
    }}
    QTableWidget::item {{
        padding: 10px 14px;
        border-bottom: 1px solid #f1f5f9;
    }}
    QTableWidget::item:selected {{
        background-color: #eff6ff;
        color: {PRIMARY};
    }}
    QTableWidget::item:hover {{
        background-color: {ADMIN_ROW_HOVER};
    }}
    QHeaderView::section {{
        background-color: #f8fafc;
        color: #64748b;
        font-weight: bold;
        font-size: 12px;
        padding: 10px 14px;
        border: none;
        border-bottom: 2px solid {ADMIN_TOOLBAR_BORDER};
        border-right: 1px solid #f1f5f9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    QHeaderView::section:last {{
        border-right: none;
    }}
    QTableCornerButton::section {{
        border: none;
        background-color: #f8fafc;
    }}
"""

SEARCH_INPUT = f"""
    QLineEdit {{
        border: 1px solid {ADMIN_TOOLBAR_BORDER};
        border-radius: 8px;
        background-color: white;
        padding: 9px 14px;
        font-size: 13px;
        color: {TEXT_PRIMARY};
        selection-background-color: #dbeafe;
    }}
    QLineEdit:focus {{
        border-color: {ADMIN_ACCENT};
    }}
    QLineEdit::placeholder {{
        color: #94a3b8;
    }}
"""

STATS_STYLE = f"""
    QFrame {{
        background-color: {ADMIN_STAT_BG};
        border: 1px solid {ADMIN_TOOLBAR_BORDER};
        border-radius: 8px;
    }}
    QLabel {{
        background: transparent;
        border: none;
    }}
"""


# ============================================================
# 辅助函数
# ============================================================

def create_separator():
    """创建水平分隔线"""
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet(f"background-color: {DIVIDER}; max-height: 1px; margin: 4px 0;")
    return sep


# ============================================================
# 进度对话框
# ============================================================

class ProgressDialog(QDialog):
    """长时间任务的进度对话框"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 状态文字
        self.status_label = QLabel("准备中...")
        self.status_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 6px;
                text-align: center;
                height: 22px;
                background: #f0f0f0;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #66BB6A);
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # 取消按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setStyleSheet(btn_danger("padding: 6px 20px;"))
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self._process = None
        self._queue = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)

    def start(self, process, queue):
        """启动进度监控"""
        self._process = process
        self._queue = queue
        self._timer.start(100)
        self.show()

    def _poll(self):
        """轮询子进程进度"""
        if self._queue is None:
            return

        while not self._queue.empty():
            msg = self._queue.get()
            msg_type = msg[0]

            if msg_type == 'progress':
                _, current, total, text = msg
                self.status_label.setText(text)
                if total > 0:
                    self.progress_bar.setValue(int(current / total * 100))

            elif msg_type == 'done':
                self._finish(msg[1])
                return

            elif msg_type == 'error':
                self._error(msg[1])
                return

        # 检查进程是否意外退出
        if self._process and not self._process.is_alive():
            self._finish(0)

    def _finish(self, count):
        """完成"""
        self._timer.stop()
        self.progress_bar.setValue(100)
        self.status_label.setText(f"完成！生成 {count} 个编码")
        self.cancel_btn.setText("关闭")
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.accept)

    def _error(self, msg):
        """错误"""
        self._timer.stop()
        self.status_label.setText(f"错误: {msg}")
        self.cancel_btn.setText("关闭")
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.reject)


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
        self.setWindowTitle("人脸识别人事管理系统")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(ADMIN_BASE)

        # ── 中央部件 ──
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ══════════════════════════════════════════
        # 顶部深色标题栏
        # ══════════════════════════════════════════
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(HEADER_STYLE)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        logo = QLabel("👤 人脸识别人事管理系统")
        logo.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {ADMIN_HEADER_TEXT}; "
            f"background: transparent;"
        )
        header_layout.addWidget(logo)

        header_layout.addStretch()

        # 右上角：统计概览
        self.header_stats = QLabel("")
        self.header_stats.setStyleSheet(
            f"font-size: 13px; color: #94a3b8; background: transparent;"
        )
        header_layout.addWidget(self.header_stats)

        root.addWidget(header)

        # ══════════════════════════════════════════
        # 工具栏（操作按钮 + 搜索）
        # ══════════════════════════════════════════
        toolbar = QFrame()
        toolbar.setFixedHeight(60)
        toolbar.setStyleSheet(TOOLBAR_STYLE)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(24, 10, 24, 10)
        toolbar_layout.setSpacing(8)

        # 左侧：操作按钮组
        # 主要操作（蓝色）
        add_btn = QPushButton("＋ 添加员工")
        add_btn.setStyleSheet(TOOLBAR_BTN_PRIMARY)
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.clicked.connect(self.show_add_user_dialog)
        toolbar_layout.addWidget(add_btn)

        quick_btn = QPushButton("⚡ 快速注册")
        quick_btn.setStyleSheet(TOOLBAR_BTN_PRIMARY)
        quick_btn.setCursor(Qt.PointingHandCursor)
        quick_btn.clicked.connect(self.show_quick_register_dialog)
        toolbar_layout.addWidget(quick_btn)

        toolbar_layout.addSpacing(4)

        # 次要操作
        batch_btn = QPushButton("📂 批量导入")
        batch_btn.setStyleSheet(TOOLBAR_BTN)
        batch_btn.setCursor(Qt.PointingHandCursor)
        batch_btn.clicked.connect(self.show_batch_import_dialog)
        toolbar_layout.addWidget(batch_btn)

        uploads_btn = QPushButton("📥 导入 uploads")
        uploads_btn.setStyleSheet(TOOLBAR_BTN)
        uploads_btn.setCursor(Qt.PointingHandCursor)
        uploads_btn.clicked.connect(self.import_from_uploads)
        toolbar_layout.addWidget(uploads_btn)

        gen_btn = QPushButton("🧠 生成编码")
        gen_btn.setStyleSheet(TOOLBAR_BTN)
        gen_btn.setCursor(Qt.PointingHandCursor)
        gen_btn.clicked.connect(self.generate_encodings)
        toolbar_layout.addWidget(gen_btn)

        toolbar_layout.addStretch()

        # 右侧：搜索框
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 搜索姓名 / 工号 / 部门...")
        self.search_input.setFixedWidth(280)
        self.search_input.setStyleSheet(SEARCH_INPUT)
        self.search_input.textChanged.connect(self.search_users)
        toolbar_layout.addWidget(self.search_input)

        root.addWidget(toolbar)

        # ══════════════════════════════════════════
        # 主内容区：表格（直接放入 root，不套 wrapper）
        # ══════════════════════════════════════════
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(6)
        self.user_table.setHorizontalHeaderLabels(["工号", "姓名", "部门", "电话", "人脸数", "状态"])
        self.user_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.user_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.user_table.setAlternatingRowColors(True)
        self.user_table.verticalHeader().setVisible(False)
        self.user_table.setShowGrid(False)
        self.user_table.setStyleSheet(TABLE_STYLE)
        self.user_table.setContentsMargins(24, 0, 24, 0)
        self.user_table.itemSelectionChanged.connect(self.on_user_select)
        # 列宽策略
        header_view = self.user_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 工号
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # 姓名
        header_view.setSectionResizeMode(2, QHeaderView.Stretch)           # 部门（拉伸填满）
        header_view.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 电话
        header_view.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 人脸数
        header_view.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # 状态
        root.addWidget(self.user_table, 1)  # stretch=1：占据所有剩余空间

        # ══════════════════════════════════════════
        # 底部：选中员工详情卡
        # ══════════════════════════════════════════
        self.detail_card = QFrame()
        self.detail_card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 1px solid {ADMIN_TOOLBAR_BORDER};
                border-radius: 12px;
            }}
        """)
        self.detail_card.setFixedHeight(120)
        detail_layout = QHBoxLayout(self.detail_card)
        detail_layout.setContentsMargins(20, 16, 20, 16)
        detail_layout.setSpacing(20)

        # 左侧：工作照片
        self.work_photo_preview = QLabel()
        self.work_photo_preview.setAlignment(Qt.AlignCenter)
        self.work_photo_preview.setFixedSize(88, 88)
        self.work_photo_preview.setStyleSheet(
            f"border: 2px dashed {BORDER}; background-color: #fafbfc; "
            f"color: {TEXT_MUTED}; font-size: 11px; border-radius: 10px;"
        )
        self.work_photo_preview.setText("暂无照片")
        detail_layout.addWidget(self.work_photo_preview)

        # 中间：用户信息
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        self.detail_name_label = QLabel("选择左侧员工查看详情")
        self.detail_name_label.setStyleSheet(
            f"font-size: 15px; font-weight: bold; color: {TEXT_PRIMARY}; background: transparent;"
        )
        info_layout.addWidget(self.detail_name_label)

        self.detail_info_label = QLabel("")
        self.detail_info_label.setStyleSheet(
            f"font-size: 12px; color: {TEXT_SECONDARY}; background: transparent; line-height: 1.5;"
        )
        info_layout.addWidget(self.detail_info_label)

        info_layout.addStretch()
        detail_layout.addLayout(info_layout, 1)

        # 右侧：操作按钮（2×2 网格）
        actions_layout = QGridLayout()
        actions_layout.setSpacing(6)
        actions_layout.setContentsMargins(0, 4, 0, 4)

        edit_btn = QPushButton("✏️ 编辑")
        edit_btn.setStyleSheet(TOOLBAR_BTN + "padding: 6px 10px; font-size: 12px;")
        edit_btn.setCursor(Qt.PointingHandCursor)
        edit_btn.clicked.connect(self.show_edit_user_dialog)
        actions_layout.addWidget(edit_btn, 0, 0)

        add_face_btn = QPushButton("📷 补充人脸")
        add_face_btn.setToolTip("为已有员工添加更多人脸照片，提升识别准确率")
        add_face_btn.setStyleSheet(TOOLBAR_BTN + "padding: 6px 10px; font-size: 12px;")
        add_face_btn.setCursor(Qt.PointingHandCursor)
        add_face_btn.clicked.connect(self.show_add_face_dialog)
        actions_layout.addWidget(add_face_btn, 0, 1)

        import_photo_btn = QPushButton("📂 导入照片")
        import_photo_btn.setToolTip("从文件夹批量导入照片到选中员工")
        import_photo_btn.setStyleSheet(TOOLBAR_BTN + "padding: 6px 10px; font-size: 12px;")
        import_photo_btn.setCursor(Qt.PointingHandCursor)
        import_photo_btn.clicked.connect(self.import_photos_from_folder)
        actions_layout.addWidget(import_photo_btn, 1, 0)

        disable_btn = QPushButton("🚫 停用")
        disable_btn.setToolTip("停用员工（可恢复）")
        disable_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {WARNING}; "
            f"border: 1px solid #fde68a; border-radius: 8px; padding: 6px 10px; "
            f"font-size: 12px; font-weight: 500; }}"
            f"QPushButton:hover {{ background-color: #fef3c7; border-color: {WARNING}; }}"
        )
        disable_btn.setCursor(Qt.PointingHandCursor)
        disable_btn.clicked.connect(self.disable_user)
        actions_layout.addWidget(disable_btn, 1, 1)

        actions_layout.setColumnStretch(0, 1)
        actions_layout.setColumnStretch(1, 1)

        detail_layout.addLayout(actions_layout)

        root.addWidget(self.detail_card)

        # ══════════════════════════════════════════
        # 底部统计条
        # ══════════════════════════════════════════
        stats_frame = QFrame()
        stats_frame.setFixedHeight(40)
        stats_frame.setStyleSheet(STATS_STYLE)
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(24, 0, 24, 0)

        self.stats_label = QLabel()
        self.stats_label.setStyleSheet(
            f"font-size: 12px; color: {TEXT_SECONDARY}; background: transparent;"
        )
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()

        # 批量停用按钮
        batch_disable_btn = QPushButton("🚫 批量停用")
        batch_disable_btn.setToolTip("批量停用选中的员工（按住Ctrl/Shift可多选）")
        batch_disable_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {WARNING}; "
            f"border: 1px solid #fde68a; border-radius: 6px; padding: 5px 12px; "
            f"font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #fef3c7; border-color: {WARNING}; }}"
        )
        batch_disable_btn.setCursor(Qt.PointingHandCursor)
        batch_disable_btn.clicked.connect(self.batch_disable_users)
        stats_layout.addWidget(batch_disable_btn)

        # 彻底删除按钮（红色警告）
        hard_del_btn = QPushButton("⚠ 彻底删除")
        hard_del_btn.setToolTip("彻底删除选中的员工（不可恢复！按住Ctrl/Shift可多选）")
        hard_del_btn.setStyleSheet(
            f"QPushButton {{ background-color: {DANGER}; color: white; "
            f"border: none; border-radius: 6px; padding: 5px 12px; "
            f"font-size: 12px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: #d33426; }}"
        )
        hard_del_btn.setCursor(Qt.PointingHandCursor)
        hard_del_btn.clicked.connect(self.hard_delete_users)
        stats_layout.addWidget(hard_del_btn)

        root.addWidget(stats_frame)

        # ── 状态栏 ──
        self.statusBar().showMessage("✅ 系统就绪")
        self.statusBar().setStyleSheet(
            f"background-color: white; color: {TEXT_SECONDARY}; "
            f"border-top: 1px solid {ADMIN_TOOLBAR_BORDER}; padding: 6px 16px; font-size: 12px;"
        )

    # ── 表格填充 ────────────────────────────────────────────

    def _populate_table(self, users):
        """填充用户表格"""
        self.user_table.setRowCount(len(users))
        for i, user in enumerate(users):
            face_count = self.db.get_user_face_count(user["id"])
            is_active = user["is_active"]

            self.user_table.setItem(i, 0, QTableWidgetItem(user.get("employee_id") or "—"))
            self.user_table.setItem(i, 1, QTableWidgetItem(user["name"]))
            self.user_table.setItem(i, 2, QTableWidgetItem(user.get("department") or "—"))
            self.user_table.setItem(i, 3, QTableWidgetItem(user.get("phone") or "—"))

            # 人脸数（带颜色）
            face_item = QTableWidgetItem(str(face_count))
            face_item.setTextAlignment(Qt.AlignCenter)
            if face_count < MIN_FACE_PHOTOS:
                face_item.setForeground(QColor(WARNING))
            else:
                face_item.setForeground(QColor(SUCCESS))
            self.user_table.setItem(i, 4, face_item)

            # 状态
            status = "✅ 启用" if is_active else "❌ 禁用"
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            if not is_active:
                status_item.setForeground(QColor(DANGER))
            self.user_table.setItem(i, 5, status_item)

            # 将用户ID存储到工号列的UserRole中
            self.user_table.item(i, 0).setData(Qt.UserRole, user["id"])

        # 设置行高
        for row in range(self.user_table.rowCount()):
            self.user_table.setRowHeight(row, 44)

    def _get_selected_user_id(self):
        """获取当前选中的用户ID"""
        selected_rows = self.user_table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        row = selected_rows[0].row()
        return self.user_table.item(row, 0).data(Qt.UserRole)

    # ── 数据加载 ────────────────────────────────────────────

    def load_users(self):
        """加载用户列表"""
        users = self.user_manager.get_all_users()
        self._populate_table(users)

        # 更新统计
        stats = self.db.get_statistics()
        total = stats['total_users']
        active = stats['active_users']
        faces = stats['total_faces']

        self.stats_label.setText(
            f"📊 总员工  {total}   |   启用  {active}   |   人脸照片  {faces} 张"
        )
        self.header_stats.setText(
            f"员工 {total}  ·  启用 {active}  ·  人脸编码 {faces}"
        )

    def search_users(self):
        """搜索用户"""
        keyword = self.search_input.text().strip()
        if keyword:
            users = self.user_manager.search_users(keyword)
        else:
            users = self.user_manager.get_all_users()
        self._populate_table(users)

    # ── 用户选中 → 底部详情卡更新 ────────────────────────────

    def on_user_select(self):
        """用户选中事件"""
        user_id = self._get_selected_user_id()
        if user_id is None:
            self.detail_name_label.setText("选择左侧员工查看详情")
            self.detail_info_label.setText("")
            self.work_photo_preview.setText("暂无照片")
            self.work_photo_preview.setStyleSheet(
                f"border: 2px dashed {BORDER}; background-color: #fafbfc; "
                f"color: {TEXT_MUTED}; font-size: 12px; border-radius: 10px;"
            )
            self.work_photo_preview.setPixmap(QPixmap())
            return

        user = self.user_manager.get_user(user_id)
        if not user:
            return

        face_count = self.db.get_user_face_count(user_id)

        # 名字
        self.detail_name_label.setText(f"👤 {user['name']}")

        # 信息文字
        emp_id = user.get('employee_id') or '—'
        dept = user.get('department') or '—'
        phone = user.get('phone') or '—'
        created = user.get('created_at', '—')

        face_color = SUCCESS if face_count >= MIN_FACE_PHOTOS else WARNING
        self.detail_info_label.setText(
            f"工号: {emp_id}　　部门: {dept}　　电话: {phone}\n"
            f"人脸照片: <span style='color:{face_color}; font-weight:bold;'>{face_count} 张</span>"
            f"　　注册时间: {created}"
        )

        # 加载工作照片
        self._load_work_photo(user)

    def _set_work_photo_placeholder(self):
        """设置工作照片预览区域的占位提示"""
        self.work_photo_preview.setText("暂无照片")
        self.work_photo_preview.setStyleSheet(
            f"border: 2px dashed {BORDER}; background-color: #fafbfc; "
            f"color: {TEXT_MUTED}; font-size: 12px; border-radius: 10px;"
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
                        f"border: 2px solid {SUCCESS}; background-color: #fff; border-radius: 10px;"
                    )
                    scaled_pixmap = pixmap.scaled(
                        self.work_photo_preview.size(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.work_photo_preview.setPixmap(scaled_pixmap)
                    return
        self._set_work_photo_placeholder()

    # ── 照片操作 ────────────────────────────────────────────

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

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = _safe_face_locations(rgb_image, "hog")
        if len(face_locations) == 0:
            QMessageBox.warning(self, "提示", "照片中未检测到人脸，请选择包含人脸的照片！")
            return

        os.makedirs(WORK_PHOTOS_DIR, exist_ok=True)
        save_path = os.path.join(WORK_PHOTOS_DIR, f"{user_id}.jpg")
        cv2.imwrite(save_path, image)

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

    # ── 对话框入口 ────────────────────────────────────────────

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

    # ── 导入操作 ────────────────────────────────────────────

    def import_from_uploads(self):
        """一键从 data/uploads 目录导入人员"""
        try:
            self._do_import_from_uploads()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败：{str(e)}")

    def generate_encodings(self):
        """为所有用户生成人脸编码（可选择模型）"""
        from admin.generate_encodings import get_last_model

        # 获取上次使用的模型
        last_model = get_last_model()

        # 模型选择对话框
        from PyQt5.QtWidgets import QInputDialog
        model, ok = QInputDialog.getItem(
            self, "选择编码模型",
            "选择用于生成人脸编码的模型：\n\n"
            "• face_recognition: 基于 dlib，128维，推荐用于门禁验证\n"
            "• 训练模型: MobileFaceNet，256维，用于分类任务",
            ["face_recognition", "trained"],
            0 if last_model == "face_recognition" else 1,
            False
        )

        if not ok:
            return

        # 检查是否切换了模型
        force = (model != last_model)

        if force:
            reply = QMessageBox.question(
                self, "模型切换",
                f"检测到模型切换：{last_model} → {model}\n\n"
                "切换模型将清除所有旧编码并重新生成。\n"
                "新编码将与旧编码维度不同，门禁系统将使用新模型。\n\n"
                "是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
        else:
            reply = QMessageBox.question(
                self, "确认",
                f"将使用 {model} 为所有用户生成人脸编码。\n\n"
                "需要几秒到几分钟，取决于照片数量。\n\n"
                "是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

        if reply != QMessageBox.Yes:
            return

        proc, queue = generate_encodings_async(model=model, force=force)
        dialog = ProgressDialog("生成人脸编码", self)
        dialog.start(proc, queue)
        dialog.exec_()
        self.load_users()

    def _do_import_from_uploads(self):
        """一键从 data/uploads 目录导入人员"""
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

        proc, queue, _ = batch_register_async(uploads_dir, None)
        self._import_proc = proc
        self._import_queue = queue
        self._import_timer.start(200)

    def _poll_import_result(self):
        """轮询子进程结果"""
        if self._import_queue is None or not self._import_queue.empty():
            status, data = self._import_queue.get()
            self._import_timer.stop()
            self._import_proc = None
            self._import_queue = None

            if status == 'error':
                QMessageBox.critical(self, "导入失败", f"子进程错误：{data}")
            else:
                result = data
                new_count = result.get('success', 0) + result.get('reactivate', 0)
                msg = f"导入完成！\n\n新增: {result['success']} 人"
                if result.get('reactivate', 0) > 0:
                    msg += f"\n恢复（之前删除的）: {result['reactivate']} 人"
                if result['skip'] > 0:
                    msg += f"\n跳过（已存在）: {result['skip']} 人"
                if result['errors']:
                    msg += f"\n失败: {result['fail']} 人\n\n" + "\n".join(result['errors'][:5])
                QMessageBox.information(self, "导入结果", msg)
                self.load_users()

                if new_count > 0:
                    from admin.generate_encodings import get_last_model
                    last_model = get_last_model()
                    proc, queue = generate_encodings_async(model=last_model, force=True)
                    dialog = ProgressDialog("生成人脸编码", self)
                    dialog.start(proc, queue)
                    dialog.exec_()
                    self.load_users()
            return

        if self._import_proc and not self._import_proc.is_alive():
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

    # ── 删除操作 ────────────────────────────────────────────

    def disable_user(self):
        """停用选中的用户（软删除，可恢复）"""
        user_id = self._get_selected_user_id()
        if user_id is None:
            QMessageBox.warning(self, "提示", "请先选择要停用的员工！")
            return

        user = self.user_manager.get_user(user_id)
        user_name = user["name"] if user else ""

        reply = QMessageBox.question(
            self, "确认停用",
            f"确定要停用员工 '{user_name}' 吗？\n\n"
            "停用后该员工将无法通过门禁识别，但数据会保留。\n"
            "可以通过「添加员工」功能重新激活。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            work_photo = user.get('work_photo') if user else None
            if work_photo and os.path.exists(work_photo):
                os.remove(work_photo)

            if self.user_manager.delete_user(user_id):
                QMessageBox.information(self, "成功", f"员工 '{user_name}' 已停用")
                self.load_users()
            else:
                QMessageBox.critical(self, "错误", "停用员工失败")

    def batch_disable_users(self):
        """批量停用选中的员工"""
        selected_rows = self.user_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "提示", "请先选择要停用的员工！\n（按住 Ctrl 或 Shift 可多选）")
            return

        users_to_disable = []
        for index in selected_rows:
            row = index.row()
            user_id = self.user_table.item(row, 0).data(Qt.UserRole)
            name = self.user_table.item(row, 1).text()
            emp_id = self.user_table.item(row, 0).text()
            users_to_disable.append({'id': user_id, 'name': name, 'emp_id': emp_id})

        user_list = "\n".join([f"  • {u['name']}（工号: {u['emp_id']}）" for u in users_to_disable])
        reply = QMessageBox.question(
            self, "确认批量停用",
            f"确定要停用以下 {len(users_to_disable)} 名员工吗？\n\n"
            f"{user_list}\n\n"
            "停用后员工将无法通过门禁识别，但数据会保留。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        success_count = 0
        fail_count = 0
        for u in users_to_disable:
            try:
                work_photo = None
                user = self.user_manager.get_user(u['id'])
                if user:
                    work_photo = user.get('work_photo')
                if work_photo and os.path.exists(work_photo):
                    os.remove(work_photo)

                if self.user_manager.delete_user(u['id']):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception:
                fail_count += 1

        msg = f"批量停用完成！\n\n✅ 成功: {success_count} 人"
        if fail_count > 0:
            msg += f"\n❌ 失败: {fail_count} 人"
        QMessageBox.information(self, "停用结果", msg)
        self.load_users()

    def hard_delete_users(self):
        """彻底删除选中的员工（不可恢复！）"""
        selected_rows = self.user_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "提示", "请先选择要彻底删除的员工！\n（按住 Ctrl 或 Shift 可多选）")
            return

        users_to_delete = []
        for index in selected_rows:
            row = index.row()
            user_id = self.user_table.item(row, 0).data(Qt.UserRole)
            name = self.user_table.item(row, 1).text()
            emp_id = self.user_table.item(row, 0).text()
            users_to_delete.append({'id': user_id, 'name': name, 'emp_id': emp_id})

        user_list = "\n".join([f"  • {u['name']}（工号: {u['emp_id']}）" for u in users_to_delete])
        reply = QMessageBox.warning(
            self, "⚠️ 彻底删除确认",
            f"确定要【彻底删除】以下 {len(users_to_delete)} 名员工吗？\n\n"
            f"{user_list}\n\n"
            "⚠️ 此操作不可恢复！\n"
            "• 所有人脸照片将被永久删除\n"
            "• 数据库记录将被彻底移除\n"
            "• 通行日志中用户名将标记为「已删除」",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 二次确认
        reply2 = QMessageBox.warning(
            self, "最终确认",
            "再次确认：此操作不可撤销！\n\n点击「确定」执行彻底删除。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply2 != QMessageBox.Yes:
            return

        success_count = 0
        fail_count = 0
        for u in users_to_delete:
            try:
                work_photo = None
                user = self.user_manager.get_user(u['id'])
                if user:
                    work_photo = user.get('work_photo')
                if work_photo and os.path.exists(work_photo):
                    os.remove(work_photo)

                if self.user_manager.hard_delete_user(u['id']):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception:
                fail_count += 1

        msg = f"彻底删除完成！\n\n✅ 成功: {success_count} 人"
        if fail_count > 0:
            msg += f"\n❌ 失败: {fail_count} 人"
        QMessageBox.information(self, "删除结果", msg)
        self.load_users()

    def import_photos_from_folder(self):
        """从文件夹批量导入选中员工的照片"""
        user_id = self._get_selected_user_id()
        if user_id is None:
            QMessageBox.warning(self, "提示", "请先选择一个员工！")
            return

        user = self.user_manager.get_user(user_id)
        if not user:
            QMessageBox.warning(self, "提示", "用户不存在！")
            return

        folder_path = QFileDialog.getExistingDirectory(
            self, f"选择 {user['name']} 的照片文件夹", "",
            QFileDialog.ShowDirsOnly
        )
        if not folder_path:
            return

        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in valid_exts]

        if not image_files:
            QMessageBox.warning(self, "提示",
                f"所选文件夹中没有找到图片文件！\n\n"
                f"支持的格式: {', '.join(valid_exts)}")
            return

        current_count = self.db.get_user_face_count(user_id)
        reply = QMessageBox.question(
            self, "确认导入",
            f"为员工 '{user['name']}' 导入照片\n\n"
            f"📁 文件夹: {folder_path}\n"
            f"📷 找到图片: {len(image_files)} 张\n"
            f"📋 当前已有: {current_count} 张\n\n"
            f"是否导入？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        try:
            result = self.user_manager.import_photos_from_folder(user_id, folder_path)
            new_count = self.db.get_user_face_count(user_id)

            msg = f"导入完成！\n\n✅ 成功导入: {result['success']} 张\n📋 当前总计: {new_count} 张"
            if result['errors']:
                msg += f"\n⚠️ 失败: {len(result['errors'])} 张"
                if len(result['errors']) <= 3:
                    msg += "\n" + "\n".join(result['errors'])
            msg += "\n\n💡 请点击「🧠 生成编码」生成人脸编码"
            QMessageBox.information(self, "导入结果", msg)
            self.load_users()
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))


# ============================================================
# 添加用户对话框
# ============================================================

class AddUserDialog(QDialog):
    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("添加新员工")
        self.setMinimumWidth(420)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("添加新员工")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {PRIMARY};")
        layout.addWidget(title)

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
    def __init__(self, user_manager: UserManager, user: dict, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.user = user
        self.setWindowTitle("编辑员工信息")
        self.setMinimumWidth(420)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel(f"编辑 — {self.user.get('name', '')}")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {PRIMARY};")
        layout.addWidget(title)

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
                name=name, employee_id=employee_id,
                department=department, phone=phone
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
        self.setWindowTitle(f"补充人脸照片 — {user_name}")
        self.setMinimumWidth(620)
        self.setMinimumHeight(520)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self.current_face_count = user_manager.db.get_user_face_count(user_id)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.captured_images = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(30)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

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

        self.photo_list = QListWidget()
        self.photo_list.setMaximumHeight(80)
        layout.addWidget(self.photo_list)

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
        self.captured_images = []
        self.photo_list.clear()
        self.status_label.setText("已拍摄: 0 张")
        self.status_label.setStyleSheet(f"font-size: 14px; padding: 8px; color: {TEXT_SECONDARY};")

    def save_faces(self):
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
        self.timer.stop()
        self.cap.release()
        self.reject()

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()


# ============================================================
# 快速注册对话框
# ============================================================

class QuickRegisterDialog(QDialog):
    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("快速注册员工")
        self.setMinimumWidth(720)
        self.setMinimumHeight(620)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.captured_images = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(30)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        tip = QLabel("💡 填写信息 → 对准摄像头拍照（至少5张）→ 点击注册")
        tip.setStyleSheet(
            f"font-size: 13px; color: {TEXT_SECONDARY}; padding: 10px; "
            f"background: {PRIMARY_LIGHT}; border-radius: 8px;"
        )
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)

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
        self.captured_images = []
        self.status_label.setText(f"已拍摄: 0 张（最少需要 {MIN_FACE_PHOTOS} 张）")
        self.status_label.setStyleSheet(f"font-size: 13px; padding: 8px; color: {TEXT_SECONDARY};")

    def register_user(self):
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
                name=name, images=self.captured_images,
                employee_id=employee_id, department=department, phone=phone
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
        self.timer.stop()
        self.cap.release()
        self.reject()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


# ============================================================
# 批量导入对话框
# ============================================================

class BatchImportDialog(QDialog):
    def __init__(self, user_manager: UserManager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("批量导入人员")
        self.setMinimumWidth(620)
        self.setMinimumHeight(520)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        self.import_dir = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

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

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet(
            f"padding: 10px 14px; background: {PRIMARY_LIGHT}; border-radius: 8px; "
            f"font-size: 13px; color: {TEXT_SECONDARY};"
        )
        layout.addWidget(self.stats_label)

        layout.addStretch()

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

            self.preview_table.setRowHeight(i, 38)
            total_images += person['count']

        valid_count = len([p for p in persons if p['count'] >= MIN_FACE_PHOTOS])
        self.stats_label.setText(
            f"📊 共 <b>{len(persons)}</b> 人  |  "
            f"<b>{total_images}</b> 张照片  |  "
            f"<b style='color:{SUCCESS}'>{valid_count}</b> 人可导入"
        )

    def do_import(self):
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
                        name=person['name'], images=images
                    )
                    success_count += 1
            except Exception as e:
                fail_count += 1

        QMessageBox.information(self, "导入完成",
            f"✅ 导入完成！\n\n成功: {success_count} 人\n失败: {fail_count} 人")
        self.accept()

"""
人事管理窗口模块
提供员工管理和人脸数据录入功能
"""

import cv2
import os
import sys
import shutil
import subprocess

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from PyQt5.QtWidgets import QStyledItemDelegate, QStyle
from PyQt5.QtCore import QModelIndex, Q_ARG


class DownloadManager:
    """独立的下载管理器（不依赖任何 UI 组件）"""

    BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    BUFFALO_L_DIR = os.path.join(ROOT_DIR, "deploy", "models", "buffalo_l")

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        self.downloading = False
        self.paused = False
        self.progress = 0        # 0-100, -1 = indeterminate
        self.detail = ""
        self.error = ""
        self.finished = False    # 本次下载已结束
        self._thread = None
        self._callbacks = []     # 状态变更回调

    def add_callback(self, fn):
        self._callbacks.append(fn)

    def remove_callback(self, fn):
        if fn in self._callbacks:
            self._callbacks.remove(fn)

    def _notify(self):
        for fn in self._callbacks:
            try:
                fn()
            except Exception:
                pass

    def start(self):
        """开始下载"""
        if self.downloading:
            return
        self.downloading = True
        self.paused = False
        self.finished = False
        self.error = ""
        self.progress = 0
        self.detail = "准备下载..."
        self._notify()

        import threading
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self):
        if self.downloading:
            self.paused = True
            self.detail = "已暂停"
            self._notify()

    def resume(self):
        if self.downloading and self.paused:
            self.paused = False
            self.detail = "继续下载..."
            self._notify()

    def cancel(self):
        if self.downloading:
            self.paused = False
            self.detail = "正在取消..."
            self._notify()
            # 设标志让线程退出
            self._cancel_flag = True

    def _run(self):
        """后台线程：下载 + 解压"""
        import zipfile
        import urllib.request
        import time

        self._cancel_flag = False
        zip_dir = os.path.dirname(self.BUFFALO_L_DIR)
        zip_path = os.path.join(zip_dir, "buffalo_l.zip")
        os.makedirs(zip_dir, exist_ok=True)

        try:
            # 获取文件大小
            req = urllib.request.Request(self.BUFFALO_L_URL, method='HEAD')
            with urllib.request.urlopen(req, timeout=15) as resp:
                total_size = int(resp.headers.get('Content-Length', 0))

            existing_size = 0
            if os.path.exists(zip_path):
                existing_size = os.path.getsize(zip_path)

            downloaded = existing_size
            start_time = time.time()

            req = urllib.request.Request(self.BUFFALO_L_URL)
            if existing_size > 0:
                req.add_header('Range', f'bytes={existing_size}-')
                mode = 'ab'
            else:
                mode = 'wb'

            with urllib.request.urlopen(req, timeout=30) as resp:
                if existing_size > 0 and resp.status == 200:
                    downloaded = 0
                    mode = 'wb'

                block_size = 1024 * 64
                with open(zip_path, mode) as f:
                    while True:
                        if self._cancel_flag:
                            # 取消：删除已下载文件
                            f.close()
                            if os.path.exists(zip_path):
                                os.remove(zip_path)
                            self._finish(False, "已取消")
                            return

                        while self.paused:
                            time.sleep(0.3)
                            if self._cancel_flag:
                                f.close()
                                if os.path.exists(zip_path):
                                    os.remove(zip_path)
                                self._finish(False, "已取消")
                                return

                        chunk = resp.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        if total_size > 0:
                            pct = min(99, int(downloaded / total_size * 100))
                            speed_mb = speed / (1024 * 1024)
                            remain = (total_size - downloaded) / speed if speed > 0 else 0
                            self.progress = pct
                            self.detail = f"下载中... {pct}%  {speed_mb:.1f}MB/s  剩余 {int(remain)}s"
                        else:
                            self.detail = f"下载中... {downloaded // (1024*1024)}MB"
                        self._notify()

            # 解压
            self.progress = 95
            self.detail = "解压中..."
            self._notify()
            os.makedirs(self.BUFFALO_L_DIR, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    parts = member.split('/')
                    if len(parts) >= 2 and parts[0] == 'buffalo_l' and parts[-1]:
                        target = os.path.join(self.BUFFALO_L_DIR, parts[-1])
                        with zf.open(member) as src, open(target, 'wb') as dst:
                            dst.write(src.read())

            if os.path.exists(zip_path):
                os.remove(zip_path)

            self._finish(True, "")

        except Exception as e:
            self._finish(False, str(e))

    def _finish(self, success, error_msg):
        self.downloading = False
        self.paused = False
        self.finished = True
        self.error = error_msg
        if success:
            self.progress = 100
            self.detail = "下载完成"
        elif error_msg == "已取消":
            self.progress = 0
            self.detail = "下载已取消"
        else:
            self.detail = f"下载失败: {error_msg}"
        self._notify()


class RowHighlightDelegate(QStyledItemDelegate):
    """自定义 Delegate：整行 hover/选中时无缝高亮"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover_row = -1

    def paint(self, painter, option, index):
        from PyQt5.QtGui import QColor, QPen, QFont

        row = index.row()
        is_hover = (row == self._hover_row)
        is_selected = option.state & QStyle.State_Selected

        # 1. 画背景（整行无缝）
        if is_selected:
            painter.fillRect(option.rect, QColor("#e8f0fe"))
        elif is_hover:
            painter.fillRect(option.rect, QColor("#eef1f6"))
        else:
            painter.fillRect(option.rect, QColor("white"))

        # 2. 画文字（不用 super().paint，避免默认格子高亮）
        painter.save()
        painter.setClipRect(option.rect)
        text = index.data(Qt.DisplayRole) or ""
        font = index.data(Qt.FontRole)
        if font:
            painter.setFont(font)
        # 对齐方式
        align = option.displayAlignment if option.displayAlignment else Qt.AlignLeft | Qt.AlignVCenter
        # 文字颜色
        text_color = index.data(Qt.ForegroundRole)
        if text_color:
            painter.setPen(QPen(text_color.color() if hasattr(text_color, 'color') else QColor(text_color)))
        elif is_selected:
            painter.setPen(QPen(QColor("#1a73e8")))
        else:
            painter.setPen(QPen(QColor("#333333")))

        text_rect = option.rect.adjusted(10, 0, -10, 0)
        painter.drawText(text_rect, align, str(text))
        painter.restore()

    def setHoverRow(self, row):
        self._hover_row = row


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
        padding: 8px 14px;
        font-size: 13px;
        font-weight: 500;
        font-family: 'Microsoft YaHei', 'Segoe UI Emoji', sans-serif;
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
        padding: 8px 14px;
        font-size: 13px;
        font-weight: bold;
        font-family: 'Microsoft YaHei', 'Segoe UI Emoji', sans-serif;
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

        # 缩放比例基准：默认窗口宽度 1400，基准字体 13px
        self._base_width = 1400
        self._base_font = 13
        self._last_scale = 1.0

    def eventFilter(self, obj, event):
        """追踪表格 hover 行号 + 下载标签点击"""
        from PyQt5.QtCore import QEvent
        # 下载状态标签点击 → 打开模型管理
        if obj == self._dl_label and event.type() == QEvent.MouseButtonRelease:
            self.show_model_manage_dialog()
            return True
        if hasattr(self, 'user_table') and obj == self.user_table.viewport() and hasattr(self, '_row_delegate'):
            from PyQt5.QtCore import QEvent
            if event.type() == QEvent.MouseMove:
                pos = event.pos()
                index = self.user_table.indexAt(pos)
                row = index.row() if index.isValid() else -1
                if self._row_delegate._hover_row != row:
                    # 刷新旧行和新行
                    old_row = self._row_delegate._hover_row
                    self._row_delegate.setHoverRow(row)
                    if old_row >= 0:
                        self.user_table.viewport().update(
                            self.user_table.visualItemRect(self.user_table.item(old_row, 0)))
                    if row >= 0:
                        self.user_table.viewport().update(
                            self.user_table.visualItemRect(self.user_table.item(row, 0)))
            elif event.type() == QEvent.Leave:
                old_row = self._row_delegate._hover_row
                if old_row >= 0:
                    self._row_delegate.setHoverRow(-1)
                    self.user_table.viewport().update(
                        self.user_table.visualItemRect(self.user_table.item(old_row, 0)))
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        """窗口大小变化时动态缩放所有 UI 元素"""
        super().resizeEvent(event)
        w = event.size().width()
        s = max(0.65, min(1.8, w / self._base_width))

        # 变化太小时跳过，避免频繁刷样式
        if abs(s - self._last_scale) < 0.03:
            return
        self._last_scale = s

        # ── 1. 全局字体缩放（让 QLabel/QTableWidgetItem/QPushButton 文字自动缩放）──
        from PyQt5.QtWidgets import QApplication
        font = QApplication.font()
        font.setPixelSize(int(self._base_font * s))
        QApplication.setFont(font)

        # ── 2. 标题栏高度 ──
        layout = self.centralWidget().layout()
        header = layout.itemAt(0).widget()
        if header:
            header.setFixedHeight(int(52 * s))

        # ── 3. 工具栏高度 + 按钮样式 ──
        toolbar = layout.itemAt(1).widget()
        if toolbar:
            toolbar.setFixedHeight(int(56 * s))
            pad_v = int(8 * s)
            pad_h = int(14 * s)
            radius = int(8 * s)
            fs = int(13 * s)
            primary_names = {"+ 添加员工", "快速注册"}
            for btn in toolbar.findChildren(QPushButton):
                is_primary = btn.text() in primary_names
                if is_primary:
                    btn.setStyleSheet(
                        f"QPushButton {{ background-color: {ADMIN_ACCENT}; color: white; "
                        f"border: none; border-radius: {radius}px; "
                        f"padding: {pad_v}px {pad_h}px; font-size: {fs}px; font-weight: bold; }}"
                        f"QPushButton:hover {{ background-color: {ADMIN_ACCENT_HOVER}; }}"
                        f"QPushButton:pressed {{ background-color: #1d4ed8; }}"
                    )
                else:
                    btn.setStyleSheet(
                        f"QPushButton {{ background-color: transparent; color: #475569; "
                        f"border: 1px solid #e2e8f0; border-radius: {radius}px; "
                        f"padding: {pad_v}px {pad_h}px; font-size: {fs}px; font-weight: 500; }}"
                        f"QPushButton:hover {{ background-color: #f1f5f9; border-color: #cbd5e1; color: #1e293b; }}"
                        f"QPushButton:pressed {{ background-color: #e2e8f0; }}"
                    )

        # ── 4. 表格行高 + 表头/内容字体 ──
        table = layout.itemAt(2).widget()
        if table and hasattr(table, 'setRowHeight'):
            row_h = int(44 * s)
            for row in range(table.rowCount()):
                table.setRowHeight(row, row_h)
            # 动态更新表格样式（整行 hover / 选中高亮）
            table.setStyleSheet(f"""
                QTableWidget {{
                    border: 1px solid {ADMIN_TOOLBAR_BORDER};
                    border-radius: 10px;
                    background-color: white;
                    gridline-color: transparent;
                    font-size: {int(13*s)}px;
                    selection-background-color: #e8f0fe;
                    selection-color: {PRIMARY};
                    outline: none;
                }}
                QTableWidget::item {{
                    padding: {int(10*s)}px {int(14*s)}px;
                    border: none;
                }}
                QTableWidget::item:selected {{
                    background-color: #e8f0fe;
                    color: {PRIMARY};
                }}
                QTableWidget::item:hover {{
                    background-color: #eef1f6;
                }}
                QHeaderView::section {{
                    background-color: #f8fafc;
                    color: #64748b;
                    font-weight: bold;
                    font-size: {int(12*s)}px;
                    padding: {int(10*s)}px {int(14*s)}px;
                    border: none;
                    border-bottom: 2px solid {ADMIN_TOOLBAR_BORDER};
                    border-right: 1px solid transparent;
                }}
                QHeaderView::section:last {{
                    border-right: none;
                }}
                QTableCornerButton::section {{
                    border: none;
                    background-color: #f8fafc;
                }}
            """)
            table.verticalHeader().setDefaultSectionSize(int(44 * s))
            table.verticalHeader().setMinimumSectionSize(int(36 * s))

        # ── 5. 详情卡高度 + 内部元素 ──
        detail = layout.itemAt(3).widget()
        if detail:
            detail.setFixedHeight(int(110 * s))
            detail.setContentsMargins(int(16*s), int(10*s), int(16*s), int(10*s))
            # 工作照片预览区缩放
            photo_size = int(80 * s)
            for lbl in detail.findChildren(QLabel):
                if lbl.text() == "暂无照片" or (lbl.pixmap() and not lbl.pixmap().isNull()):
                    lbl.setFixedSize(photo_size, photo_size)
                    lbl.setStyleSheet(
                        f"border: 2px dashed {BORDER}; background-color: #fafbfc; "
                        f"color: {TEXT_MUTED}; font-size: {int(10*s)}px; border-radius: {int(8*s)}px;"
                    )
                elif "选择" in lbl.text() or lbl == self.detail_name_label:
                    lbl.setStyleSheet(
                        f"font-size: {int(15*s)}px; font-weight: bold; "
                        f"color: {TEXT_PRIMARY}; background: transparent;"
                    )
                elif lbl == self.detail_info_label:
                    lbl.setStyleSheet(
                        f"font-size: {int(12*s)}px; color: {TEXT_SECONDARY}; "
                        f"background: transparent;"
                    )
            # 按钮缩放
            for btn in detail.findChildren(QPushButton):
                btn.setStyleSheet(
                    f"QPushButton {{ background-color: transparent; color: #475569; "
                    f"border: 1px solid #e2e8f0; border-radius: {int(8*s)}px; "
                    f"padding: {int(5*s)}px {int(8*s)}px; font-size: {int(12*s)}px; "
                    f"min-width: {int(65*s)}px; }}"
                    f"QPushButton:hover {{ background-color: #f1f5f9; border-color: #cbd5e1; color: #1e293b; }}"
                )

        # ── 6. 统计条高度 + 按钮缩放 ──
        stats = layout.itemAt(4).widget()
        if stats:
            stats.setFixedHeight(int(38 * s))
            stats.setContentsMargins(int(16*s), 0, int(16*s), 0)
            for lbl in stats.findChildren(QLabel):
                lbl.setStyleSheet(
                    f"font-size: {int(12*s)}px; color: {TEXT_SECONDARY}; background: transparent;"
                )
            for btn in stats.findChildren(QPushButton):
                btn.setStyleSheet(
                    f"QPushButton {{ background-color: transparent; color: {WARNING}; "
                    f"border: 1px solid #fde68a; border-radius: {int(6*s)}px; "
                    f"padding: {int(5*s)}px {int(12*s)}px; font-size: {int(12*s)}px; }}"
                    f"QPushButton:hover {{ background-color: #fef3c7; border-color: {WARNING}; }}"
                )
            # 彻底删除按钮（红色）
            for btn in stats.findChildren(QPushButton):
                if btn.text() == "彻底删除":
                    btn.setStyleSheet(
                        f"QPushButton {{ background-color: {DANGER}; color: white; "
                        f"border: none; border-radius: {int(6*s)}px; "
                        f"padding: {int(5*s)}px {int(12*s)}px; font-size: {int(12*s)}px; font-weight: bold; }}"
                        f"QPushButton:hover {{ background-color: #d33426; }}"
                    )

        # ── 7. 状态栏字体 ──
        self.statusBar().setStyleSheet(
            f"background-color: white; color: {TEXT_SECONDARY}; "
            f"border-top: 1px solid {ADMIN_TOOLBAR_BORDER}; "
            f"padding: {int(4*s)}px 16px; font-size: {int(11*s)}px;"
        )

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("人脸识别人事管理系统")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(900, 580)
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
        header.setMinimumHeight(48)
        header.setMaximumHeight(64)
        header.setStyleSheet(HEADER_STYLE)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        logo = QLabel("人脸识别人事管理系统")
        logo.setStyleSheet(
            f"font-size: 15px; font-weight: bold; color: {ADMIN_HEADER_TEXT}; "
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
        toolbar.setMinimumHeight(52)
        toolbar.setMaximumHeight(68)
        toolbar.setStyleSheet(TOOLBAR_STYLE)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(16, 8, 16, 8)
        toolbar_layout.setSpacing(6)

        # 左侧：操作按钮组
        add_btn = QPushButton("+ 添加员工")
        add_btn.setStyleSheet(TOOLBAR_BTN_PRIMARY)
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.clicked.connect(self.show_add_user_dialog)
        toolbar_layout.addWidget(add_btn)

        quick_btn = QPushButton("快速注册")
        quick_btn.setStyleSheet(TOOLBAR_BTN_PRIMARY)
        quick_btn.setCursor(Qt.PointingHandCursor)
        quick_btn.clicked.connect(self.show_quick_register_dialog)
        toolbar_layout.addWidget(quick_btn)

        toolbar_layout.addSpacing(2)

        batch_btn = QPushButton("批量导入")
        batch_btn.setStyleSheet(TOOLBAR_BTN)
        batch_btn.setCursor(Qt.PointingHandCursor)
        batch_btn.clicked.connect(self.show_batch_import_dialog)
        toolbar_layout.addWidget(batch_btn)

        uploads_btn = QPushButton("导入 uploads")
        uploads_btn.setStyleSheet(TOOLBAR_BTN)
        uploads_btn.setCursor(Qt.PointingHandCursor)
        uploads_btn.clicked.connect(self.import_from_uploads)
        toolbar_layout.addWidget(uploads_btn)

        gen_btn = QPushButton("生成编码")
        gen_btn.setStyleSheet(TOOLBAR_BTN)
        gen_btn.setCursor(Qt.PointingHandCursor)
        gen_btn.clicked.connect(self.generate_encodings)
        toolbar_layout.addWidget(gen_btn)

        model_btn = QPushButton("模型管理")
        model_btn.setStyleSheet(TOOLBAR_BTN)
        model_btn.setCursor(Qt.PointingHandCursor)
        model_btn.clicked.connect(self.show_model_manage_dialog)
        toolbar_layout.addWidget(model_btn)

        toolbar_layout.addStretch(1)

        # 下载状态指示器（有下载时显示）
        self._dl_label = QLabel("")
        self._dl_label.setStyleSheet(
            f"font-size: 12px; color: {ADMIN_ACCENT}; background: transparent; "
            f"padding: 4px 10px; border-radius: 6px; "
            f"background-color: #eff6ff;"
        )
        self._dl_label.setCursor(Qt.PointingHandCursor)
        self._dl_label.setVisible(False)
        self._dl_label.installEventFilter(self)
        toolbar_layout.addWidget(self._dl_label)

        # 右侧：搜索框（弹性宽度）
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索姓名 / 工号 / 部门...")
        self.search_input.setMinimumWidth(160)
        self.search_input.setMaximumWidth(320)
        self.search_input.setStyleSheet(SEARCH_INPUT)
        self.search_input.textChanged.connect(self.search_users)
        toolbar_layout.addWidget(self.search_input, 1)

        root.addWidget(toolbar)

        # ══════════════════════════════════════════
        # 主内容区：表格（直接放入 root，不套 wrapper）
        # ══════════════════════════════════════════
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(6)
        self.user_table.setHorizontalHeaderLabels(["工号", "姓名", "部门", "电话", "人脸数", "状态"])
        self.user_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.user_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.user_table.setAlternatingRowColors(False)
        self.user_table.setShowGrid(False)
        self.user_table.verticalHeader().setMinimumSectionSize(0)
        self.user_table.verticalHeader().setMaximumSectionSize(16777215)
        self.user_table.verticalHeader().setVisible(False)
        self.user_table.setShowGrid(False)
        self.user_table.setStyleSheet(TABLE_STYLE)
        self.user_table.setContentsMargins(16, 0, 16, 0)
        self.user_table.itemSelectionChanged.connect(self.on_user_select)
        # 列宽策略
        header_view = self.user_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 工号
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # 姓名
        header_view.setSectionResizeMode(2, QHeaderView.Stretch)           # 部门（拉伸填满）
        header_view.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 电话
        header_view.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 人脸数
        header_view.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # 状态
        # 整行 hover 高亮 delegate
        self._row_delegate = RowHighlightDelegate(self.user_table)
        self.user_table.setItemDelegate(self._row_delegate)
        self.user_table.setMouseTracking(True)
        self.user_table.viewport().installEventFilter(self)
        root.addWidget(self.user_table, 1)  # stretch=1：占据所有剩余空间

        # ══════════════════════════════════════════
        # 底部：选中员工详情卡
        # ══════════════════════════════════════════
        self.detail_card = QFrame()
        self.detail_card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 1px solid {ADMIN_TOOLBAR_BORDER};
                border-radius: 10px;
                margin-left: 16px;
                margin-right: 16px;
            }}
        """)
        self.detail_card.setMinimumHeight(90)
        self.detail_card.setMaximumHeight(140)
        detail_layout = QHBoxLayout(self.detail_card)
        detail_layout.setContentsMargins(20, 16, 20, 16)
        detail_layout.setSpacing(20)

        # 左侧：工作照片
        self.work_photo_preview = QLabel("未上传")
        self.work_photo_preview.setAlignment(Qt.AlignCenter)
        self.work_photo_preview.setFixedSize(100, 100)
        self.work_photo_preview.setWordWrap(True)
        self.work_photo_preview.setStyleSheet(
            f"border: 2px dashed {BORDER}; background-color: #f5f5f5; "
            f"color: #888888; font-size: 12px; border-radius: 10px;"
        )
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
        self.detail_info_label.setTextFormat(Qt.RichText)
        self.detail_info_label.setStyleSheet(
            f"font-size: 12px; color: {TEXT_SECONDARY}; background: transparent; line-height: 1.5;"
        )
        info_layout.addWidget(self.detail_info_label)

        info_layout.addStretch()
        detail_layout.addLayout(info_layout, 1)

        # 右侧：操作按钮（3×2 网格）
        actions_layout = QGridLayout()
        actions_layout.setSpacing(5)
        actions_layout.setContentsMargins(0, 2, 0, 2)

        btn_style_detail = TOOLBAR_BTN + "padding: 5px 8px; font-size: 12px; min-width: 70px;"

        edit_btn = QPushButton("编辑")
        edit_btn.setStyleSheet(btn_style_detail)
        edit_btn.setCursor(Qt.PointingHandCursor)
        edit_btn.clicked.connect(self.show_edit_user_dialog)
        actions_layout.addWidget(edit_btn, 0, 0)

        add_face_btn = QPushButton("补充人脸")
        add_face_btn.setToolTip("为已有员工添加更多人脸照片，提升识别准确率")
        add_face_btn.setStyleSheet(btn_style_detail)
        add_face_btn.setCursor(Qt.PointingHandCursor)
        add_face_btn.clicked.connect(self.show_add_face_dialog)
        actions_layout.addWidget(add_face_btn, 0, 1)

        upload_photo_btn = QPushButton("上传工作照")
        upload_photo_btn.setToolTip("上传员工证件照/工作照，用于人事管理展示")
        upload_photo_btn.setStyleSheet(btn_style_detail)
        upload_photo_btn.setCursor(Qt.PointingHandCursor)
        upload_photo_btn.clicked.connect(self.upload_work_photo)
        actions_layout.addWidget(upload_photo_btn, 0, 2)

        import_photo_btn = QPushButton("导入照片")
        import_photo_btn.setToolTip("从文件夹批量导入照片到选中员工")
        import_photo_btn.setStyleSheet(btn_style_detail)
        import_photo_btn.setCursor(Qt.PointingHandCursor)
        import_photo_btn.clicked.connect(self.import_photos_from_folder)
        actions_layout.addWidget(import_photo_btn, 1, 0)

        disable_btn = QPushButton("停用")
        disable_btn.setToolTip("停用员工（可恢复）")
        disable_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {WARNING}; "
            f"border: 1px solid #fde68a; border-radius: 8px; padding: 5px 8px; "
            f"font-size: 12px; font-weight: 500; min-width: 70px; }}"
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
        stats_frame.setMinimumHeight(36)
        stats_frame.setMaximumHeight(48)
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
        batch_disable_btn = QPushButton("批量停用")
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
        hard_del_btn = QPushButton("彻底删除")
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
        self.statusBar().showMessage("系统就绪")
        self.statusBar().setStyleSheet(
            f"background-color: white; color: {TEXT_SECONDARY}; "
            f"border-top: 1px solid {ADMIN_TOOLBAR_BORDER}; padding: 4px 16px; font-size: 11px;"
        )
        self.statusBar().setMinimumHeight(28)
        self.statusBar().setMaximumHeight(36)

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
            status = "启用" if is_active else "禁用"
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
            f"总员工  {total}   |   启用  {active}   |   人脸照片  {faces} 张"
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
            self.work_photo_preview.setText("未上传")
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
        self.detail_name_label.setText(f"{user['name']}")

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
        self.work_photo_preview.setText("未上传")
        self.work_photo_preview.setStyleSheet(
            f"border: 2px dashed {BORDER}; background-color: #fafbfc; "
            f"color: {TEXT_MUTED}; font-size: 12px; border-radius: 10px;"
        )
        self.work_photo_preview.setPixmap(QPixmap())

    def _load_work_photo(self, user):
        """加载并显示用户的工作照片"""
        work_photo_path = user.get('work_photo')
        print(f"[DEBUG] work_photo path: {work_photo_path}")
        print(f"[DEBUG] path exists: {os.path.exists(work_photo_path) if work_photo_path else 'N/A'}")

        if work_photo_path and os.path.exists(work_photo_path):
            try:
                pixmap = QPixmap()
                loaded = pixmap.load(work_photo_path)
                print(f"[DEBUG] pixmap loaded: {loaded}, size: {pixmap.width()}x{pixmap.height()}, null: {pixmap.isNull()}")
                if loaded and not pixmap.isNull():
                    self.work_photo_preview.setText("")
                    self.work_photo_preview.setStyleSheet(
                        f"border: 2px solid {SUCCESS}; background-color: #fff; border-radius: 10px;"
                    )
                    scaled = pixmap.scaled(
                        self.work_photo_preview.size(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    print(f"[DEBUG] scaled size: {scaled.width()}x{scaled.height()}")
                    self.work_photo_preview.setPixmap(scaled)
                    self.work_photo_preview.update()
                    return
            except Exception as e:
                print(f"[DEBUG] load error: {e}")
        self._set_work_photo_placeholder()

    # ── 照片操作 ────────────────────────────────────────────

    def upload_work_photo(self):
        """上传工作照片（纯展示用，不参与编码）"""
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

        # 弹出裁剪对话框
        crop_dialog = CropDialog(image, self)
        if not crop_dialog.exec_():
            return

        cropped = crop_dialog.get_cropped_image()
        if cropped is None:
            return

        os.makedirs(WORK_PHOTOS_DIR, exist_ok=True)
        save_path = os.path.join(WORK_PHOTOS_DIR, f"{user_id}.jpg")
        # 用 imencode + tofile 避免中文路径问题
        ext = os.path.splitext(save_path)[1]
        ok, buf = cv2.imencode(ext, cropped)
        if ok:
            buf.tofile(save_path)
        else:
            QMessageBox.critical(self, "错误", f"图片保存失败（编码失败）")
            return

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

    def show_model_manage_dialog(self):
        """显示模型管理对话框"""
        dialog = ModelManageDialog(self)
        dialog.exec_()
        # 对话框关闭后检查是否还在下载
        dm = DownloadManager()
        if dm.downloading:
            self._show_download_indicator()

    def _show_download_indicator(self):
        """在状态栏显示下载进度"""
        dm = DownloadManager()
        if not dm.downloading:
            self.statusBar().showMessage("系统就绪")
            return
        self.statusBar().showMessage(f"模型下载中... {dm.detail}")
        # 定时刷新
        if not hasattr(self, '_dl_poll'):
            from PyQt5.QtCore import QTimer
            self._dl_poll = QTimer(self)
            self._dl_poll.timeout.connect(self._refresh_status_bar)
        self._dl_poll.start(1000)

    def _refresh_status_bar(self):
        dm = DownloadManager()
        if dm.downloading:
            self.statusBar().showMessage(f"模型下载中... {dm.detail}")
            self._dl_label.setText(f"下载中 {dm.progress}%")
            self._dl_label.setVisible(True)
        else:
            self._dl_poll.stop()
            self._dl_label.setVisible(False)
            if dm.finished and not dm.error and not dm.error:
                self.statusBar().showMessage("模型下载完成")
            else:
                self.statusBar().showMessage("系统就绪")

    # ── 导入操作 ────────────────────────────────────────────

    def import_from_uploads(self):
        """一键从 data/uploads 目录导入人员"""
        try:
            self._do_import_from_uploads()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败：{str(e)}")

    def generate_encodings(self):
        """为所有用户生成人脸编码（可选择模型，仅显示可用模型）"""
        from admin.generate_encodings import get_last_model

        last_model = get_last_model()

        # 检查各模型是否可用
        available_models = []

        # mobilenet：需要至少一个训练好的模型
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_root = os.path.join(project_root, "trainer", "models", "saved")
            if os.path.isdir(models_root):
                for vdir in os.listdir(models_root):
                    if not os.path.isdir(os.path.join(models_root, vdir)):
                        continue
                    candidate = os.path.join(models_root, vdir, "inference_model.pth")
                    if os.path.isfile(candidate):
                        available_models.append(("mobilenet", "MobileFaceNet (128维, 自训练模型)"))
                        break
        except Exception:
            pass

        # insightface：需要 buffalo_l 模型
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            buffalo_dir = os.path.join(project_root, "deploy", "models", "buffalo_l")
            if os.path.isdir(buffalo_dir):
                for fname in ["w600k_r100.onnx", "w600k_r50.onnx"]:
                    if os.path.isfile(os.path.join(buffalo_dir, fname)):
                        available_models.append(("insightface", "InsightFace ArcFace-R100 (512维, 高精度)"))
                        break
        except Exception:
            pass

        # face_recognition：需要 pip 包
        try:
            import face_recognition as _fr  # noqa: F401
            available_models.append(("face_recognition", "face_recognition (128维, 兜底方案)"))
        except Exception:
            pass

        if not available_models:
            QMessageBox.warning(self, "无可用模型",
                "没有可用的识别模型！\n\n"
                "请先：\n"
                "1. 训练 mobilenet 模型，或\n"
                "2. 下载 InsightFace buffalo_l 模型，或\n"
                "3. 安装 face_recognition (pip install face-recognition)")
            return

        # 构建选项列表
        model_keys = [m[0] for m in available_models]
        model_labels = [m[1] for m in available_models]
        from PyQt5.QtWidgets import QInputDialog

        # 定位上次使用的模型
        default_idx = 0
        if last_model in model_keys:
            default_idx = model_keys.index(last_model)

        model_key, ok = QInputDialog.getItem(
            self, "选择编码模型",
            f"选择用于生成人脸编码的模型（共 {len(available_models)} 个可用）：",
            model_labels,
            default_idx,
            False
        )

        if not ok:
            return

        # 获取对应的模型 key
        model = model_keys[model_labels.index(model_key)]

        force = (model != last_model)

        if force:
            reply = QMessageBox.question(
                self, "模型切换",
                f"检测到模型切换：{last_model} → {model}\n\n"
                f"将清除 {last_model} 的旧编码并使用 {model} 重新生成。\n"
                "其他模型的编码不受影响。\n\n"
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

        save_btn = QPushButton("保存")
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

        save_btn = QPushButton("保存修改")
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
            f"当前已有 <b>{self.current_face_count}</b> 张  |  "
            f"建议每人至少 20 张，识别更准确"
        )
        tip.setStyleSheet(
            f"font-size: 13px; color: {TEXT_SECONDARY}; padding: 10px; "
            f"background: {PRIMARY_LIGHT}; border-radius: 8px;"
        )
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)

        preview_card = QGroupBox("摄像头预览")
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

        capture_btn = QPushButton("拍摄一张")
        capture_btn.setStyleSheet(btn_primary("min-height: 40px; font-size: 14px;"))
        capture_btn.clicked.connect(self.capture_one)
        btn_layout.addWidget(capture_btn)

        upload_btn = QPushButton("上传照片")
        upload_btn.setStyleSheet(btn_info("min-height: 40px; font-size: 14px;"))
        upload_btn.clicked.connect(self.upload_photos)
        btn_layout.addWidget(upload_btn)

        layout.addLayout(btn_layout)

        self.photo_list = QListWidget()
        self.photo_list.setMaximumHeight(80)
        layout.addWidget(self.photo_list)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        clear_btn = QPushButton("清空")
        clear_btn.setStyleSheet(btn_danger())
        clear_btn.clicked.connect(self.clear_photos)
        bottom_layout.addWidget(clear_btn)

        bottom_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(btn_outline())
        cancel_btn.clicked.connect(self.close_dialog)
        bottom_layout.addWidget(cancel_btn)

        save_btn = QPushButton("保存")
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
        self.photo_list.addItem(f"第 {len(self.captured_images)} 张 - 已拍摄")
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
                        self.photo_list.addItem(f"{os.path.basename(file_path)} - 已选择")
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

        tip = QLabel("填写信息 → 对准摄像头拍照（至少5张）→ 点击注册")
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

        preview_card = QGroupBox("摄像头预览")
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

        capture_btn = QPushButton("拍摄一张")
        capture_btn.setStyleSheet(btn_primary("min-height: 40px; font-size: 14px;"))
        capture_btn.clicked.connect(self.capture_one)
        capture_layout.addWidget(capture_btn)

        upload_btn = QPushButton("批量上传")
        upload_btn.setStyleSheet(btn_info("min-height: 40px; font-size: 14px;"))
        upload_btn.clicked.connect(self.upload_photos)
        capture_layout.addWidget(upload_btn)

        clear_btn = QPushButton("清空重拍")
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

        register_btn = QPushButton("一键注册")
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

        browse_btn = QPushButton("选择文件夹")
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

        import_btn = QPushButton("开始导入")
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
            status = "就绪" if is_ready else "图片不足"
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


# ============================================================
# 模型管理对话框
# ============================================================

class CropDialog(QDialog):
    """照片裁剪对话框：拖拽选框裁剪工作照"""

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("裁剪工作照片")
        self.setMinimumSize(600, 500)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self.original_image = image.copy()
        self.display_image = image.copy()
        self.crop_rect = None  # (x1, y1, x2, y2) 在原图坐标系
        self._dragging = False
        self._drag_start = None
        self._drag_offset = None  # (ox, oy) 拖拽时鼠标在框内的偏移
        self._resize_handle = None  # 'tl','tr','bl','br','t','b','l','r' or None
        self._handle_size = 8

        h, w = image.shape[:2]
        self._img_h, self._img_w = h, w

        # 默认裁剪框：居中 80%
        cw, ch = int(w * 0.8), int(h * 0.8)
        cx, cy = (w - cw) // 2, (h - ch) // 2
        self.crop_rect = [cx, cy, cx + cw, cy + ch]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        tip = QLabel("拖拽调整裁剪区域，四个角和四条边可拖拽缩放")
        tip.setStyleSheet(f"font-size: 13px; color: {TEXT_SECONDARY};")
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)

        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        self.canvas.setMinimumHeight(300)
        self.canvas.setMouseTracking(True)
        layout.addWidget(self.canvas, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(btn_outline())
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        confirm_btn = QPushButton("确认裁剪")
        confirm_btn.setStyleSheet(btn_success("min-height: 38px; min-width: 100px;"))
        confirm_btn.clicked.connect(self.accept)
        btn_row.addWidget(confirm_btn)
        layout.addLayout(btn_row)

        self._refresh()

    def get_cropped_image(self):
        """返回裁剪后的图片（numpy array）"""
        if self.crop_rect is None:
            return None
        x1, y1, x2, y2 = self.crop_rect
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(self._img_w, int(x2)), min(self._img_h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return self.original_image[y1:y2, x1:x2].copy()

    def _refresh(self):
        """重绘画布"""
        img = self.display_image.copy()
        h, w = img.shape[:2]

        # 半透明遮罩
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        x1, y1, x2, y2 = self.crop_rect
        overlay[y1:y2, x1:x2] = img[y1:y2, x1:x2].copy()
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

        # 画裁剪框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 100), 2)

        # 四角手柄
        hs = self._handle_size
        for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            cv2.rectangle(img, (hx - hs, hy - hs), (hx + hs, hy + hs), (0, 200, 100), -1)

        # BGR -> RGB -> QPixmap
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img.copy())
        scaled = pixmap.scaled(self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.canvas.setPixmap(scaled)

        # 记录缩放映射
        self._scale = min(self.canvas.width() / w, self.canvas.height() / h) if w > 0 and h > 0 else 1.0
        self._offset_x = (self.canvas.width() - int(w * self._scale)) // 2
        self._offset_y = (self.canvas.height() - int(h * self._scale)) // 2

    def _canvas_to_img(self, cx, cy):
        """画布坐标 -> 原图坐标"""
        ix = int((cx - self._offset_x) / self._scale)
        iy = int((cy - self._offset_y) / self._scale)
        return ix, iy

    def _hit_handle(self, ix, iy):
        """检测鼠标是否在拖拽手柄上"""
        x1, y1, x2, y2 = self.crop_rect
        hs = self._handle_size * 2
        # 四角
        if abs(ix - x1) < hs and abs(iy - y1) < hs: return 'tl'
        if abs(ix - x2) < hs and abs(iy - y1) < hs: return 'tr'
        if abs(ix - x1) < hs and abs(iy - y2) < hs: return 'bl'
        if abs(ix - x2) < hs and abs(iy - y2) < hs: return 'br'
        # 四边
        if y1 - hs < iy < y1 + hs and x1 < ix < x2: return 't'
        if y2 - hs < iy < y2 + hs and x1 < ix < x2: return 'b'
        if x1 - hs < ix < x1 + hs and y1 < iy < y2: return 'l'
        if x2 - hs < ix < x2 + hs and y1 < iy < y2: return 'r'
        # 框内
        if x1 < ix < x2 and y1 < iy < y2: return 'move'
        return None

    def _clamp_rect(self):
        """限制裁剪框在图片范围内"""
        x1, y1, x2, y2 = self.crop_rect
        size = 20  # 最小尺寸
        x1 = max(0, min(x1, self._img_w - size))
        y1 = max(0, min(y1, self._img_h - size))
        x2 = max(x1 + size, min(x2, self._img_w))
        y2 = max(y1 + size, min(y2, self._img_h))
        self.crop_rect = [x1, y1, x2, y2]

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos() - self.canvas.pos()
            ix, iy = self._canvas_to_img(pos.x(), pos.y())
            handle = self._hit_handle(ix, iy)
            if handle:
                self._dragging = True
                self._resize_handle = handle
                self._drag_start = (ix, iy)
                self._drag_offset = (ix - self.crop_rect[0], iy - self.crop_rect[1])

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        pos = event.pos() - self.canvas.pos()
        ix, iy = self._canvas_to_img(pos.x(), pos.y())
        dx = ix - self._drag_start[0]
        dy = iy - self._drag_start[1]

        x1, y1, x2, y2 = self.crop_rect
        handle = self._resize_handle

        if handle == 'move':
            w, h = x2 - x1, y2 - y1
            nx1 = ix - self._drag_offset[0]
            ny1 = iy - self._drag_offset[1]
            self.crop_rect = [nx1, ny1, nx1 + w, ny1 + h]
        elif handle == 'tl':
            self.crop_rect = [x1 + dx, y1 + dy, x2, y2]
        elif handle == 'tr':
            self.crop_rect = [x1, y1 + dy, x2 + dx, y2]
        elif handle == 'bl':
            self.crop_rect = [x1 + dx, y1, x2, y2 + dy]
        elif handle == 'br':
            self.crop_rect = [x1, y1, x2 + dx, y2 + dy]
        elif handle == 't':
            self.crop_rect = [x1, y1 + dy, x2, y2]
        elif handle == 'b':
            self.crop_rect = [x1, y1, x2, y2 + dy]
        elif handle == 'l':
            self.crop_rect = [x1 + dx, y1, x2, y2]
        elif handle == 'r':
            self.crop_rect = [x1, y1, x2 + dx, y2]

        self._clamp_rect()
        self._drag_start = (ix, iy)
        self._refresh()

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self._resize_handle = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class ModelManageDialog(QDialog):
    """模型管理：查看状态、下载、删除模型权重，首次使用引导"""

    BUFFALO_L_DIR = DownloadManager.BUFFALO_L_DIR

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型管理")
        self.setMinimumWidth(640)
        self.setMinimumHeight(560)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        self._dm = DownloadManager()
        self._poll_timer = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("模型权重管理")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {PRIMARY}; background: transparent;")
        layout.addWidget(title)

        # ── 首次使用引导条 ──
        self._guide_frame = QFrame()
        self._guide_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #fffbeb;
                border: 1px solid #fde68a;
                border-radius: 10px;
            }}
        """)
        guide_layout = QVBoxLayout(self._guide_frame)
        guide_layout.setContentsMargins(16, 12, 16, 12)
        guide_layout.setSpacing(8)

        self._guide_title = QLabel()
        self._guide_title.setStyleSheet(f"font-size: 14px; font-weight: bold; color: #92400e; background: transparent;")
        guide_layout.addWidget(self._guide_title)

        self._guide_steps = QLabel()
        self._guide_steps.setStyleSheet(f"font-size: 12px; color: #78350f; background: transparent; line-height: 1.6;")
        self._guide_steps.setWordWrap(True)
        guide_layout.addWidget(self._guide_steps)

        layout.addWidget(self._guide_frame)

        # ── InsightFace buffalo_l ──
        self._build_insightface_card(layout)

        # ── MobileFaceNet 训练模型 ──
        self._build_mobilenet_card(layout)

        # ── face_recognition ──
        self._build_face_recognition_card(layout)

        layout.addStretch()

        # 关闭按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet(btn_outline())
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # 刷新引导
        self._refresh_guide()

    def _build_insightface_card(self, parent_layout):
        """InsightFace buffalo_l 模型卡片"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(10)

        # 标题行
        header = QHBoxLayout()
        name_label = QLabel("InsightFace ArcFace-R100")
        name_label.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {TEXT_PRIMARY}; background: transparent;")
        header.addWidget(name_label)

        dim_label = QLabel("512维 · 高精度预训练")
        dim_label.setStyleSheet(f"font-size: 12px; color: {TEXT_SECONDARY}; background: transparent;")
        header.addWidget(dim_label)
        header.addStretch()

        self.if_status_label = QLabel()
        self.if_status_label.setStyleSheet(f"font-size: 12px; background: transparent;")
        header.addWidget(self.if_status_label)
        card_layout.addLayout(header)

        # 路径 + 大小
        self.if_detail_label = QLabel()
        self.if_detail_label.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED}; background: transparent;")
        self.if_detail_label.setWordWrap(True)
        card_layout.addWidget(self.if_detail_label)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.if_download_btn = QPushButton("下载模型")
        self.if_download_btn.setStyleSheet(btn_primary("padding: 8px 20px; font-size: 13px;"))
        self.if_download_btn.setCursor(Qt.PointingHandCursor)
        self.if_download_btn.clicked.connect(self._start_download)
        btn_row.addWidget(self.if_download_btn)

        self.if_delete_btn = QPushButton("删除模型")
        self.if_delete_btn.setStyleSheet(btn_danger("padding: 8px 16px; font-size: 13px;"))
        self.if_delete_btn.setCursor(Qt.PointingHandCursor)
        self.if_delete_btn.clicked.connect(self._delete_buffalo_l)
        btn_row.addWidget(self.if_delete_btn)

        self.if_pause_btn = QPushButton("暂停")
        self.if_pause_btn.setStyleSheet(TOOLBAR_BTN + "padding: 8px 16px; font-size: 13px;")
        self.if_pause_btn.setCursor(Qt.PointingHandCursor)
        self.if_pause_btn.clicked.connect(self._toggle_pause)
        self.if_pause_btn.setVisible(False)
        btn_row.addWidget(self.if_pause_btn)

        self.if_cancel_btn = QPushButton("取消")
        self.if_cancel_btn.setStyleSheet(btn_danger("padding: 8px 16px; font-size: 13px;"))
        self.if_cancel_btn.setCursor(Qt.PointingHandCursor)
        self.if_cancel_btn.clicked.connect(self._cancel_download)
        self.if_cancel_btn.setVisible(False)
        btn_row.addWidget(self.if_cancel_btn)

        btn_row.addStretch()
        card_layout.addLayout(btn_row)

        # 进度条（下载时显示）
        self.if_progress = QProgressBar()
        self.if_progress.setVisible(False)
        self.if_progress.setStyleSheet("""
            QProgressBar { border: 1px solid #ccc; border-radius: 6px; text-align: center; height: 20px; background: #f0f0f0; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4CAF50, stop:1 #66BB6A); border-radius: 5px; }
        """)
        card_layout.addWidget(self.if_progress)

        parent_layout.addWidget(card)
        self._refresh_insightface_status()

    def _build_mobilenet_card(self, parent_layout):
        """MobileFaceNet 训练模型卡片"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(10)

        header = QHBoxLayout()
        name_label = QLabel("MobileFaceNet")
        name_label.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {TEXT_PRIMARY}; background: transparent;")
        header.addWidget(name_label)

        dim_label = QLabel("128维 · 自训练模型")
        dim_label.setStyleSheet(f"font-size: 12px; color: {TEXT_SECONDARY}; background: transparent;")
        header.addWidget(dim_label)
        header.addStretch()
        card_layout.addLayout(header)

        # 模型版本列表
        self.mbn_list = QTableWidget()
        self.mbn_list.setColumnCount(4)
        self.mbn_list.setHorizontalHeaderLabels(["版本", "大小", "状态", "操作"])
        self.mbn_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mbn_list.verticalHeader().setVisible(False)
        self.mbn_list.setShowGrid(False)
        self.mbn_list.setMaximumHeight(150)
        self.mbn_list.setStyleSheet(TABLE_STYLE)
        card_layout.addWidget(self.mbn_list)

        parent_layout.addWidget(card)
        self._refresh_mobilenet_status()

    def _build_face_recognition_card(self, parent_layout):
        """face_recognition 模型卡片"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
        """)
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)

        name_label = QLabel("face_recognition (dlib 内置)")
        name_label.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {TEXT_PRIMARY}; background: transparent;")
        card_layout.addWidget(name_label)

        dim_label = QLabel("128维 · 兜底方案")
        dim_label.setStyleSheet(f"font-size: 12px; color: {TEXT_SECONDARY}; background: transparent;")
        card_layout.addWidget(dim_label)
        card_layout.addStretch()

        # 检测是否已安装
        try:
            import face_recognition
            status = QLabel("已安装")
            status.setStyleSheet(f"font-size: 12px; color: {SUCCESS}; background: transparent;")
        except ImportError:
            status = QLabel("未安装")
            status.setStyleSheet(f"font-size: 12px; color: {DANGER}; background: transparent;")
        card_layout.addWidget(status)

        parent_layout.addWidget(card)

    # ── InsightFace 状态刷新 ──

    def _refresh_insightface_status(self):
        """刷新 InsightFace 模型状态"""
        self._refresh_guide()
        r100_path = os.path.join(self.BUFFALO_L_DIR, "w600k_r100.onnx")
        r50_path = os.path.join(self.BUFFALO_L_DIR, "w600k_r50.onnx")

        if os.path.exists(r100_path):
            size_mb = os.path.getsize(r100_path) / (1024 * 1024)
            self.if_status_label.setText("已下载")
            self.if_status_label.setStyleSheet(f"font-size: 12px; color: {SUCCESS}; background: transparent;")
            self.if_detail_label.setText(f"路径: {self.BUFFALO_L_DIR}\n模型: w600k_r100.onnx ({size_mb:.0f}MB)")
            self.if_download_btn.setEnabled(False)
            self.if_download_btn.setText("已下载")
            self.if_delete_btn.setEnabled(True)
        elif os.path.exists(r50_path):
            size_mb = os.path.getsize(r50_path) / (1024 * 1024)
            self.if_status_label.setText("已下载 (R50)")
            self.if_status_label.setStyleSheet(f"font-size: 12px; color: {SUCCESS}; background: transparent;")
            self.if_detail_label.setText(f"路径: {self.BUFFALO_L_DIR}\n模型: w600k_r50.onnx ({size_mb:.0f}MB)")
            self.if_download_btn.setEnabled(False)
            self.if_download_btn.setText("已下载")
            self.if_delete_btn.setEnabled(True)
        else:
            self.if_status_label.setText("未下载")
            self.if_status_label.setStyleSheet(f"font-size: 12px; color: {DANGER}; background: transparent;")
            self.if_detail_label.setText(f"路径: {self.BUFFALO_L_DIR}\n需要下载 buffalo_l 包 (~300MB)")
            self.if_download_btn.setEnabled(True)
            self.if_download_btn.setText("下载模型")
            self.if_delete_btn.setEnabled(False)

    def _refresh_guide(self):
        """刷新首次使用引导条"""
        # 检查各组件状态
        has_if = False
        r100 = os.path.join(self.BUFFALO_L_DIR, "w600k_r100.onnx")
        r50 = os.path.join(self.BUFFALO_L_DIR, "w600k_r50.onnx")
        if os.path.exists(r100) or os.path.exists(r50):
            has_if = True

        has_mbn = False
        models_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "trainer", "models", "saved"
        )
        if os.path.isdir(models_root):
            for vdir in os.listdir(models_root):
                if os.path.isfile(os.path.join(models_root, vdir, "inference_model.pth")):
                    has_mbn = True
                    break

        has_fr = False
        try:
            import face_recognition as _fr  # noqa: F401
            has_fr = True
        except Exception:
            pass

        # 检查编码是否存在
        has_encodings = False
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for bin_name in ["embeddings_mbn.bin", "embeddings_if.bin", "embeddings_fr.bin"]:
            if os.path.isfile(os.path.join(project_root, "deploy", "models", bin_name)):
                has_encodings = True
                break

        any_model = has_if or has_mbn or has_fr

        if not any_model:
            self._guide_frame.setStyleSheet(f"""
                QFrame {{ background-color: #fef2f2; border: 1px solid #fca5a5; border-radius: 10px; }}
            """)
            self._guide_title.setText("首次使用：需要获取识别模型")
            self._guide_steps.setText(
                "当前没有任何识别模型，无法生成人脸编码。请按以下任一方式获取：\n\n"
                "方式 1（推荐）：点击下方「InsightFace」卡片的「下载模型」按钮，一键下载高精度预训练模型\n"
                "方式 2：在终端运行 pip install face-recognition 安装兜底方案\n"
                "方式 3：使用 CASIA-WebFace 数据集训练自己的模型（详见 trainer/README.md）"
            )
            self._guide_frame.show()
        elif not has_encodings:
            self._guide_frame.setStyleSheet(f"""
                QFrame {{ background-color: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; }}
            """)
            self._guide_title.setText("下一步：生成人脸编码")
            self._guide_steps.setText(
                "模型已就绪，但还没有生成人脸编码。\n"
                "请关闭此窗口，点击工具栏「生成编码」按钮，选择模型后为已注册员工生成编码。"
            )
            self._guide_frame.show()
        else:
            self._guide_frame.setStyleSheet(f"""
                QFrame {{ background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 10px; }}
            """)
            self._guide_title.setText("一切就绪")
            self._guide_steps.setText("模型和编码均已配置，系统可正常使用。")
            self._guide_frame.show()

    def _start_download(self):
        """开始下载（委托给 DownloadManager）"""
        if self._dm.downloading:
            return
        reply = QMessageBox.question(
            self, "下载确认",
            "将从 GitHub 下载 InsightFace buffalo_l 模型包 (~300MB)。\n\n"
            "支持暂停和恢复下载，关闭窗口后下载继续在后台运行。\n\n"
            "是否开始下载？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return
        self._dm.start()
        self._sync_download_ui()
        self._start_poll()

    def _toggle_pause(self):
        """暂停/恢复"""
        if self._dm.paused:
            self._dm.resume()
        else:
            self._dm.pause()
        self._sync_download_ui()

    def _cancel_download(self):
        """取消下载"""
        if not self._dm.downloading:
            return
        reply = QMessageBox.question(
            self, "取消下载",
            "确定要取消下载吗？\n\n已下载的部分将被删除。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return
        self._dm.cancel()

    def _sync_download_ui(self):
        """根据 DownloadManager 状态同步 UI"""
        dm = self._dm
        if dm.downloading:
            self.if_download_btn.setVisible(False)
            self.if_delete_btn.setVisible(False)
            self.if_pause_btn.setVisible(True)
            self.if_pause_btn.setText("恢复" if dm.paused else "暂停")
            self.if_cancel_btn.setVisible(True)
            self.if_cancel_btn.setEnabled(not dm.paused or True)
            self.if_progress.setVisible(True)
            if dm.progress >= 0:
                self.if_progress.setValue(dm.progress)
            self.if_detail_label.setText(dm.detail)
        else:
            self.if_download_btn.setVisible(True)
            self.if_delete_btn.setVisible(True)
            self.if_pause_btn.setVisible(False)
            self.if_cancel_btn.setVisible(False)
            self.if_progress.setVisible(False)
            if dm.finished:
                if dm.error:
                    if "取消" not in dm.error:
                        self.if_detail_label.setText(f"下载失败: {dm.error}")
                    else:
                        self.if_detail_label.setText("下载已取消")
                else:
                    self.if_detail_label.setText("下载完成")
            self._refresh_insightface_status()

    def _start_poll(self):
        """启动定时轮询 DownloadManager 状态"""
        if self._poll_timer is None:
            from PyQt5.QtCore import QTimer
            self._poll_timer = QTimer(self)
            self._poll_timer.timeout.connect(self._sync_download_ui)
        self._poll_timer.start(500)

    def _stop_poll(self):
        if self._poll_timer:
            self._poll_timer.stop()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_download_ui()
        if self._dm.downloading:
            self._start_poll()

    def closeEvent(self, event):
        """关闭窗口：停止轮询，但不停止下载"""
        self._stop_poll()
        super().closeEvent(event)

    def _delete_buffalo_l(self):
        """删除 InsightFace buffalo_l 模型"""
        if not os.path.exists(self.BUFFALO_L_DIR):
            return

        # 计算大小
        total_size = 0
        file_count = 0
        for f in os.listdir(self.BUFFALO_L_DIR):
            fp = os.path.join(self.BUFFALO_L_DIR, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
                file_count += 1

        size_mb = total_size / (1024 * 1024)

        reply = QMessageBox.question(
            self, "删除确认",
            f"确定要删除 InsightFace buffalo_l 模型吗？\n\n"
            f"文件: {file_count} 个\n"
            f"大小: {size_mb:.0f} MB\n"
            f"路径: {self.BUFFALO_L_DIR}\n\n"
            "删除后将无法使用 InsightFace 识别器。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        try:
            shutil.rmtree(self.BUFFALO_L_DIR)
            QMessageBox.information(self, "删除成功", "InsightFace buffalo_l 模型已删除。")
        except Exception as e:
            QMessageBox.critical(self, "删除失败", f"删除失败：{e}")

        self._refresh_insightface_status()

    # ── MobileFaceNet 状态刷新 ──

    def _refresh_mobilenet_status(self):
        """刷新 MobileFaceNet 模型版本列表"""
        models_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "trainer", "models", "saved"
        )

        self.mbn_list.setRowCount(0)

        if not os.path.exists(models_root):
            return

        row = 0
        for name in sorted(os.listdir(models_root)):
            model_dir = os.path.join(models_root, name)
            if not os.path.isdir(model_dir):
                continue

            has_inference = os.path.exists(os.path.join(model_dir, "inference_model.pth"))
            has_best = os.path.exists(os.path.join(model_dir, "best_model.pth"))

            if not has_inference and not has_best:
                continue

            # 计算总大小
            total_size = sum(
                os.path.getsize(os.path.join(model_dir, f))
                for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))
            )
            size_str = f"{total_size / (1024 * 1024):.1f} MB"

            status = "就绪" if has_inference else "仅 best_model"
            status_color = SUCCESS if has_inference else WARNING

            self.mbn_list.insertRow(row)
            self.mbn_list.setItem(row, 0, QTableWidgetItem(name))

            size_item = QTableWidgetItem(size_str)
            size_item.setTextAlignment(Qt.AlignCenter)
            self.mbn_list.setItem(row, 1, size_item)

            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setForeground(QColor(status_color))
            self.mbn_list.setItem(row, 2, status_item)

            del_btn = QPushButton("删除")
            del_btn.setStyleSheet(btn_danger("padding: 4px 10px; font-size: 11px;"))
            del_btn.setCursor(Qt.PointingHandCursor)
            del_btn.clicked.connect(lambda checked, d=model_dir, n=name: self._delete_mobilenet_version(d, n))
            self.mbn_list.setCellWidget(row, 3, del_btn)
            self.mbn_list.setRowHeight(row, 36)

            row += 1

    def _delete_mobilenet_version(self, model_dir, name):
        """删除指定版本的 MobileFaceNet 模型"""
        reply = QMessageBox.question(
            self, "删除确认",
            f"确定要删除 MobileFaceNet 模型版本 '{name}' 吗？\n\n"
            f"路径: {model_dir}\n\n"
            "此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        try:
            shutil.rmtree(model_dir)
            QMessageBox.information(self, "删除成功", f"模型版本 '{name}' 已删除。")
        except Exception as e:
            QMessageBox.critical(self, "删除失败", f"删除失败：{e}")

        self._refresh_mobilenet_status()

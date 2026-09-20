"""
首次运行配置向导
检测缺失的依赖和模型，引导用户完成初始配置
"""

import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QProgressBar, QTextEdit,
                             QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


BG = "#f0f2f5"
SURFACE = "#ffffff"
BORDER = "#e2e8f0"
TEXT = "#1e293b"
TEXT_DIM = "#64748b"
SUCCESS = "#16a34a"
DANGER = "#dc2626"
WARN = "#f59e0b"
ACCENT = "#2563eb"


class SetupWorker(QThread):
    """后台安装线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, task):
        super().__init__()
        self.task = task

    def run(self):
        try:
            if self.task == "install_packages":
                self._install_packages()
            elif self.task == "download_yolo":
                self._download_yolo()
            elif self.task == "download_buffalo":
                self._download_buffalo()
            elif self.task == "install_face_recognition":
                self._install_face_recognition()
        except Exception as e:
            self.finished.emit(False, str(e))

    def _install_packages(self):
        packages = [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("PyQt5", "PyQt5"),
            ("ultralytics", "Ultralytics (YOLO)"),
            ("cv2", "OpenCV"),
        ]
        for module, name in packages:
            try:
                __import__(module)
                self.progress.emit(f"✓ {name} 已安装")
            except ImportError:
                self.progress.emit(f"安装 {name}...")
                pkg = module if module != "cv2" else "opencv-python"
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    self.progress.emit(f"✓ {name} 安装成功")
                else:
                    self.progress.emit(f"✗ {name} 安装失败: {result.stderr[:100]}")

        self.finished.emit(True, "依赖检查完成")

    def _download_yolo(self):
        yolo_path = os.path.join(ROOT_DIR, "trainer", "models", "yolov11l-face.pt")
        if os.path.exists(yolo_path):
            self.finished.emit(True, "YOLO 模型已存在")
            return

        self.progress.emit("下载 YOLO 人脸检测模型 (~49MB)...")
        try:
            from ultralytics import YOLO
            model = YOLO("yolov11l-face.pt")
            # ultralytics 会自动下载到缓存，我们需要复制到项目目录
            import shutil
            cache_path = model.ckpt.get("path", "") if hasattr(model, 'ckpt') else ""
            # 直接用 YOLO 的内置下载机制
            model = YOLO("yolov11n-face.pt")  # 先试小模型
            # 实际上 ultralytics 下载后会在 ~/.cache/ultralytics
            # 更简单的方式：直接下载文件
            self._download_file(
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov11n.pt",
                yolo_path
            )
            self.finished.emit(True, "YOLO 模型下载完成")
        except Exception as e:
            self.finished.emit(False, f"YOLO 下载失败: {e}")

    def _download_buffalo(self):
        buffalo_dir = os.path.join(ROOT_DIR, "deploy", "models", "buffalo_l")
        if os.path.isdir(buffalo_dir):
            for f in ["w600k_r100.onnx", "w600k_r50.onnx"]:
                if os.path.isfile(os.path.join(buffalo_dir, f)):
                    self.finished.emit(True, "InsightFace 模型已存在")
                    return

        self.progress.emit("下载 InsightFace buffalo_l (~300MB)...")
        url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        zip_path = os.path.join(ROOT_DIR, "deploy", "models", "buffalo_l.zip")
        try:
            self._download_file(url, zip_path)
            self.progress.emit("解压中...")
            import zipfile
            os.makedirs(buffalo_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    parts = member.split('/')
                    if len(parts) >= 2 and parts[0] == 'buffalo_l' and parts[-1]:
                        target = os.path.join(buffalo_dir, parts[-1])
                        with zf.open(member) as src, open(target, 'wb') as dst:
                            dst.write(src.read())
            os.remove(zip_path)
            self.finished.emit(True, "InsightFace 模型下载完成")
        except Exception as e:
            self.finished.emit(False, f"InsightFace 下载失败: {e}")

    def _install_face_recognition(self):
        self.progress.emit("安装 face_recognition...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "face-recognition"],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            self.finished.emit(True, "face_recognition 安装成功")
        else:
            self.finished.emit(False, f"安装失败（需要 Visual C++ Build Tools）")

    def _download_file(self, url, dest):
        import urllib.request
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(99, int(block_num * block_size / total_size * 100))
                self.progress.emit(f"下载中... {pct}%")

        urllib.request.urlretrieve(url, dest, reporthook=_progress)


class SetupWizard(QDialog):
    """首次运行配置向导"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("首次配置向导")
        self.setMinimumSize(560, 480)
        self.setStyleSheet(f"background-color: {BG};")
        self._worker = None
        self.init_ui()
        self._check_all()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        title = QLabel("首次配置向导")
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {TEXT};")
        layout.addWidget(title)

        desc = QLabel("检测到系统缺少必要的模型和依赖，需要完成以下配置才能使用：")
        desc.setStyleSheet(f"font-size: 13px; color: {TEXT_DIM};")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # ── 检查项列表 ──
        self.check_frame = QFrame()
        self.check_frame.setStyleSheet(f"""
            QFrame {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
        """)
        self.check_layout = QVBoxLayout(self.check_frame)
        self.check_layout.setContentsMargins(16, 12, 16, 12)
        self.check_layout.setSpacing(8)
        layout.addWidget(self.check_frame)

        # ── 日志 ──
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setStyleSheet(f"""
            QTextEdit {{
                background: #1e293b; color: #e2e8f0; border: none;
                border-radius: 8px; padding: 8px; font-size: 12px;
                font-family: 'Consolas', monospace;
            }}
        """)
        layout.addWidget(self.log)

        # ── 进度条 ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{ border: 1px solid {BORDER}; border-radius: 6px; height: 20px; background: #f0f0f0; }}
            QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ACCENT}, stop:1 #60a5fa); border-radius: 5px; }}
        """)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        # ── 按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.skip_btn = QPushButton("跳过，稍后配置")
        self.skip_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {TEXT_DIM}; border: 1px solid {BORDER};
                border-radius: 8px; padding: 8px 16px; font-size: 13px; }}
            QPushButton:hover {{ background: #f1f5f9; }}
        """)
        self.skip_btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.skip_btn)

        self.setup_btn = QPushButton("一键配置")
        self.setup_btn.setStyleSheet(f"""
            QPushButton {{ background: {ACCENT}; color: white; border: none;
                border-radius: 8px; padding: 8px 24px; font-size: 14px; font-weight: bold; }}
            QPushButton:hover {{ background: #1d4ed8; }}
            QPushButton:disabled {{ background: #94a3b8; }}
        """)
        self.setup_btn.setCursor(Qt.PointingHandCursor)
        self.setup_btn.clicked.connect(self._start_setup)
        btn_row.addWidget(self.setup_btn)

        layout.addLayout(btn_row)

    def _check_all(self):
        """检测所有缺失项"""
        self._checks = []

        # 1. PyTorch
        ok = self._check("PyTorch", "import torch")
        self._checks.append(("torch", ok))

        # 2. ultralytics
        ok = self._check("Ultralytics (YOLO)", "import ultralytics")
        self._checks.append(("ultralytics", ok))

        # 3. YOLO 模型
        yolo_path = os.path.join(ROOT_DIR, "trainer", "models", "yolov11l-face.pt")
        ok = os.path.exists(yolo_path)
        self._add_check_item("YOLO 人脸检测模型", ok,
                             "已就绪" if ok else f"缺失 ({yolo_path})")
        self._checks.append(("yolo_model", ok))

        # 4. 识别模型（三选一）
        has_rec = False
        # insightface
        buffalo_dir = os.path.join(ROOT_DIR, "deploy", "models", "buffalo_l")
        if os.path.isdir(buffalo_dir):
            for f in ["w600k_r100.onnx", "w600k_r50.onnx"]:
                if os.path.isfile(os.path.join(buffalo_dir, f)):
                    has_rec = True
                    break
        # mobilenet
        if not has_rec:
            models_root = os.path.join(ROOT_DIR, "trainer", "models", "saved")
            if os.path.isdir(models_root):
                for v in os.listdir(models_root):
                    if os.path.isfile(os.path.join(models_root, v, "inference_model.pth")):
                        has_rec = True
                        break
        # face_recognition
        fr_ok = False
        try:
            import face_recognition
            fr_ok = True
        except ImportError:
            pass
        if fr_ok:
            has_rec = True

        self._add_check_item("识别模型（至少需要一个）", has_rec,
                             "已就绪" if has_rec else "缺失（InsightFace / MobileFaceNet / face_recognition）")
        self._checks.append(("recognition", has_rec))

        # 5. 编码文件
        has_enc = False
        for name in ["embeddings_mbn.bin", "embeddings_if.bin", "embeddings_fr.bin"]:
            if os.path.isfile(os.path.join(ROOT_DIR, "deploy", "models", name)):
                has_enc = True
                break
        self._add_check_item("人脸编码数据库", has_enc,
                             "已就绪" if has_enc else "缺失（注册员工后生成）")
        self._checks.append(("encoding", has_enc))

    def _check(self, name, import_str):
        """检测 Python 包是否已安装"""
        try:
            exec(import_str)
            self._add_check_item(name, True, "已安装")
            return True
        except ImportError:
            self._add_check_item(name, False, "未安装")
            return False

    def _add_check_item(self, name, ok, detail):
        """添加一个检查项到界面"""
        frame = QFrame()
        frame.setStyleSheet(f"background: transparent; border: none;")
        row = QHBoxLayout(frame)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        icon = QLabel("✓" if ok else "✗")
        icon.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {SUCCESS if ok else DANGER}; background: transparent;")
        icon.setFixedWidth(20)
        row.addWidget(icon)

        lbl = QLabel(name)
        lbl.setStyleSheet(f"font-size: 13px; color: {TEXT}; background: transparent;")
        row.addWidget(lbl, 1)

        detail_lbl = QLabel(detail)
        detail_lbl.setStyleSheet(f"font-size: 11px; color: {TEXT_DIM}; background: transparent;")
        row.addWidget(detail_lbl)

        self.check_layout.addWidget(frame)

    def _start_setup(self):
        """开始一键配置"""
        self.setup_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)  # 不确定进度
        self.log.clear()

        # 按需执行任务
        tasks = []
        packages_ok = all(c[1] for c in self._checks if c[0] in ("torch", "ultralytics"))
        if not packages_ok:
            tasks.append("install_packages")
        if not dict(self._checks).get("yolo_model"):
            tasks.append("download_yolo")
        if not dict(self._checks).get("recognition"):
            tasks.append("download_buffalo")

        if not tasks:
            self.log.append("所有组件已就绪，无需配置。")
            self._finish_setup(True, "")
            return

        self._task_queue = tasks
        self._run_next_task()

    def _run_next_task(self):
        if not self._task_queue:
            self._finish_setup(True, "配置完成")
            return

        task = self._task_queue.pop(0)
        self.log.append(f"\n▶ 执行: {task}")
        self._worker = SetupWorker(task)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_task_done)
        self._worker.start()

    def _on_progress(self, msg):
        self.log.append(f"  {msg}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _on_task_done(self, success, msg):
        if success:
            self.log.append(f"  ✓ {msg}")
        else:
            self.log.append(f"  ✗ {msg}")
        self._run_next_task()

    def _finish_setup(self, success, msg):
        self.progress_bar.setVisible(False)
        self.setup_btn.setEnabled(True)
        self.skip_btn.setEnabled(True)

        if success:
            self.log.append("\n✅ 配置完成！重新检查...")
            self.check_layout.removeWidget(self.check_frame)
            self.check_frame.deleteLater()
            self._check_all()
        else:
            self.log.append(f"\n❌ {msg}")


def check_first_run():
    """检查是否需要首次配置，返回 True 表示需要弹出向导"""
    # 如果任何一个关键组件缺失，弹出向导
    yolo = os.path.exists(os.path.join(ROOT_DIR, "trainer", "models", "yolov11l-face.pt"))
    buffalo = os.path.isdir(os.path.join(ROOT_DIR, "deploy", "models", "buffalo_l"))
    mbn = False
    models_root = os.path.join(ROOT_DIR, "trainer", "models", "saved")
    if os.path.isdir(models_root):
        for v in os.listdir(models_root):
            if os.path.isfile(os.path.join(models_root, v, "inference_model.pth")):
                mbn = True
                break
    fr = False
    try:
        import face_recognition
        fr = True
    except ImportError:
        pass

    return not (yolo and (buffalo or mbn or fr))

"""
门禁系统 - 主程序入口
"""

import sys
import os

# 添加项目根目录到Python路径（必须在最前面）
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# 确保 gate 目录也在路径中
gate_dir = os.path.dirname(os.path.abspath(__file__))
if gate_dir not in sys.path:
    sys.path.insert(0, gate_dir)

# 使用统一的 PyTorch 工具模块
from common import torch_utils
if torch_utils.TORCH_AVAILABLE:
    print(f"Torch loaded: {torch_utils.torch.__version__}")
    print(f"CUDA available: {torch_utils.CUDA_AVAILABLE}")
else:
    print("Warning: PyTorch not available")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from gui.gate_window import GateWindow


def main():
    """主函数"""
    print("=" * 50)
    print("人脸识别门禁系统 v2.0")
    print("=" * 50)
    print("正在启动系统...")
    print("请确保摄像头已连接")
    print("=" * 50)

    # 创建Qt应用
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    # 设置默认字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 创建主窗口
    window = GateWindow()
    window.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

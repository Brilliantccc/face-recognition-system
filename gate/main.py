"""
门禁系统 - 主程序入口
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from gui.gate_window import GateWindow


def main():
    """主函数"""
    print("=" * 50)
    print("Face Recognition Access Control System v2.0")
    print("=" * 50)
    print("Starting system...")
    print("Please ensure camera is connected")
    print("=" * 50)

    # 创建Qt应用
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    # 创建主窗口
    window = GateWindow()
    window.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

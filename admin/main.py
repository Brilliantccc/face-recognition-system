"""
人事管理系统 - 主程序入口
"""

import sys
import os
import multiprocessing

# Windows 下必须在最开始调用 freeze_support()，否则子进程可能异常
multiprocessing.freeze_support()

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from gui.admin_window import AdminWindow


def main():
    """主函数"""
    print("=" * 50)
    print("人脸识别人事管理系统 v2.0")
    print("=" * 50)
    print("正在启动系统...")

    # 创建Qt应用
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    # 设置默认字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 首次运行检查
    try:
        from setup_wizard import check_first_run, SetupWizard
        if check_first_run():
            wizard = SetupWizard()
            wizard.exec_()
    except Exception:
        pass

    # 创建主窗口
    window = AdminWindow()
    window.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

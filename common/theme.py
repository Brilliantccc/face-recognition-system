"""
统一主题模块
提供现代化的 UI 配色、字体和样式
"""

# ============================================================
# 配色方案
# ============================================================

# 主色调
PRIMARY = "#1a73e8"           # 主蓝
PRIMARY_DARK = "#1557b0"      # 深蓝
PRIMARY_LIGHT = "#e8f0fe"     # 浅蓝背景
PRIMARY_HOVER = "#1557b0"     # 悬停蓝

# 辅助色
SUCCESS = "#34a853"           # 成功绿
SUCCESS_LIGHT = "#e6f4ea"     # 浅绿背景
DANGER = "#ea4335"            # 危险红
DANGER_LIGHT = "#fce8e6"      # 浅红背景
WARNING = "#fbbc04"           # 警告橙
WARNING_LIGHT = "#fef7e0"     # 浅橙背景
INFO = "#4285f4"              # 信息蓝
INFO_LIGHT = "#e8f0fe"        # 浅蓝背景

# 中性色
BG_MAIN = "#f0f2f5"           # 主背景灰
BG_CARD = "#ffffff"           # 卡片白
BG_SIDEBAR = "#1e293b"        # 侧边栏深色
BG_INPUT = "#f8f9fa"          # 输入框背景
BORDER = "#e0e0e0"            # 边框色
BORDER_LIGHT = "#f0f0f0"      # 浅边框
TEXT_PRIMARY = "#202124"       # 主文字
TEXT_SECONDARY = "#5f6368"     # 次要文字
TEXT_MUTED = "#9aa0a6"         # 辅助文字
TEXT_WHITE = "#ffffff"         # 白色文字
DIVIDER = "#e8eaed"            # 分隔线

# 门禁专用
GATE_PASS_BG = "#e6f4ea"
GATE_PASS_BORDER = "#34a853"
GATE_PASS_TEXT = "#137333"
GATE_REJECT_BG = "#fce8e6"
GATE_REJECT_BORDER = "#ea4335"
GATE_REJECT_TEXT = "#a50e0e"
GATE_STANDBY_BG = "#f0f2f5"
GATE_STANDBY_BORDER = "#9aa0a6"
GATE_STANDBY_TEXT = "#5f6368"

# 字体
FONT_FAMILY = "'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', sans-serif"
FONT_FAMILY_MONO = "'Consolas', 'Courier New', monospace"


# ============================================================
# 全局 QSS 样式表
# ============================================================

GLOBAL_STYLESHEET = f"""
/* 全局基础 */
* {{
    font-family: {FONT_FAMILY};
}}

QMainWindow {{
    background-color: {BG_MAIN};
}}

QWidget {{
    color: {TEXT_PRIMARY};
}}

/* 按钮 */
QPushButton {{
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid {BORDER};
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    min-height: 20px;
}}
QPushButton:hover {{
    background-color: {PRIMARY_LIGHT};
    border-color: {PRIMARY};
    color: {PRIMARY};
}}
QPushButton:pressed {{
    background-color: {PRIMARY};
    color: white;
}}

/* 输入框 */
QLineEdit {{
    padding: 8px 12px;
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {BG_CARD};
    font-size: 13px;
    color: {TEXT_PRIMARY};
    selection-background-color: {PRIMARY_LIGHT};
}}
QLineEdit:focus {{
    border-color: {PRIMARY};
}}
QLineEdit::placeholder {{
    color: {TEXT_MUTED};
}}

/* 表格 */
QTableWidget {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background-color: {BG_CARD};
    gridline-color: {BORDER_LIGHT};
    font-size: 13px;
    selection-background-color: {PRIMARY_LIGHT};
    selection-color: {TEXT_PRIMARY};
}}
QTableWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid {BORDER_LIGHT};
}}
QTableWidget::item:selected {{
    background-color: {PRIMARY_LIGHT};
    color: {PRIMARY_DARK};
}}
QTableWidget::item:hover {{
    background-color: #f5f8ff;
}}
QHeaderView::section {{
    background-color: {BG_MAIN};
    color: {TEXT_SECONDARY};
    font-weight: bold;
    font-size: 12px;
    padding: 10px 12px;
    border: none;
    border-bottom: 2px solid {BORDER};
    border-right: 1px solid {BORDER_LIGHT};
}}
QHeaderView::section:last {{
    border-right: none;
}}

/* 分组框 */
QGroupBox {{
    font-weight: bold;
    font-size: 14px;
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 14px;
    padding: 18px 12px 12px 12px;
    background-color: {BG_CARD};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: {TEXT_PRIMARY};
}}

/* 列表 */
QListWidget {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background-color: {BG_CARD};
    font-size: 13px;
}}
QListWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid {BORDER_LIGHT};
}}
QListWidget::item:selected {{
    background-color: {PRIMARY_LIGHT};
    color: {PRIMARY};
}}
QListWidget::item:hover {{
    background-color: #f5f8ff;
}}

/* 标签页 */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background-color: {BG_CARD};
    top: -1px;
}}
QTabBar::tab {{
    padding: 10px 24px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
    background-color: {BG_MAIN};
    color: {TEXT_SECONDARY};
}}
QTabBar::tab:selected {{
    background-color: {BG_CARD};
    color: {PRIMARY};
    border-bottom: 2px solid {PRIMARY};
    font-weight: bold;
}}
QTabBar::tab:hover:!selected {{
    background-color: {PRIMARY_LIGHT};
    color: {PRIMARY};
}}

/* 下拉框 */
QComboBox {{
    padding: 8px 12px;
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {BG_CARD};
    font-size: 13px;
    min-width: 120px;
}}
QComboBox:focus {{
    border-color: {PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 30px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SECONDARY};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {BG_CARD};
    selection-background-color: {PRIMARY_LIGHT};
    selection-color: {PRIMARY};
    padding: 4px;
}}

/* 滚动条 */
QScrollBar:vertical {{
    background-color: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {BORDER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background-color: transparent;
    height: 8px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background-color: {BORDER};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {TEXT_MUTED};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* 文本编辑框 */
QTextEdit {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background-color: {BG_CARD};
    font-size: 13px;
    padding: 8px;
    selection-background-color: {PRIMARY_LIGHT};
}}

/* 状态栏 */
QStatusBar {{
    background-color: {BG_CARD};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {BORDER};
    padding: 4px 12px;
    font-size: 12px;
}}

/* 进度条 */
QProgressBar {{
    border: none;
    border-radius: 4px;
    background-color: {BG_MAIN};
    text-align: center;
    height: 8px;
}}
QProgressBar::chunk {{
    border-radius: 4px;
    background-color: {PRIMARY};
}}

/* 对话框 */
QDialog {{
    background-color: {BG_MAIN};
}}

/* 工具提示 */
QToolTip {{
    background-color: {BG_SIDEBAR};
    color: {TEXT_WHITE};
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}
"""


# ============================================================
# 按钮样式工厂
# ============================================================

def btn_primary(extra=""):
    """主要操作按钮（蓝色）"""
    return f"""
        QPushButton {{
            background-color: {PRIMARY};
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            {extra}
        }}
        QPushButton:hover {{
            background-color: {PRIMARY_HOVER};
        }}
        QPushButton:pressed {{
            background-color: #1251a3;
        }}
        QPushButton:disabled {{
            background-color: {BORDER};
            color: {TEXT_MUTED};
        }}
    """


def btn_success(extra=""):
    """成功操作按钮（绿色）"""
    return f"""
        QPushButton {{
            background-color: {SUCCESS};
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            {extra}
        }}
        QPushButton:hover {{
            background-color: #2d9248;
        }}
        QPushButton:pressed {{
            background-color: #237a3a;
        }}
        QPushButton:disabled {{
            background-color: {BORDER};
            color: {TEXT_MUTED};
        }}
    """


def btn_danger(extra=""):
    """危险操作按钮（红色）"""
    return f"""
        QPushButton {{
            background-color: {DANGER};
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            {extra}
        }}
        QPushButton:hover {{
            background-color: #d33426;
        }}
        QPushButton:pressed {{
            background-color: #b71c1c;
        }}
        QPushButton:disabled {{
            background-color: {BORDER};
            color: {TEXT_MUTED};
        }}
    """


def btn_warning(extra=""):
    """警告操作按钮（橙色）"""
    return f"""
        QPushButton {{
            background-color: {WARNING};
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            {extra}
        }}
        QPushButton:hover {{
            background-color: #e6a800;
        }}
        QPushButton:pressed {{
            background-color: #cc9600;
        }}
    """


def btn_info(extra=""):
    """信息按钮（浅蓝）"""
    return f"""
        QPushButton {{
            background-color: {INFO_LIGHT};
            color: {PRIMARY};
            border: 1px solid {PRIMARY};
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            {extra}
        }}
        QPushButton:hover {{
            background-color: {PRIMARY};
            color: white;
        }}
    """


def btn_outline(extra=""):
    """边框按钮（默认灰色）"""
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER};
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            {extra}
        }}
        QPushButton:hover {{
            background-color: {BG_MAIN};
            border-color: {TEXT_MUTED};
        }}
    """


def btn_large_primary():
    """大号主要按钮"""
    return btn_primary("min-height: 44px; font-size: 16px; padding: 12px 32px;")


def btn_large_danger():
    """大号危险按钮"""
    return btn_danger("min-height: 44px; font-size: 16px; padding: 12px 32px;")


# ============================================================
# 卡片样式
# ============================================================

CARD_STYLE = f"""
    QGroupBox {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 20px 16px 16px 16px;
        margin-top: 16px;
        font-size: 14px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 16px;
        padding: 0 8px;
        color: {TEXT_PRIMARY};
    }}
"""


# ============================================================
# 信息卡片样式（用于状态展示）
# ============================================================

def status_card_style(bg_color, border_color):
    """返回状态卡片的样式"""
    return f"""
        background-color: {bg_color};
        border: 2px solid {border_color};
        border-radius: 12px;
        padding: 16px;
        font-size: 14px;
    """

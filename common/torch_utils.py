"""
PyTorch 工具模块
统一管理 PyTorch 的可用性检测、CUDA/CPU 设备选择
支持用户使用 CUDA 版本或 CPU 版本的 PyTorch

支持的情况：
1. PyTorch 未安装 → 提示安装命令
2. PyTorch CUDA 版本 + CUDA 可用 → 使用 GPU
3. PyTorch CUDA 版本 + CUDA 初始化失败 → 降级到 CPU（警告）
4. PyTorch CPU 版本 → 使用 CPU
"""

import os
import sys

# PyTorch 状态
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
torch = None
nn = None
transforms = None
Image = None

# 设备信息
DEVICE = None
GPU_NAME = None
GPU_MEMORY_GB = 0

# PyTorch 类型
TORCH_TYPE = "none"  # "cuda", "cpu", "none"


def init_torch():
    """
    初始化 PyTorch 环境
    检测 PyTorch 是否可用，CUDA 是否可用，并设置全局设备
    """
    global TORCH_AVAILABLE, CUDA_AVAILABLE, torch, nn, transforms
    global DEVICE, GPU_NAME, GPU_MEMORY_GB, Image, TORCH_TYPE

    # 检查是否是主进程（避免 DataLoader worker 重复打印）
    import multiprocessing
    is_main_process = multiprocessing.current_process().name == 'MainProcess'

    # ========== 情况 1: 尝试导入 PyTorch ==========
    try:
        import torch as _torch
        torch = _torch

        import torch.nn as _nn
        nn = _nn

        from torchvision import transforms as _transforms
        transforms = _transforms

        from PIL import Image as _Image
        Image = _Image

        TORCH_AVAILABLE = True
        if is_main_process:
            try:
                print(f"[torch_utils] PyTorch {torch.__version__} loaded")
            except UnicodeEncodeError:
                print(f"[torch_utils] PyTorch loaded")

    except ImportError as e:
        # PyTorch 未安装
        TORCH_AVAILABLE = False
        CUDA_AVAILABLE = False
        DEVICE = None
        TORCH_TYPE = "none"
        try:
            print(f"[torch_utils] PyTorch not installed: {e}")
            print(f"[torch_utils] Install with: pip install torch torchvision")
        except UnicodeEncodeError:
            print("[torch_utils] PyTorch not installed")
            print("[torch_utils] Install with: pip install torch torchvision")
        return False

    except Exception as e:
        # 其他导入错误（如 DLL 初始化失败）
        try:
            print(f"[torch_utils] Error importing PyTorch: {e}")
        except UnicodeEncodeError:
            print("[torch_utils] Error importing PyTorch")

        # 尝试以 CPU 模式重新导入（CUDA DLL 失败时）
        if "DLL" in str(e) or "1114" in str(e) or "c10.dll" in str(e):
            print("[torch_utils] CUDA DLL failed, trying CPU-only mode...")
            try:
                # 强制禁用 CUDA，让 PyTorch 只加载 CPU 后端
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

                # 清除已失败的 torch 模块缓存，让重新导入生效
                for mod_name in list(sys.modules.keys()):
                    if mod_name.startswith("torch"):
                        del sys.modules[mod_name]

                import torch as _torch
                torch = _torch

                import torch.nn as _nn
                nn = _nn

                from torchvision import transforms as _transforms
                transforms = _transforms

                from PIL import Image as _Image
                Image = _Image

                TORCH_AVAILABLE = True
                CUDA_AVAILABLE = False
                DEVICE = torch.device("cpu")
                GPU_NAME = None
                GPU_MEMORY_GB = 0
                TORCH_TYPE = "cpu"
                print(f"[torch_utils] PyTorch {torch.__version__} loaded in CPU mode")
                return True
            except Exception as e2:
                print(f"[torch_utils] CPU fallback also failed: {e2}")
        
        TORCH_AVAILABLE = False
        CUDA_AVAILABLE = False
        DEVICE = None
        TORCH_TYPE = "none"
        return False

    # ========== 情况 2-4: 检测 CUDA ==========
    _detect_cuda()

    return True


def _detect_cuda():
    """
    检测 CUDA 可用性
    情况 2: CUDA 版本 + CUDA 可用 → 使用 GPU
    情况 3: CUDA 版本 + CUDA 失败 → 降级 CPU
    情况 4: CPU 版本 → 使用 CPU
    """
    global CUDA_AVAILABLE, DEVICE, GPU_NAME, GPU_MEMORY_GB, TORCH_TYPE

    # 检查是否是主进程
    import multiprocessing
    is_main_process = multiprocessing.current_process().name == 'MainProcess'

    # 检查 PyTorch 是否编译了 CUDA 支持
    has_cuda_build = hasattr(torch, 'version') and hasattr(torch.version, 'cuda') and torch.version.cuda is not None

    if not has_cuda_build:
        # 情况 4: CPU 版本的 PyTorch
        CUDA_AVAILABLE = False
        DEVICE = torch.device("cpu")
        GPU_NAME = None
        GPU_MEMORY_GB = 0
        TORCH_TYPE = "cpu"
        if is_main_process:
            try:
                print("[torch_utils] CPU-only PyTorch detected")
            except UnicodeEncodeError:
                pass
        return

    # PyTorch 有 CUDA 支持，检查是否可用
    if is_main_process:
        try:
            print(f"[torch_utils] PyTorch compiled with CUDA {torch.version.cuda}")
        except UnicodeEncodeError:
            print("[torch_utils] PyTorch compiled with CUDA")

    try:
        # 尝试 CUDA 基础检测
        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            # 情况 3a: CUDA 驱动/设备不可用
            CUDA_AVAILABLE = False
            DEVICE = torch.device("cpu")
            GPU_NAME = None
            GPU_MEMORY_GB = 0
            TORCH_TYPE = "cpu"
            if is_main_process:
                try:
                    print("[torch_utils] CUDA not available (no GPU or driver issue)")
                    print("[torch_utils] Falling back to CPU mode")
                except UnicodeEncodeError:
                    pass
            return

        # 尝试实际创建 CUDA 张量（更严格的测试）
        test_tensor = torch.zeros(1, device='cuda')
        del test_tensor

        # 情况 2: CUDA 完全可用
        CUDA_AVAILABLE = True
        DEVICE = torch.device("cuda")
        TORCH_TYPE = "cuda"

        try:
            GPU_NAME = torch.cuda.get_device_name(0)
            GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if is_main_process:
                print(f"[torch_utils] CUDA ready: {GPU_NAME} ({GPU_MEMORY_GB:.1f}GB)")
        except Exception as e:
            GPU_NAME = "Unknown GPU"
            GPU_MEMORY_GB = 0
            if is_main_process:
                try:
                    print(f"[torch_utils] CUDA ready (could not get GPU info: {e})")
                except UnicodeEncodeError:
                    print("[torch_utils] CUDA ready")

    except Exception as e:
        # 情况 3b: CUDA 初始化失败（DLL 问题等）
        CUDA_AVAILABLE = False
        DEVICE = torch.device("cpu")
        GPU_NAME = None
        GPU_MEMORY_GB = 0
        TORCH_TYPE = "cpu"
        if is_main_process:
            try:
                print(f"[torch_utils] CUDA initialization failed: {e}")
                print("[torch_utils] Falling back to CPU mode")
            except UnicodeEncodeError:
                print("[torch_utils] CUDA initialization failed, falling back to CPU")
        GPU_MEMORY_GB = 0
        TORCH_TYPE = "cpu"


def get_device(prefer_gpu: bool = True):
    """
    获取计算设备
    :param prefer_gpu: 是否优先使用 GPU（如果可用）
    :return: torch.device 或 None
    """
    if not TORCH_AVAILABLE:
        return None

    # 检查配置文件中的设置
    try:
        from common.config import TRAINING_DEVICE
        if TRAINING_DEVICE == "cpu":
            return torch.device("cpu")
        elif TRAINING_DEVICE == "cuda":
            if CUDA_AVAILABLE:
                return torch.device("cuda")
            else:
                print("[torch_utils] Warning: CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
    except ImportError:
        pass

    # 默认行为
    if prefer_gpu and CUDA_AVAILABLE:
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    获取设备信息
    :return: 设备信息字典
    """
    return {
        "torch_available": TORCH_AVAILABLE,
        "torch_type": TORCH_TYPE,
        "cuda_available": CUDA_AVAILABLE,
        "device": str(DEVICE) if DEVICE else "N/A",
        "gpu_name": GPU_NAME,
        "gpu_memory_gb": GPU_MEMORY_GB,
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__ if torch else None,
        "cuda_version": torch.version.cuda if torch and hasattr(torch, 'version') else None
    }


def get_device_display() -> str:
    """
    获取设备显示信息（用于界面显示）
    :return: 设备描述字符串
    """
    if not TORCH_AVAILABLE:
        return "PyTorch not installed"
    elif CUDA_AVAILABLE:
        return f"GPU: {GPU_NAME} ({GPU_MEMORY_GB:.1f}GB)"
    elif TORCH_TYPE == "cpu":
        return f"CPU mode (cores: {os.cpu_count()})"
    else:
        return f"CPU mode (CUDA failed, cores: {os.cpu_count()})"


def get_install_hint() -> str:
    """
    获取 PyTorch 安装提示
    :return: 安装命令提示
    """
    if TORCH_AVAILABLE:
        if TORCH_TYPE == "cuda":
            return "PyTorch GPU version is installed and working."
        elif TORCH_TYPE == "cpu":
            return (
                "PyTorch CPU version is installed.\n\n"
                "To use GPU, reinstall with CUDA support:\n"
                "pip uninstall torch torchvision\n"
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            )
        else:
            return "PyTorch is installed but CUDA failed. Using CPU mode."
    else:
        return (
            "PyTorch is not installed. Install with:\n\n"
            "GPU version (recommended, requires NVIDIA GPU):\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n\n"
            "CPU version:\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )


def load_model_checkpoint(model_path: str, map_location=None):
    """
    加载模型 checkpoint（自动处理设备映射）
    :param model_path: 模型文件路径
    :param map_location: 设备映射（None 则自动选择）
    :return: checkpoint 字典
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")

    if map_location is None:
        # 加载时映射到 CPU，避免 CUDA 不可用时出错
        map_location = DEVICE if CUDA_AVAILABLE else "cpu"

    return torch.load(model_path, map_location=map_location)


def get_status_summary() -> str:
    """
    获取状态摘要（用于显示或日志）
    :return: 状态摘要字符串
    """
    if not TORCH_AVAILABLE:
        return "PyTorch: NOT INSTALLED"

    lines = [
        f"PyTorch: {torch.__version__}",
        f"Build CUDA: {torch.version.cuda if hasattr(torch, 'version') and torch.version.cuda else 'None'}",
        f"Runtime CUDA: {'Available' if CUDA_AVAILABLE else 'Not Available'}",
        f"Device: {DEVICE}",
    ]

    if CUDA_AVAILABLE:
        lines.append(f"GPU: {GPU_NAME}")
        lines.append(f"GPU Memory: {GPU_MEMORY_GB:.1f} GB")

    return " | ".join(lines)


# 模块初始化时自动检测
init_torch()


# 导出常用符号
__all__ = [
    'TORCH_AVAILABLE', 'CUDA_AVAILABLE', 'DEVICE', 'TORCH_TYPE',
    'GPU_NAME', 'GPU_MEMORY_GB',
    'torch', 'nn', 'transforms', 'Image',
    'init_torch', 'get_device', 'get_device_info',
    'get_device_display', 'get_install_hint',
    'load_model_checkpoint', 'get_status_summary'
]

from loguru import logger
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"  # 日志文件名


def setup_logger():
    """配置 loguru：控制台 INFO（彩色）+ 文件 DEBUG（滚动）。"""
    logger.remove()  # 移除默认 handler，避免重复输出
    logger.add(sys.stderr, level='INFO', colorize=True)
    logger.add(LOG_FILE, level='DEBUG', rotation='10 MB', encoding='utf-8')
    return logger


setup_logger()  # 初始化日志配置
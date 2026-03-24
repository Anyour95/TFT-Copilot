import pyautogui
import random
import time
from typing import Tuple, Optional

class KeyboardController:
    """
    键盘控制器，封装 pyautogui 的键盘操作，加入随机延迟模拟人类输入习惯。
    """

    def __init__(self, random_delay: Tuple[float, float] = (0.02, 0.08)):
        """
        初始化键盘控制器
        :param random_delay: 每次操作后的随机等待时间范围 (min, max)，秒
        """
        self.random_delay = random_delay

    def _random_sleep(self):
        """随机睡眠一小段时间，模拟操作间隙"""
        if self.random_delay:
            time.sleep(random.uniform(*self.random_delay))

    def write(self, text: str, interval: float = 0.05):
        """
        输入文本，每个字符间有随机微小延迟（模拟打字）
        :param text: 要输入的字符串
        :param interval: 每个字符间的基准间隔（秒），实际会加入随机抖动
        """
        # 加入随机抖动：实际间隔 = interval * (0.8~1.2)
        for ch in text:
            pyautogui.write(ch)
            actual_interval = interval * random.uniform(0.8, 1.2)
            time.sleep(actual_interval)
        self._random_sleep()

    def press(self, key: str):
        """
        按下并释放单个按键
        :param key: 键名，如 'enter', 'a', 'f1' 等
        """
        # 按键前微小随机延迟
        time.sleep(random.uniform(0.01, 0.03))
        pyautogui.press(key)
        self._random_sleep()

    def hotkey(self, *keys):
        """
        按下组合键，例如 hotkey('ctrl', 'c')
        :param keys: 键名列表
        """
        pyautogui.hotkey(*keys)
        self._random_sleep()

    def key_down(self, key: str):
        """
        按下按键（不释放）
        """
        pyautogui.keyDown(key)
        self._random_sleep()

    def key_up(self, key: str):
        """
        释放按键
        """
        pyautogui.keyUp(key)
        self._random_sleep()

    def hold(self, key: str, duration: float = 0.1):
        """
        按住按键一段时间后释放
        """
        self.key_down(key)
        time.sleep(duration)
        self.key_up(key)
        self._random_sleep()
import pyautogui
import random
import time
from typing import Tuple, Optional

pyautogui.FAILSAFE = False  # 可选，关闭自动防故障

class MouseController:
    """
    鼠标控制器，封装 pyautogui 的鼠标操作，加入拟人化随机延迟和移动轨迹。
    """

    def __init__(self,
                 random_delay: Tuple[float, float] = (0.02, 0.08),
                 move_duration: float = 0.2,
                 use_human_move: bool = True):
        """
        初始化鼠标控制器
        :param random_delay: 每次操作后的随机等待时间范围 (min, max)，秒
        :param move_duration: 鼠标移动的默认耗时（秒）
        :param use_human_move: 是否使用带随机偏移的移动轨迹
        """
        self.random_delay = random_delay
        self.move_duration = move_duration
        self.use_human_move = use_human_move
        self.screen_width, self.screen_height = pyautogui.size()

    def _random_sleep(self):
        """随机睡眠一小段时间，模拟操作间隙"""
        if self.random_delay:
            time.sleep(random.uniform(*self.random_delay))

    def _to_absolute_coords(self,
                            x: Optional[int] = None,
                            y: Optional[int] = None,
                            x_percent: Optional[float] = None,
                            y_percent: Optional[float] = None) -> Tuple[int, int]:
        """
        将不同坐标表示转换为绝对像素坐标
        :param x: 绝对X坐标
        :param y: 绝对Y坐标
        :param x_percent: 屏幕宽度百分比 (0~1)
        :param y_percent: 屏幕高度百分比 (0~1)
        :return: (abs_x, abs_y)
        """
        if x_percent is not None:
            abs_x = int(x_percent * self.screen_width)
        elif x is not None:
            abs_x = int(x)
        else:
            raise ValueError("必须提供 x 或 x_percent")

        if y_percent is not None:
            abs_y = int(y_percent * self.screen_height)
        elif y is not None:
            abs_y = int(y)
        else:
            raise ValueError("必须提供 y 或 y_percent")

        return abs_x, abs_y

    def _human_move(self, x: int, y: int, duration: Optional[float] = None):
        """
        使用贝塞尔曲线或随机抖动进行拟人化移动（简化版：先到附近偏移点，再微调）
        """
        if duration is None:
            duration = self.move_duration

        # 随机在目标周围产生偏移，让鼠标“略微过头”再回来
        offset_x = random.randint(-5, 5)
        offset_y = random.randint(-5, 5)
        near_x = x + offset_x
        near_y = y + offset_y
        # 先移动到附近偏移点
        pyautogui.moveTo(near_x, near_y, duration=duration * 0.7, tween=pyautogui.easeInOutQuad)
        # 再微调到精确位置
        pyautogui.moveTo(x, y, duration=duration * 0.3, tween=pyautogui.easeOutQuad)

    def move_to(self,
                x: Optional[int] = None,
                y: Optional[int] = None,
                x_percent: Optional[float] = None,
                y_percent: Optional[float] = None,
                duration: Optional[float] = None):
        """
        移动鼠标到指定位置
        """
        abs_x, abs_y = self._to_absolute_coords(x, y, x_percent, y_percent)

        if self.use_human_move:
            self._human_move(abs_x, abs_y, duration)
        else:
            pyautogui.moveTo(abs_x, abs_y, duration=duration or self.move_duration)

        self._random_sleep()

    def click(self,
              button: str = 'left',
              clicks: int = 1,
              interval: float = 0.1,
              x: Optional[int] = None,
              y: Optional[int] = None,
              x_percent: Optional[float] = None,
              y_percent: Optional[float] = None):
        """
        鼠标点击
        :param button: 'left', 'right', 'middle'
        :param clicks: 点击次数
        :param interval: 连点间隔（秒）
        :param x, y: 点击前移动到的绝对坐标
        :param x_percent, y_percent: 点击前移动到的屏幕百分比坐标
        """
        if x is not None or x_percent is not None:
            self.move_to(x, y, x_percent, y_percent, duration=0.1)

        # 点击前随机微小延迟
        time.sleep(random.uniform(0.01, 0.05))
        pyautogui.click(button=button, clicks=clicks, interval=interval)
        self._random_sleep()

    def drag(self,
             start_x: Optional[int] = None,
             start_y: Optional[int] = None,
             end_x: Optional[int] = None,
             end_y: Optional[int] = None,
             end_x_percent: Optional[float] = None,
             end_y_percent: Optional[float] = None,
             button: str = 'left',
             duration: float = 0.5):
        """
        拖拽操作
        :param start_x, start_y: 起始绝对坐标（可选，不提供则从当前位置开始）
        :param end_x, end_y: 终点绝对坐标
        :param end_x_percent, end_y_percent: 终点屏幕百分比
        :param button: 鼠标按键
        :param duration: 拖拽耗时（秒）
        """
        if start_x is not None and start_y is not None:
            self.move_to(start_x, start_y, duration=0.1)

        # 计算终点
        if end_x_percent is not None:
            end_x = int(end_x_percent * self.screen_width)
        if end_y_percent is not None:
            end_y = int(end_y_percent * self.screen_height)
        if end_x is None or end_y is None:
            raise ValueError("必须提供终点坐标")

        pyautogui.dragTo(end_x, end_y, duration=duration, button=button)
        self._random_sleep()

    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None):
        """
        滚动鼠标滚轮
        :param clicks: 滚动格数（正数向上，负数向下）
        :param x, y: 滚动前移动鼠标到指定位置（可选）
        """
        if x is not None and y is not None:
            self.move_to(x, y)
        pyautogui.scroll(clicks)
        self._random_sleep()

    def screenshot(self, region: Optional[Tuple[int, int, int, int]] = None, filename: str = None):
        """
        截图
        :param region: 区域 (left, top, width, height)
        :param filename: 保存文件名（若提供则保存，否则返回Image对象）
        :return: 如果 filename 为空则返回 Image 对象
        """
        if region:
            img = pyautogui.screenshot(region=region)
        else:
            img = pyautogui.screenshot()
        if filename:
            img.save(filename)
        return img

    def wait(self, seconds: float = None, random_range: Tuple[float, float] = None):
        """
        等待指定时间，如果提供了 random_range 则忽略 seconds，用于操作间隙。
        """
        if random_range:
            wait_time = random.uniform(*random_range)
        elif seconds is not None:
            wait_time = seconds
        else:
            wait_time = random.uniform(0.1, 0.3)
        time.sleep(wait_time)
import pygame
import sys
import flappy_UAV  # 引入 flappy_UAV.py
import flappy_UAV_ga_train  # 引入 flappy_UAV_ga_train.py
import flappy_UAV_ga_after_train  # 引入 flappy_UAV_ga_after_train.py
import os

from flappy_UAV_ga_train import Linear_Net
from flappy_UAV_ga_after_train import Linear_Net

import os
import sys
# 1

# 获取资源文件的绝对路径
def resource_path(relative_path):
    """用于获取资源文件的路径，适配 PyInstaller 打包后的路径"""
    if hasattr(sys, '_MEIPASS'):  # PyInstaller 打包后的临时目录
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# 初始化 pygame
pygame.init()

# 窗口设置
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600  # 增大窗口尺寸
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy UAV Selector")

# 设置窗口位置为屏幕中央
os.environ['SDL_VIDEO_CENTERED'] = '1'

# 加载背景图片
bg_image = pygame.image.load(resource_path("resources/bg1.gif"))  # 确保 bg1.gif 在 resources 目录下
bg_image = pygame.transform.scale(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))  # 调整背景图片大小

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PRIMARY_COLOR = (25, 25, 112)  # 午夜蓝
HOVER_COLOR = (65, 105, 225)   # 皇家蓝

# 字体
font = pygame.font.Font(None, 36)

# 定义按钮
button_width, button_height = 200, 50
button_spacing = 50  # 按钮之间的间距

# 计算按钮的起始x坐标，使三个按钮在屏幕中居中
total_width = (button_width * 3) + (button_spacing * 2)
start_x = (SCREEN_WIDTH - total_width) // 2

# 按钮类
class Button:
    def __init__(self, text, x, y, width, height, color, hover_color, action):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.hover_color = hover_color
        self.action = action
        self.pressed = False

    def draw(self, surface, mouse_pos):
        offset = 0
        if self.is_hovered(mouse_pos):
            if self.pressed:
                offset = 2  # 按下时的偏移
            else:
                offset = -2  # 悬停时的偏移

        # 绘制按钮阴影
        shadow_rect = pygame.Rect(self.x + 3, self.y + 3, self.width, self.height)
        pygame.draw.rect(surface, (0, 0, 0, 100), shadow_rect, border_radius=10)

        # 绘制按钮主体
        button_rect = pygame.Rect(self.x + offset, self.y + offset, self.width, self.height)
        if self.is_hovered(mouse_pos):
            pygame.draw.rect(surface, self.hover_color, button_rect, border_radius=10)
        else:
            pygame.draw.rect(surface, self.color, button_rect, border_radius=10)

        # 绘制按钮边框
        border_rect = pygame.Rect(self.x + offset, self.y + offset, self.width, self.height)
        pygame.draw.rect(surface, (255, 255, 255, 50), border_rect, 2, border_radius=10)
        
        # 绘制文本
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.x + self.width // 2 + offset, 
                                                self.y + self.height // 2 + offset))
        surface.blit(text_surface, text_rect)

    def is_hovered(self, mouse_pos):
        return (self.x <= mouse_pos[0] <= self.x + self.width and 
                self.y <= mouse_pos[1] <= self.y + self.height)

    def click(self, mouse_pos):
        if self.is_hovered(mouse_pos):
            self.pressed = True
            self.action()
        else:
            self.pressed = False


# 切换到 flappy_UAV（人工模式）
def play_manual_mode():
    flappy_UAV.main()  # 调用 flappy_UAV 的游戏逻辑


# 切换到 flappy_UAV_ga_train（训练模式）
def play_training_mode():
    flappy_UAV_ga_train.main()  # 调用 flappy_UAV_ga_train 的游戏逻辑


# 切换到 flappy_UAV_ga_after_train（训练后模式）
def play_after_training_mode():
    flappy_UAV_ga_after_train.main()  # 调用 flappy_UAV_ga_after_train 的游戏逻辑


# 主函数
def main_menu():
    clock = pygame.time.Clock()

    # 定义按钮
    button_manual = Button("Manual Mode", 
                          start_x, 
                          SCREEN_HEIGHT // 2, 
                          button_width, 
                          button_height, 
                          PRIMARY_COLOR, 
                          HOVER_COLOR, 
                          play_manual_mode)

    button_training = Button("Train Mode", 
                            start_x + button_width + button_spacing, 
                            SCREEN_HEIGHT // 2, 
                            button_width, 
                            button_height, 
                            PRIMARY_COLOR, 
                            HOVER_COLOR, 
                            play_training_mode)

    button_after_training = Button("After Train Mode", 
                                  start_x + (button_width + button_spacing) * 2, 
                                  SCREEN_HEIGHT // 2, 
                                  button_width, 
                                  button_height, 
                                  PRIMARY_COLOR, 
                                  HOVER_COLOR, 
                                  play_after_training_mode)

    while True:
        screen.blit(bg_image, (0, 0))  # 绘制背景图片
        mouse_pos = pygame.mouse.get_pos()  # 获取鼠标位置

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    button_manual.click(mouse_pos)
                    button_training.click(mouse_pos)
                    button_after_training.click(mouse_pos)

        # 绘制按钮
        button_manual.draw(screen, mouse_pos)
        button_training.draw(screen, mouse_pos)
        button_after_training.draw(screen, mouse_pos)

        # 刷新屏幕
        pygame.display.flip()
        clock.tick(60)


# 启动主菜单
if __name__ == "__main__":
    main_menu()
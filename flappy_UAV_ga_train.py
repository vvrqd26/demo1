import pygame, sys                         # pygame: for game logic and rendering; sys: for exiting the game
from pygame.locals import *                # Import useful constants like QUIT, etc.
import random                              # Random number generation
import numpy as np                         # For numerical operations and arrays
import torch.nn.functional as F            # Functional interface for PyTorch (activation functions, etc.)
import torch                               # PyTorch for AI modeling
import torch.nn as nn                      # Neural Network module from PyTorch
from copy import deepcopy                  # To copy neural network weights deeply
import math 
import os

import os
import sys

# 获取资源文件的绝对路径
def resource_path(relative_path):
    """用于获取资源文件的路径，适配 PyInstaller 打包后的路径"""
    if hasattr(sys, '_MEIPASS'):  # PyInstaller 打包后的临时目录
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Define a simple Neural Network for controlling the bird (AI logic)
class Linear_Net(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):  # Define network architecture
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)      # First layer: input → hidden
        self.linear2 = nn.Linear(hidden_size, output_size)     # Second layer: hidden → output

    def forward(self, x):                 # Forward pass logic
        x = F.relu(self.linear1(x))      # Apply ReLU on output of first layer
        x = self.linear2(x)
        return x

    def get_weight(self):                # Get model weights (for genetic algorithm)
        return deepcopy([self.linear1.weight.data,
				self.linear1.bias.data,
				self.linear2.weight.data,
				self.linear2.bias.data])

    def set_weight(self,weights):        # Set model weights (for genetic algorithm)
        weights = deepcopy(weights)
        self.linear1.weight = nn.Parameter(weights[0])
        self.linear1.bias = nn.Parameter(weights[1])
        self.linear2.weight = nn.Parameter(weights[2])
        self.linear2.bias = nn.Parameter(weights[3])



class Bird(pygame.sprite.Sprite):
    
    jump_size = 10                    
    hidden_size = 128                 # Number of neurons in hidden layer
    output_size = 2                  # Output size (2 choices: flap or not flap)
    
    def __init__(self, x, y,with_cargo=True):
        super().__init__()
        
        self.images_with_cargo = []  # UAV 1-3 (with cargo)
        self.images_without_cargo = []  # UAV 4-6 (without cargo)
        
        self.index = 0
        self.counter = 0
        self.vel = 0
        self.cap = 9 # Max upward velocity
        self.a = 0.5
        self.failed = False
        
        # Load images for both states
        for num in range(1, 4):
            img = pygame.image.load(resource_path(f"resources/UAV_{num}.png"))
            self.images_with_cargo.append(img)
        for num in range(4, 7):
            img = pygame.image.load(resource_path(f"resources/UAV_{num}.png"))
            self.images_without_cargo.append(img)
        
        if with_cargo==True:
            self.current_images = self.images_with_cargo  # Start without cargo
            self.has_cargo = True  # Start without cargo
        else:
            self.current_images = self.images_without_cargo  # Start without cargo
            self.has_cargo = False  # Start without cargo

        self.image = self.current_images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.wing = pygame.mixer.Sound(resource_path('resources/wing.wav'))

        self.cargo =None
        self.fitness = 0          # Fitness score for AI  适应度
        self.score = 0               # Score (how many pipes passed)
        self.model_jump = Linear_Net(Bird.jump_size, Bird.hidden_size, Bird.output_size)  # Neural network model
        
        self.fitness_result = []
        self.state_result = []
        self.jump_result = []
        
        self.missed_cargos = 0
        self.missed_boxes = 0
        self.cumsum = 0
        self.cumsum1 = 0

    def handle_action(self,action):           # If AI chooses action=1 → flap
        if action == 1:
            self.vel = -1 * self.cap          # Negative velocity (goes up)
            self.wing.play()                  # Play flap sound

    # 做出的动作
    def get_action(self, state_jump):              # Predict action based on state (AI decision)
        prediction = self.model_jump(torch.Tensor(state_jump))  # Forward pass with state
        prediction = prediction.detach().numpy().squeeze()  # Detach from PyTorch graph
        jump = prediction.argmax()            # Choose the action with the highest score
        self.jump_result.append(jump)
        return jump

    # 吸收参数
    def get_state_jump(self, observed):            # Get normalized state (input to neural net)
        state = np.array([
         # UAV
			int(self.vel)/self.cap,  # Normalized velocity
         int(self.has_cargo),
            
         # Pipe
			(observed['pipe_dist_top'] - self.rect.bottom)/Pipe.pipe_gap,
         (observed['pipe_dist_centerx']-100)/Pipe.pipe_gap,
         observed['pipe_pickup'],
         observed['pipe_dropoff'],
         
         # Air balloon
         (self.rect.top-observed['balloon_dist_bottom'])/Pipe.pipe_gap,
         (observed['balloon_dist_centerx']-100)/Pipe.pipe_gap,  

         # Obird
         (observed['obird_dist_top']-self.rect.bottom)/Pipe.pipe_gap,
         (observed['obird_dist_centerx']-100)/Pipe.pipe_gap,
         ], dtype=float)
        self.state_result.append(state)
        return state

    # 计算适应度
    def get_fitness(self, observed,group):         # Increase fitness if bird is within pipe opening
        self.fitness_result.append(0)
        self.fitness+=0.000001
        # 通过适应度
        ## Pipe
        if (observed['pipe_dist_top'] - self.rect.bottom)>0 and self.rect.top>30:
            fitness_unit = 1
            self.fitness += fitness_unit
            self.fitness_result[-1]+= fitness_unit
            
        ## Balloon
        #if (observed['balloon_dist_right']<observed['pipe_dist_right']) and (observed['balloon_dist_right']<observed['obird_dist_right']):
        t1 = (observed['balloon_dist_centerx']-self.rect.centerx)/HotAirBalloon.scroll_speed
        if self.vel+0.5*t1<=8:
            d1 = self.vel*t1+0.5*0.5*t1*t1
        elif self.vel+0.5*t1>8:
            t2 = (8-self.vel)/0.5
            d1 = self.vel*t2+0.5*0.5*t2*t2+8*(t1-t2)
        d2 = observed['balloon_dist_bottom'] - self.rect.top
        
        if d2-d1<-60:
            fitness_unit = 1
            self.fitness += fitness_unit
            self.fitness_result[-1]+= fitness_unit
                
            # if (self.rect.top-observed['balloon_dist_bottom'])>0 and self.rect.bottom<500:
            #     fitness_unit = 1
            #     self.fitness += fitness_unit
            #     self.fitness_result[-1]+= fitness_unit
            
        ## Obird
        #if (observed['obird_dist_right']<observed['pipe_dist_right']) and (observed['obird_dist_right']<observed['balloon_dist_right']):
        t1 = (observed['obird_dist_centerx']-self.rect.centerx)/ObstacleBird.scroll_speed
        if self.vel+0.5*t1<=8:
            d1 = self.vel*t1+0.5*0.5*t1*t1
        elif self.vel+0.5*t1>8:
            t2 = (8-self.vel)/0.5
            d1 = self.vel*t2+0.5*0.5*t2*t2+8*(t1-t2)
        
        d2 = observed['obird_dist_top']-20 - self.rect.centery
        if d2 > 0:
            if d2-d1>100:
                fitness_unit = 1
                self.fitness += fitness_unit
                self.fitness_result[-1]+= fitness_unit
        if d2 < 0:
            if d2-d1<-100:
                fitness_unit = 1
                self.fitness += fitness_unit
                self.fitness_result[-1]+= fitness_unit
                
        if ((observed['obird_dist_top']- self.rect.bottom> 0) and self.rect.top>30) or ((self.rect.top-observed['obird_dist_bottom']> 0) and self.rect.bottom<500):
            fitness_unit = 1
            self.fitness += fitness_unit
            self.fitness_result[-1]+= fitness_unit

        # if (observed['pipe_dist_top'] - self.rect.bottom > 0) and (self.rect.top-observed['balloon_dist_bottom']>0) and ((observed['obird_dist_top']- self.rect.bottom> 0) or (self.rect.top-observed['obird_dist_bottom']> 0)):
        #     # fitness_unit = 10/(observed['pipe_dist_right'] - self.rect.left)+10/(observed['balloon_dist_right'] - self.rect.left)+10/(observed['obird_dist_right'] - self.rect.left)
        #     fitness_unit = 1
        #     self.fitness += fitness_unit
        #     self.fitness_result[-1] += (fitness_unit)
        # else:
        #     print('通过不适')
        #     self.fitness += 0
        #     self.fitness_result.append(0)
        
        # 取送适应度
        if (observed['pipe_pickup']==1) and (self.has_cargo == False):
            t1 = (observed['pipe_dist_centerx']-self.rect.centerx)/Pipe.scroll_speed
            if self.vel+self.a*t1<=8:
                d1 = self.vel*t1+0.5*self.a*t1*t1
            elif self.vel+self.a*t1>8:
                t2 = (8-self.vel)/self.a
                d1 = self.vel*t2+0.5*self.a*t2*t2+8*(t1-t2)
            d2 = observed['pipe_dist_top'] - self.rect.bottom
            if (d2-d1<80) and (d2-d1>0):
                self.cumsum += 2
                fitness_unit = self.cumsum/(self.missed_cargos+0.5)
                self.fitness += fitness_unit
                self.fitness_result[-1]+= fitness_unit
            else:
                self.cumsum = 0

        elif (observed['pipe_dropoff']==1) and (observed['pipe_pickup']==0) and (self.has_cargo == True):
            t1 = (observed['pipe_dist_centerx']-self.rect.centerx)/Pipe.scroll_speed
            if self.vel+self.a*t1<=8:
                d1 = self.vel*t1+0.5*self.a*t1*t1
            elif self.vel+self.a*t1>8:
                t2 = (8-self.vel)/self.a
                d1 = self.vel*t2+0.5*self.a*t2*t2+8*(t1-t2)
            d2 = observed['pipe_dist_top'] - self.rect.bottom
            if (d2-d1<80) and (d2-d1>0):
                self.cumsum1 += 2
                fitness_unit = self.cumsum1/(self.missed_boxes+0.5)
                self.fitness += fitness_unit
                self.fitness_result[-1]+= fitness_unit

                if self.has_cargo and not self.failed and self.vel>0:
                    cargo = Cargo(self.rect.centerx, self.rect.centery, self.vel, self)
                    group.add(cargo)
                    self.has_cargo = False
                    self.cargo = cargo
            else:
                self.cumsum1 = 0

        # 掷物适应度
        if self.cargo is not None:
            if (observed['pipe_dropoff']==1):
                x1,y1,v1,a1 = self.cargo.rect.centerx,self.cargo.rect.centery,self.cargo.vel,self.cargo.a
                x2,x3 = observed['pipe_dist_right']-(70-(70-Pipe.delivery_zone_width)/2),observed['pipe_dist_right']-(70-Pipe.delivery_zone_width)/2
                y2 = observed['pipe_dist_top']
                v2 = Pipe.scroll_speed
                delta = v1**2 - 2 * a1 * (y1 - y2)
                if delta > 0:
                    sqrt_delta = math.sqrt(delta)
                    t = (-v1 + sqrt_delta) / a1
                    x_left = x2 - v2 * t
                    x_right = x3 - v2 * t
                    
                    if x_left <= x1 <= x_right:
                        self.fitness += 5
                        self.fitness_result[-1]+= 5
                    else:
                        self.fitness -= 1
                        self.fitness_result[-1]-= 1
            else:
                self.fitness -= 1
                self.fitness_result[-1]-= 1
                    
            
    def modify_fitness_from_cargo(self,s):
        self.fitness += s
        self.fitness_result[-1] += s
        
    #     # print('1',observed['pipe_dist_top'] - self.rect.bottom)
    #     # print('2',self.rect.top-observed['balloon_dist_bottom'])
    #     # print('3',observed['obird_dist_top']- self.rect.bottom)
    #     # if self.score>0:
    #     #     self.fitness += self.score/10
        
        
    def touch_ground(self):
        return self.rect.bottom >= Game.ground_y

    def animation(self):
        flap_cooldown = 5
        self.counter += 1
        if self.counter > flap_cooldown:
            self.counter = 0
            self.index = (self.index + 1) % 3
            self.image = self.current_images[self.index]
        self.image = pygame.transform.rotate(self.current_images[self.index], self.vel * 0)

    def check_collision(self):             # Check if bird hits top or bottom of the screen
        if self.rect.top < 0 or self.rect.bottom >= Game.ground_y:
            self.failed = True
            
    def update(self,action,group):
        self.vel += self.a                    # Gravity effect
        if self.vel > 8:                   # Limit downward speed
            self.vel = 8
        
        if not self.touch_ground():
            self.rect.y += int(self.vel)
        
        # Update current images based on cargo state
        if self.has_cargo:
            self.current_images = self.images_with_cargo
        else:
            self.current_images = self.images_without_cargo
    
        if not self.failed:
            self.handle_action(action)
            self.animation()
            
        self.check_collision()             # See if bird crashed

class Pipe(pygame.sprite.Sprite):
    scroll_speed = 4
    pipe_gap = 180
    delivery_zone_height = 70
    delivery_zone_width = 70
    
    def __init__(self, x, y,state='empty'):
        super().__init__()
        self.passed = False
        self.state = state
        self.images = {
            'empty': [pygame.image.load(resource_path(f"resources/building{num}.png")) for num in ( 1,2,3)],
            'pickup': [pygame.image.load(resource_path(f"resources/building{num}.png")) for num in ( 7,8,9)],
            'dropoff': [pygame.image.load(resource_path(f"resources/building{num}.png")) for num in ( 4,5,6)],
        }
        self.type = random.randint(0, 2)
        self.image = self.images[self.state][self.type]
        self.rect = self.image.get_rect() 
        self.rect.bottomleft = [x, y]

    def set_state(self, state):
        # 保存当前rect的关键属性
        old_left = self.rect.left
        old_bottom = self.rect.bottom
        
        # 更新状态和图片
        self.state = state
        self.image = self.images[self.state][self.type]
        
        # 重新设置rect，保持底部和左侧位置不变
        self.rect = self.image.get_rect()
        self.rect.left = old_left    # 保持x坐标不变
        self.rect.bottom = old_bottom  # 保持底部位置不变

    def get_delivery_zone(self):
        """返回更小的取货/送货安全区"""
        return pygame.Rect(
            self.rect.centerx - Pipe.delivery_zone_width // 2,  # 居中X位置
            self.rect.y - Pipe.delivery_zone_height//2,            # 顶部上方50px
            Pipe.delivery_zone_width,                           # 宽度30px
            Pipe.delivery_zone_height                           # 高度50px
        )
        
    def update(self):
        self.rect.x -= Pipe.scroll_speed
        if self.rect.right < 0:
            self.kill()


class Cargo(pygame.sprite.Sprite):
    def __init__(self, x, y, speed, bird):
        super().__init__()
        self.image = pygame.image.load(resource_path("resources/cargo.png"))
        self.rect = self.image.get_rect(center=(x, y))
        self.vel = speed
        self.a = 0.5
        self.bird = bird

    def touch_ground(self):
        return self.rect.bottom > Game.ground_y

    def update(self):
        self.vel += self.a
        if self.vel > 8:
            self.vel = 8
        if not self.touch_ground():
            self.rect.y += int(self.vel)

class HotAirBalloon(pygame.sprite.Sprite):
    scroll_speed = 3

    def __init__(self, x, y):
        super().__init__()
        self.passed = False
        original_image = pygame.image.load(resource_path("resources/hot_air_balloon.png"))
        new_width = original_image.get_width() // 4
        new_height = original_image.get_height() // 4
        self.image = pygame.transform.scale(original_image, (new_width, new_height))
        self.rect = self.image.get_rect()
        self.rect.topleft = [x, y]
        
        # 添加更小的碰撞检测矩形（缩小到原尺寸的80%）
        self.hitbox = pygame.Rect(0, 0, new_width*0.5, new_height*0.5)
        self.hitbox.center = self.rect.center

    def update(self):
        self.rect.x -= HotAirBalloon.scroll_speed
        self.hitbox.center = self.rect.center  # 同步更新hitbox位置
        if self.rect.right < 0:
            self.kill()

class ObstacleBird(pygame.sprite.Sprite):
    scroll_speed = 5

    def __init__(self, x, y):
        super().__init__()
        self.passed = False
        original_image = pygame.image.load(resource_path("resources/obstacle_bird1.png"))
        new_width = original_image.get_width() // 8
        new_height = original_image.get_height() // 8
        self.image = pygame.transform.scale(original_image, (new_width, new_height))
        self.rect = self.image.get_rect()
        self.rect.topleft = [x, y]
        
        # 添加更小的碰撞检测矩形（缩小到原尺寸的90%）
        self.hitbox = pygame.Rect(0, 0, new_width*0.7, new_height*0.7)
        self.hitbox.center = self.rect.center
        
    def update(self):
        self.rect.x -= ObstacleBird.scroll_speed
        self.hitbox.center = self.rect.center  # 同步更新hitbox位置
        if self.rect.right < 0:
            self.kill()

class Game():
    ground_y = 650
    parameter_len_jump = (Bird.jump_size+1)*Bird.hidden_size+(Bird.hidden_size+1)*Bird.output_size

    def __init__(self, Width=1200, Height=800):
        pygame.init()
        self.Win_width, self.Win_height = (Width, Height)
        self.surface = pygame.display.set_mode((self.Win_width, self.Win_height))
        # 设置窗口位置为屏幕中央
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.display.set_caption('Flappy UAV Training')
        self.ground_x = 0
        self.score = 0
        self.pipe_counter = 0
        
        self.observed = dict()
        self.Clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Futura', 50)
        self.images = self.loadImages()
        self.sounds = self.loadSounds()
        
        self.n_generations = 100 # 最大进化代数
        self.generation_size = 50  # 每一代鸟的数量
        self.weights_jump = [] # 当前代每只鸟的模型参数
        self.fitness = [] # 当前代每只鸟的适应度
        
        self.Bird_fitness_result = []
        self.Bird_state_result = []
        self.Bird_jump_result = []
        self.Bird_model_jump = []
        
        self.fps = 60
        self.hot_air_balloon_timer = 0
        self.obstacle_bird_timer = 0
    
        self.pipe_group = pygame.sprite.Group()
        self.bird_group = pygame.sprite.Group()
        self.cargo_group = pygame.sprite.Group()
        self.hot_air_balloon_group = pygame.sprite.Group()
        self.obstacle_bird_group = pygame.sprite.Group()
        
        self.new_birds() # 创建鸟
        self.new_pipes(time=0, initial_offset=900)
        self.get_pipe_dist()
        self.air_balloon_update()
        self.obstacle_bird_update()

        pygame.display.set_caption('FlappyUAV')
        pygame.mixer.music.load(resource_path('resources/BGMUSIC.mp3'))
        pygame.mixer.music.play()

#%% 图像音频配置
    def loadImages(self):
        background = pygame.image.load(resource_path('resources/bg1.gif'))
        ground = pygame.image.load(resource_path('resources/ground.png'))
        ground = pygame.transform.scale(ground, (self.Win_width, ground.get_height()))
        return {'bg': background, 'ground': ground}

    def loadSounds(self):
        hit = pygame.mixer.Sound(resource_path('resources/hit.wav'))
        point = pygame.mixer.Sound(resource_path('resources/point.wav'))
        pickup = pygame.mixer.Sound(resource_path('resources/pickup.wav'))  # Add pickup sound
        deliver = pygame.mixer.Sound(resource_path('resources/deliver.wav'))  # Add delivery sound
        return {'hit': hit, 'point': point, 'pickup': pickup, 'deliver': deliver}

#%% 重置
    def reset(self,next_generation_jump):
        self.score = 0
        self.fitness = []
        self.weights_jump = [] # 当前代每只鸟的模型参数
        
        self.Bird_fitness_result = []
        self.Bird_state_result = []
        self.Bird_jump_result = []
        self.Bird_model_jump = []
        
        self.pipe_group.empty()
        self.hot_air_balloon_group.empty()
        self.obstacle_bird_group.empty()
        self.cargo_group.empty()
        self.bird_group.empty()
        
        self.new_pipes(time=0)
        self.get_pipe_dist()
        self.new_birds()
        
        for i, bird in enumerate(self.bird_group.sprites()):
            bird.model_jump.set_weight(next_generation_jump[i])
        pygame.mixer.music.play(-1)

#%% 更新鸟参数
    def new_birds(self):
        for i in range(self.generation_size//2):
            bird_height = random.randint(-200, 200)
            bird = Bird(100, int(self.Win_height / 2+bird_height),True) #初始位置
            self.bird_group.add(bird)
        for i in range(self.generation_size//2):
            bird_height = random.randint(-200, 200)
            bird = Bird(100, int(self.Win_height / 2+bird_height),False) #初始位置
            self.bird_group.add(bird)

    def birds_update(self):
        for i, bird in enumerate(self.bird_group.sprites()):
            if not bird.failed:
                self.score += bird.score # 累加总分
                state_jump = bird.get_state_jump(self.observed) # 获取鸟当前观察到的环境信息
                action = bird.get_action(state_jump) # AI 推理出动作（0 不动 or 1 扑腾）
                bird.update(action,self.cargo_group)  # 更新鸟的位置
                
                bird.get_fitness(self.observed,self.cargo_group) # 更新鸟的适应度
                # 通过事件适应度
                if bird.rect.left >= self.observed['pipe_dist_right']:
                    bird.score += 1 # 鸟飞过水管，加分

                    if bird.has_cargo == True and self.pipe_group.sprites()[0].state == 'dropoff':
                        if self.observed['obird_dist_right']-bird.rect.left>100:
                            bird.missed_boxes += 1
                            bird.fitness -= 50
                            bird.fitness_result[-1] -= 50
                        else:
                            bird.fitness += 2
                            bird.fitness_result[-1] += 2
                    elif bird.has_cargo == False and self.pipe_group.sprites()[0].state == 'pickup':
                        if self.observed['obird_dist_right']-bird.rect.left>100:
                            bird.missed_cargos += 1
                            bird.fitness -= 50
                            bird.fitness_result[-1] -= 50
                        else:
                            bird.fitness += 2
                            bird.fitness_result[-1] += 2
                    else:
                        bird.fitness += 10
                        bird.fitness_result[-1] += 10
                    
                    self.pipe_group.sprites()[0].passed = True # 标记水管为"已通过"
                    self.sounds['point'].play()
                    
                if bird.rect.left >= self.observed['balloon_dist_right'] and len(self.hot_air_balloon_group.sprites())>0:
                    bird.score += 1 # 鸟飞过水管，加分
                    bird.fitness += 10
                    bird.fitness_result[-1] += 10
                    self.hot_air_balloon_group.sprites()[0].passed = True # 标记水管为"已通过"
                    self.sounds['point'].play()

                if bird.rect.left >= self.observed['obird_dist_right'] and len(self.obstacle_bird_group.sprites())>0:
                    bird.score += 1 # 鸟飞过水管，加分
                    bird.fitness += 10
                    bird.fitness_result[-1] += 10
                    self.obstacle_bird_group.sprites()[0].passed = True # 标记水管为"已通过"
                    self.sounds['point'].play()
                    
                if bird.score > 1000:
                    bird.failed = True # 如果得分过高，终止当前鸟（通常用于避免跑太久）
                if (bird.missed_cargos>=10) or (bird.missed_boxes>=10):
                    bird.failed = True
            
            if bird.failed:
                self.weights_jump.append(bird.model_jump.get_weight()) # 保存失败鸟的神经网络参数
                self.fitness.append(bird.fitness) # 保存适应度（供遗传算法使用）
                
                self.Bird_fitness_result.append(bird.fitness_result)
                self.Bird_state_result.append(bird.state_result)
                self.Bird_jump_result.append(bird.jump_result)
                self.Bird_model_jump.append(bird.model_jump)
                
                bird.kill() # 删除鸟（从组中移除）

#%% 处理碰撞
    def handle_collision(self):
        # 气球碰撞检测
        for balloon in self.hot_air_balloon_group:
            for bird in self.bird_group:
                if bird.rect.colliderect(balloon.hitbox):
                    self.sounds['hit'].play()
                    bird.failed = True  # 标记鸟为失败状态
    
        # 飞鸟碰撞检测（使用hitbox）
        for obird in self.obstacle_bird_group:
            for bird in self.bird_group:
                if bird.rect.colliderect(obird.hitbox):
                    self.sounds['hit'].play()
                    bird.failed = True  # 标记鸟为失败状态

        # Special building collision logic
        for pipe in self.pipe_group:
            # Check if in delivery zone (safe)
            delivery_zone = pipe.get_delivery_zone()
            # 取送事件适应度
            for bird in self.bird_group:
            # UAV & building
                if delivery_zone.colliderect(bird.rect):
                    # Check pickup/delivery conditions
                    if not bird.has_cargo and pipe.state == 'pickup':
                        # Pick up cargo
                        # pipe.set_state('empty')
                        bird.has_cargo = True
                        bird.score += 1  # +1 for pickup
                        bird.fitness += 50
                        bird.fitness_result[-1] += 50
                        bird.missed_cargos = 0
                        self.sounds['point'].play()
                    elif bird.has_cargo and pipe.state == 'dropoff':
                        # Deliver cargo
                        bird.has_cargo = False
                        bird.score += 3  # +3 for delivery
                        bird.fitness += 50
                        bird.fitness_result[-1] += 50
                        bird.missed_boxes = 0
                        self.sounds['point'].play()
                    elif not bird.has_cargo and pipe.state == 'dropoff':
                        pass
                    elif bird.has_cargo and pipe.state == 'pickup':
                        pass
                    else:
                        bird.failed = True
                        self.sounds['hit'].play()
                    # Check collision with building body
                elif bird.rect.colliderect(pipe.rect):
                        bird.failed = True
                        self.sounds['hit'].play()
            
            # Cargo & building
            for cargo in self.cargo_group.sprites():
               if delivery_zone.colliderect(cargo.rect):
                   if pipe.state == 'dropoff':
                       cargo.bird.score += 3
                       cargo.bird.missed_cargos = 0
                       # cargo.bird.modify_fitness_from_cargo(5*5)
                       self.sounds['point'].play()
                       cargo.bird.cargo = None
                       cargo.kill()
                   else:
                       cargo.bird.score -= 5
                       # cargo.bird.modify_fitness_from_cargo(-5*5)
                       self.sounds['hit'].play()
                       cargo.bird.cargo = None
                       cargo.kill()
               elif pipe.rect.colliderect(cargo.rect):
                       cargo.bird.score -= 5
                       # cargo.bird.modify_fitness_from_cargo(-5*5)
                       self.sounds['hit'].play()
                       cargo.bird.cargo = None
                       cargo.kill()
                       
               elif cargo.rect.bottom >= Game.ground_y:
                       cargo.bird.score -= 5
                       # cargo.bird.modify_fitness_from_cargo(-5*5)
                       self.sounds['hit'].play()
                       cargo.bird.cargo = None
                       cargo.kill()  

#%% 更新建筑参数
    def pipe_update(self):
        self.new_pipes() # 是否需要添加新管道
        self.pipe_group.update()  # 更新管道位置（自动向左移动）
        if len(self.pipe_group)>0:
            self.get_pipe_dist() # 获取新的最近水管的信息

    def new_pipes(self, time=70, initial_offset=None):
        self.pipe_counter += 1
        if self.pipe_counter >= time:
            # 如果是第一个管道，让它离玩家更近（比如 400px 而不是屏幕最右侧）
            x = self.Win_width if initial_offset is None else initial_offset
            y = self.ground_y + 10
            state = random.choice(['empty', 'pickup', 'dropoff'])
            temp_pipe = Pipe(x, y, state=state)
            self.pipe_group.add(temp_pipe)
            self.pipe_counter = 0

    def get_pipe_dist(self):
        pipe_closest = [pipe for pipe in self.pipe_group.sprites() if pipe.passed==False][:1]
        for pipe in pipe_closest:
            self.observed['pipe_dist_right'] = pipe.rect.right # 水管右边缘 x 坐标
            self.observed['pipe_dist_top'] = pipe.rect.top # 水管底部的 y 坐标
            self.observed['pipe_dist_centerx'] = pipe.rect.centerx
            if pipe.state == 'empty':
                self.observed['pipe_dropoff'] = 0
                self.observed['pipe_pickup'] = 0
            elif pipe.state ==  'pickup':
                self.observed['pipe_dropoff'] = 0
                self.observed['pipe_pickup'] = 1
            elif pipe.state == 'dropoff':
                self.observed['pipe_dropoff'] = 1
                self.observed['pipe_pickup'] = 0

        # pipe_closest_pickup = [pipe for pipe in self.pipe_group.sprites() if pipe.passed==False and pipe.state=='pickup']
        # for pipe in pipe_closest_pickup[:1]:
        #     self.observed['pipe_pickup_dist_right'] = pipe.rect.right # 水管右边缘 x 坐标
        #     self.observed['pipe_pickup_dist_top'] = pipe.rect.top # 水管底部的 y 坐标       
        # pipe_closest_dropoff = [pipe for pipe in self.pipe_group.sprites() if pipe.passed==False and pipe.state=='dropoff']
        # if len(pipe_closest_dropoff)>0:
        #     for pipe in pipe_closest_dropoff[:1]:
        #         self.observed['pipe_dropoff_dist_right'] = pipe.rect.right # 水管右边缘 x 坐标
        #         self.observed['pipe_dropoff_dist_top'] = pipe.rect.top # 水管底部的 y 坐标   
        # else:
        #     self.observed['pipe_dropoff_dist_right'] = pipe.rect.right # 水管右边缘 x 坐标
        #     self.observed['pipe_dropoff_dist_top'] = pipe.rect.top # 水管底部的 y 坐标   
        

#%% 更新地面参数
    def ground_update(self):
        self.ground_x -= Pipe.scroll_speed
        if self.ground_x <= -self.Win_width:
            self.ground_x = 0

#%% 更新气球参数
    def air_balloon_update(self):
        self.new_hot_air_balloon()
        self.hot_air_balloon_group.update()
        balloon_closest = [balloon for balloon in self.hot_air_balloon_group.sprites() if balloon.passed==False]
        if len(balloon_closest)>0:
            self.get_balloon_dist(balloon_closest)
        else:
            self.observed['balloon_dist_right'] = 600
            self.observed['balloon_dist_top'] = 0 # 气球右边缘 x 坐标
            self.observed['balloon_dist_bottom'] = 0 # 气球底部的 y 坐标
            self.observed['balloon_dist_centerx'] = 600 # 气球底部的 y 坐标

    def new_hot_air_balloon(self):
        self.hot_air_balloon_timer += 1
        if self.hot_air_balloon_timer > 400:
            for _ in range(5):
                x = self.Win_width
                y = random.randint(25, 50)
                hot_air_balloon = HotAirBalloon(x, y)
    
                if not pygame.sprite.spritecollide(hot_air_balloon, self.pipe_group, False) and \
                   not pygame.sprite.spritecollide(hot_air_balloon, self.hot_air_balloon_group, False) and \
                   not pygame.sprite.spritecollide(hot_air_balloon, self.obstacle_bird_group, False) and \
                   not pygame.sprite.spritecollide(hot_air_balloon, self.bird_group, False):
                    self.hot_air_balloon_group.add(hot_air_balloon)
                    self.hot_air_balloon_timer = 0
                    break

    def get_balloon_dist(self,balloon_closest):
        #balloon_closest = [balloon for balloon in self.hot_air_balloon_group.sprites() if balloon.passed==False]
        self.observed['balloon_dist_right'] = balloon_closest[0].rect.right 
        self.observed['balloon_dist_top'] = balloon_closest[0].rect.top # 气球右边缘 x 坐标
        self.observed['balloon_dist_bottom'] = balloon_closest[0].rect.bottom # 气球底部的 y 坐标
        self.observed['balloon_dist_centerx'] = balloon_closest[0].rect.centerx # 气球底部的 y 坐标


#%% 更新障碍鸟参数
    def obstacle_bird_update(self):
        self.new_obstacle_bird()
        self.obstacle_bird_group.update()
        obird_closest = [obird for obird in self.obstacle_bird_group.sprites() if obird.passed==False]
        if len(obird_closest)>0:
            self.get_obstacle_bird_dist(obird_closest)
        else:
            self.observed['obird_dist_right'] = 600
            self.observed['obird_dist_top'] = 650 # 气球右边缘 x 坐标
            self.observed['obird_dist_bottom'] = 650 # 气球底部的 y 坐标
            self.observed['obird_dist_centerx'] = 600 # 气球底部的 y 坐标

    def new_obstacle_bird(self):
        self.obstacle_bird_timer += 1
        if self.obstacle_bird_timer > 300:
            for _ in range(5):
                x = self.Win_width
                y = random.randint(300, 400)
                obstacle_bird = ObstacleBird(x, y)
    
                if not pygame.sprite.spritecollide(obstacle_bird, self.pipe_group, False) and \
                   not pygame.sprite.spritecollide(obstacle_bird, self.hot_air_balloon_group, False) and \
                   not pygame.sprite.spritecollide(obstacle_bird, self.obstacle_bird_group, False) and \
                   not pygame.sprite.spritecollide(obstacle_bird, self.bird_group, False):
                    self.obstacle_bird_group.add(obstacle_bird)
                    self.obstacle_bird_timer = 0
                    break

    def get_obstacle_bird_dist(self,obird_closest):
        self.observed['obird_dist_right'] = obird_closest[0].rect.right 
        self.observed['obird_dist_top'] = obird_closest[0].rect.top # 气球右边缘 x 坐标
        self.observed['obird_dist_bottom'] = obird_closest[0].rect.bottom # 气球底部的 y 坐标
        self.observed['obird_dist_centerx'] = obird_closest[0].rect.centerx  # 气球底部的 y 坐标

#%% 画
    def draw(self):
        self.surface.blit(self.images['bg'], (0, 0))
        self.pipe_group.draw(self.surface)
        self.hot_air_balloon_group.draw(self.surface)
        self.obstacle_bird_group.draw(self.surface)
        self.cargo_group.draw(self.surface)
        self.bird_group.draw(self.surface)

        self.surface.blit(self.images['ground'], (self.ground_x, self.ground_y))
        self.surface.blit(self.images['ground'], (self.ground_x + self.Win_width, self.ground_y))
        
        # 显示分数（右上角）
        self.draw_text(f'Score: {self.score}', (255, 255, 255), 20, 20)

        for balloon in self.hot_air_balloon_group:
            pygame.draw.rect(self.surface, (255,0,0), balloon.hitbox, 1)
        for bird in self.obstacle_bird_group:
            pygame.draw.rect(self.surface, (255,0,0), bird.hitbox, 1)

    def draw_text(self, text, color, x, y):
        img = self.font.render(text, True, color)
        self.surface.blit(img, (x, y))

#%% 每一帧的动作
    def play_step(self):
        game_over = False
        self.score = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                #game_over = True
                pygame.quit()
                sys.exit()

        self.pipe_update() # 更新管道
        self.ground_update() # 更新地面
        self.air_balloon_update()
        self.obstacle_bird_update()
        self.cargo_group.update()
    
        self.birds_update()
        self.handle_collision() #处理碰撞
        
        if len(self.bird_group) == 0 or self.score>10000:
            game_over = True
            return game_over, self.score
    
        self.draw()
        pygame.display.update()
        self.Clock.tick(self.fps)
        return game_over, self.score



# torch.serialization.add_safe_globals([Linear_Net])
#%% GA算法
class GATrainer:
    def __init__(self,begin_with_model=True):
        self.game = Game()
        self.generate_num = 0 # 当前代数编号
        self.mutate_pop_rate = 0.2 # 每一代中会变异的个体比例（20%）
        self.mutate_net_rate = 0.1 # 每个网络中会变异的权重比例（10%）
        self.best_fitness = 0
        self.current_best_fitness = 0
        if begin_with_model:
            for bird in self.game.bird_group:
                bird.model_jump = torch.load(resource_path('.\model\model1.pth'),weights_only=False)

    @staticmethod
    def list2tensor(weights):
        return torch.concat([weights[0].flatten(),weights[1],weights[2].flatten(),weights[3]])

    @staticmethod
    def tensor2list(weights,input_size):
        output_weights = []
        index  = [
            input_size*Bird.hidden_size, 
				input_size*Bird.hidden_size+Bird.hidden_size,
				input_size*Bird.hidden_size+Bird.hidden_size+Bird.hidden_size*Bird.output_size]
        output_weights.append(weights[:index[0]].reshape(Bird.hidden_size,input_size)) # W1
        output_weights.append(weights[index[0]:index[1]]) # b1
        output_weights.append(weights[index[1]:index[2]].reshape(Bird.output_size,Bird.hidden_size)) # W2
        output_weights.append(weights[index[2]:]) # b2
        
        return output_weights

    def cross_mutate(self,weights_1, weights_2,parameter_len,input_size):
        # 转换为1D向量
        weights_1 = GATrainer.list2tensor(weights_1)
        weights_2 = GATrainer.list2tensor(weights_2)
        # 交叉：从随机位置拼接两个个体
        #crossover_idx = random.randint(0, Game.parameter_len-1)
        #new_weights = torch.concat([weights_1[:crossover_idx] , weights_2[crossover_idx:]])

        mask = torch.randint(0, 2, (parameter_len,))
        new_weights = mask * weights_1 + (1 - mask) * weights_2
      
        # 变异机制
        if (random.randint(0,self.game.generation_size-1) <= 
      		self.game.generation_size*self.mutate_pop_rate):
            mutate_num = int(self.mutate_net_rate*parameter_len) #变异的参数个数
            for _ in range(mutate_num):
                i = random.randint(0,parameter_len-1)
                new_weights[i] += torch.randn(1).numpy()/5  # 随机加减一个参数
        output_weights = GATrainer.tensor2list(new_weights,input_size)
        return 	output_weights
	
    @staticmethod
    def fitness_prob(fitness):
        fitness = np.array(fitness) # 提取适应度
        min_fitness = min(fitness)
        if min_fitness<0:
            fitness = fitness-min_fitness
        return fitness/np.sum(fitness) # 标准化适应度
    
    def compare_model_weights(self,model1_params, model2_params):
        for p1, p2 in zip(model1_params, model2_params):
            if not torch.equal(p1, p2):
                return False
        return True

    def lamarckian_fine_tune(self):
        
        horizon = 40
        window = 20
        epoch = 0
        while epoch<=5:
            index = 0
            for state_batch,fitness_batch,jump_batch,model_jump in zip(self.game.Bird_state_result,self.game.Bird_fitness_result,self.game.Bird_jump_result,self.game.Bird_model_jump):
                optimizer = torch.optim.Adam(model_jump.parameters(), lr=0.01)
                if len(state_batch)>horizon:
                    fitness_batch = fitness_batch+[-100]*window # 变成
                    fitness_batch = [sum(fitness_batch[i:i+window]) for i in range(len(fitness_batch)-window)]
    
                    state_batch = torch.tensor(state_batch, dtype=torch.float32)
                    fitness_batch = torch.tensor(fitness_batch, dtype=torch.float32)
                    jump_batch = torch.tensor(jump_batch, dtype=torch.int32)
                    
                    outputs = model_jump(state_batch)
                    jump_probs = torch.softmax(outputs, dim=-1)
                    selected_probs = jump_probs[torch.arange(len(jump_batch)), jump_batch]
                    losses = -torch.log(selected_probs + 1e-6) * fitness_batch
                    loss = losses.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print('index',index)
                    # are_weights_equal = self.compare_model_weights(self.game.weights_jump[index], model_jump.get_weight())
                    # print("权重是否完全相同:", are_weights_equal)
                    self.game.weights_jump[index] = model_jump.get_weight()
                index += 1
            epoch += 1

    def reproduce(self):
        next_generation_jump = [] 
        prob = GATrainer.fitness_prob(self.game.fitness)
        sorted_indices = list(np.argsort(prob))
        second_index, first_index= sorted_indices[-2:] # 提取最优的两个适应度
        # 保存最优模型的权重文件
        self.current_best_fitness = self.game.fitness[first_index]
        if self.current_best_fitness>self.best_fitness:
            torch.save(self.game.Bird_model_jump[first_index], resource_path('.\model\model.pth'))
            self.best_fitness = self.current_best_fitness
        
        next_generation_jump.append(self.game.weights_jump[first_index]) # 新一代加入这两个原生个体
        next_generation_jump.append(self.game.weights_jump[second_index]) # 新一代加入这两个原生个体
        # 设置精英交叉数量
        elite_crossover_num = (self.game.generation_size - 2) // 2  # 除去2个精英鸟后的一半
        random_crossover_num = (self.game.generation_size - 2) - elite_crossover_num
        elite_pool = sorted_indices[-4:]  # 取适应度前5的鸟作为精英配种池 
        #精英交叉
        for _ in range(elite_crossover_num): # 其余8个重构
            p1, p2 = np.random.choice(elite_pool,size=2, replace=False) # 按概率选取两个杂交
            next_generation_jump.append(self.cross_mutate(self.game.weights_jump[p1],self.game.weights_jump[p2],Game.parameter_len_jump,Bird.jump_size))
       #随机交叉
        for _ in range(random_crossover_num): # 其余8个重构
            p1, p2 = np.random.choice(len(prob), size=2, replace=False, p=prob) # 按概率选取两个杂交
            next_generation_jump.append(self.cross_mutate(self.game.weights_jump[p1],self.game.weights_jump[p2],Game.parameter_len_jump,Bird.jump_size))
        return next_generation_jump

    def run_GA(self):
        while True:
            game_over, score  = self.game.play_step()
            if game_over :
                self.lamarckian_fine_tune() #在每一代结束后，先做Lamarckian微调
                # 保存模型文件
                print(f"generate {self.generate_num} average fitness: {sum(self.game.fitness)/10} overall_best_fitness:{self.best_fitness} local_best_fitness:{self.current_best_fitness}")
                
                next_generation_jump = self.reproduce() # 繁衍
                self.game.reset(next_generation_jump) # 重新自然选择
                self.generate_num += 1

def main():
    trainer = GATrainer(False)
    trainer.run_GA()



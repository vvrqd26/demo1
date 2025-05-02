import pygame
from pygame.locals import *
import random
import os

import os
import sys

# 获取资源文件的绝对路径
def resource_path(relative_path):
    """用于获取资源文件的路径，适配 PyInstaller 打包后的路径"""
    if hasattr(sys, '_MEIPASS'):  # PyInstaller 打包后的临时目录
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.images_with_cargo = []  # UAV 1-3 (with cargo)
        self.images_without_cargo = []  # UAV 4-6 (without cargo)
        
        # Load images for both states
        for num in range(1, 4):
            img = pygame.image.load(resource_path(f"resources/UAV_{num}.png"))
            self.images_with_cargo.append(img)
        for num in range(4, 7):
            img = pygame.image.load(resource_path(f"resources/UAV_{num}.png"))
            self.images_without_cargo.append(img)
            
        self.current_images = self.images_without_cargo  # Start without cargo
        self.index = 0
        self.counter = 0
        self.vel = 0
        self.cap = 9
        self.flying = False
        self.failed = False
        self.clicked = False
        self.has_cargo = False  # Start without cargo

        self.image = self.current_images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.wing = pygame.mixer.Sound(resource_path('resources/wing.wav'))

    def handle_input(self):
        if pygame.mouse.get_pressed()[0] == 1 and not self.clicked:
            self.clicked = True
            self.vel = -1 * self.cap
            self.wing.play()
        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

    def animation(self):
        flap_cooldown = 5
        self.counter += 1
        if self.counter > flap_cooldown:
            self.counter = 0
            self.index = (self.index + 1) % 3
            self.image = self.current_images[self.index]
        self.image = pygame.transform.rotate(self.current_images[self.index], self.vel * 0)

    def touch_ground(self):
        return self.rect.bottom >= Game.ground_y

    def update(self):
        if self.flying:
            self.handle_input()
            self.vel += 0.4
            if self.vel > 8:
                self.vel = 8
            if not self.touch_ground():
                self.rect.y += int(self.vel)
        
        # Update current images based on cargo state
        if self.has_cargo:
            self.current_images = self.images_with_cargo
        else:
            self.current_images = self.images_without_cargo
    
        if not self.failed:
            self.animation()
        else:
            self.image = pygame.transform.rotate(self.current_images[self.index], -90)

class Pipe(pygame.sprite.Sprite):
    scroll_speed = 4

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
        self.delivery_zone_height = 70
        self.delivery_zone_width = 40
        
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
            self.rect.centerx - self.delivery_zone_width // 2,  # 居中X位置
            self.rect.y - self.delivery_zone_height,            # 顶部上方50px
            self.delivery_zone_width,                           # 宽度30px
            self.delivery_zone_height                           # 高度50px
        )
        
    def update(self):
        self.rect.x -= Pipe.scroll_speed
        if self.rect.right < 0:
            self.kill()

class Cargo(pygame.sprite.Sprite):
    def __init__(self, x, y, speed):
        super().__init__()
        self.image = pygame.image.load(resource_path("resources/cargo.png"))
        self.rect = self.image.get_rect(center=(x, y))
        self.vel = speed

    def touch_ground(self):
        return self.rect.bottom > Game.ground_y

    def update(self):
        self.vel += 0.4
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

class Button:
    def __init__(self, x, y):
        self.image = pygame.image.load(resource_path('resources/restart.png'))
        self.rect = self.image.get_rect(centerx=x, centery=y)

    def pressed(self, event):
        pressed = False
        if event.type == MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(pos):
                pressed = True
        return pressed

    def draw(self, surface):
        surface.blit(self.image, self.rect)

class Game():
    ground_y = 650

    def __init__(self, Width=1200, Height=800):
        pygame.init()
        self.Win_width, self.Win_height = (Width, Height)
        self.surface = pygame.display.set_mode((self.Win_width, self.Win_height))
        # 设置窗口位置为屏幕中央
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        self.ground_x = 0
        self.score = 0
        self.pipe_counter = 0
        
        self.observed = dict()
        self.Clock = pygame.time.Clock()
        self.fps = 60

        self.hot_air_balloon_timer = 0
        self.obstacle_bird_timer = 0
    
        self.font = pygame.font.SysFont('Futura', 50)
        self.images = self.loadImages()
        self.sounds = self.loadSounds()
        self.pipe_group = pygame.sprite.Group()
        self.bird_group = pygame.sprite.Group()
        self.cargo_group = pygame.sprite.Group()
        self.hot_air_balloon_group = pygame.sprite.Group()
        self.obstacle_bird_group = pygame.sprite.Group()
        self.flappy = Bird(100, self.ground_y // 2)
        self.bird_group.add(self.flappy)
        self.new_pipes(time=0)
        self.button = Button(self.Win_width // 2, self.Win_height // 2)

        self.new_pipes(time=0, initial_offset=900)

        pygame.display.set_caption('Flappy UAV')
        pygame.mixer.music.load(resource_path('resources/BGMUSIC.mp3'))
        pygame.mixer.music.play(-1)

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

    def start_flying(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and not self.flappy.flying and not self.flappy.failed:
            self.flappy.flying = True
            
    def handle_collision(self):
        # 气球碰撞检测
        for balloon in self.hot_air_balloon_group:
            if self.flappy.rect.colliderect(balloon.hitbox):
                self.flappy.failed = True
                self.sounds['hit'].play()
                pygame.mixer.music.stop()
                return
    
        # 飞鸟碰撞检测（使用hitbox）
        for bird in self.obstacle_bird_group:
            if self.flappy.rect.colliderect(bird.hitbox):
                self.flappy.failed = True
                self.sounds['hit'].play()
                pygame.mixer.music.stop()
                return

        # Check collision with other obstacles
        if (
                self.flappy.rect.top < 0 or
                self.flappy.rect.bottom >= Game.ground_y):
            self.flappy.failed = True
            self.sounds['hit'].play()
            pygame.mixer.music.stop()

        # Special building collision logic
        for pipe in self.pipe_group:
            # Check if in delivery zone (safe)
            delivery_zone = pipe.get_delivery_zone()
            
            # UAV & building
            if delivery_zone.colliderect(self.flappy.rect):
                # Check pickup/delivery conditions
                if not self.flappy.has_cargo and pipe.state == 'pickup':
                    # Pick up cargo
                    pipe.set_state('empty')
                    self.flappy.has_cargo = True
                    self.score += 1  # +1 for pickup
                    self.sounds['point'].play()
                elif self.flappy.has_cargo and pipe.state == 'dropoff':
                    # Deliver cargo
                    self.flappy.has_cargo = False
                    self.score += 3  # +3 for delivery
                    self.sounds['point'].play()
            else:
                # Check collision with building body
                if self.flappy.rect.colliderect(pipe.rect):
                    self.flappy.failed = True
                    self.sounds['hit'].play()
                    pygame.mixer.music.stop()
            
            # Cargo & building
            for cargo in self.cargo_group.sprites():
               if delivery_zone.colliderect(cargo.rect):
                   if pipe.state == 'dropoff':
                       self.score += 3
                       self.sounds['point'].play()
                       cargo.kill()
                   else:
                       self.score -= 5
                       self.sounds['hit'].play()
                       cargo.kill()
               elif pipe.rect.colliderect(cargo.rect):
                       self.score -= 5
                       self.sounds['hit'].play()
                       cargo.kill()
                       
               elif cargo.rect.bottom >= Game.ground_y:
                       self.score -= 5
                       self.sounds['hit'].play()
                       cargo.kill()  

    def ground_update(self):
        self.ground_x -= Pipe.scroll_speed
        if self.ground_x <= -self.Win_width:
            self.ground_x = 0

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
            
    def new_hot_air_balloon(self):
        self.hot_air_balloon_timer += 1
        if self.hot_air_balloon_timer > random.randint(500, 800):
            for _ in range(5):
                x = self.Win_width
                y = random.randint(50, 100)
                hot_air_balloon = HotAirBalloon(x, y)
    
                if not pygame.sprite.spritecollide(hot_air_balloon, self.pipe_group, False) and \
                   not pygame.sprite.spritecollide(hot_air_balloon, self.hot_air_balloon_group, False) and \
                   not pygame.sprite.spritecollide(hot_air_balloon, self.obstacle_bird_group, False) and \
                   not pygame.sprite.spritecollide(hot_air_balloon, self.bird_group, False):
                    self.hot_air_balloon_group.add(hot_air_balloon)
                    self.hot_air_balloon_timer = 0
                    break

    def new_obstacle_bird(self):
        self.obstacle_bird_timer += 1
        if self.obstacle_bird_timer > random.randint(400, 600):
            for _ in range(5):
                x = self.Win_width
                y = random.randint(150, 250)
                obstacle_bird = ObstacleBird(x, y)
    
                if not pygame.sprite.spritecollide(obstacle_bird, self.pipe_group, False) and \
                   not pygame.sprite.spritecollide(obstacle_bird, self.hot_air_balloon_group, False) and \
                   not pygame.sprite.spritecollide(obstacle_bird, self.obstacle_bird_group, False) and \
                   not pygame.sprite.spritecollide(obstacle_bird, self.bird_group, False):
                    self.obstacle_bird_group.add(obstacle_bird)
                    self.obstacle_bird_timer = 0
                    break

    def reset_game(self):
        self.pipe_group.empty()
        self.hot_air_balloon_group.empty()
        self.obstacle_bird_group.empty()
        self.new_pipes(time=0)
        self.bird_group.empty()
        self.flappy = Bird(100, self.ground_y // 2)
        self.bird_group.add(self.flappy)
        self.score = 0
        self.observed = dict()
        self.flappy.has_cargo = False  # Start without cargo
        self.cargo_group.empty()
        pygame.mixer.music.play(-1)

    def game_restart(self, event):
        if self.flappy.failed and self.button.pressed(event):
            self.flappy.failed = False
            self.reset_game()

    def check_failed(self):
        if self.flappy.failed:
            pygame.mixer.music.stop()
            if self.flappy.touch_ground():
                self.button.draw(self.surface)
                self.flappy.flying = False

    def draw_text(self, text, color, x, y):
        img = self.font.render(text, True, color)
        self.surface.blit(img, (x, y))


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
        
        # 新增：显示无人机载货状态（左上角）
        cargo_status = "Yes" if self.flappy.has_cargo else "No"
        status_color = (0, 255, 0) if self.flappy.has_cargo else (255, 255, 255)
        self.draw_text(f'Cargo: {cargo_status}', status_color, 20, 60)

        for balloon in self.hot_air_balloon_group:
            pygame.draw.rect(self.surface, (255,0,0), balloon.hitbox, 1)
        for bird in self.obstacle_bird_group:
            pygame.draw.rect(self.surface, (255,0,0), bird.hitbox, 1)

    def play_step(self):
        game_over = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            self.start_flying(event)
            self.game_restart(event)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if self.flappy.has_cargo and not self.flappy.failed:
                    cargo = Cargo(self.flappy.rect.centerx, self.flappy.rect.bottom, self.flappy.vel)
                    self.cargo_group.add(cargo)
                    self.flappy.has_cargo = False
    
        self.bird_group.update()
        if not self.flappy.failed and self.flappy.flying:
            self.handle_collision()
            self.new_pipes()
            self.new_hot_air_balloon()
            self.new_obstacle_bird()
            self.pipe_group.update()
            self.hot_air_balloon_group.update()
            self.obstacle_bird_group.update()
            self.cargo_group.update()
            self.ground_update() 
    
            # Check passing obstacles for score
            for obstacle in (self.pipe_group, self.hot_air_balloon_group, self.obstacle_bird_group):
                for obj in obstacle:
                    if not obj.passed and self.flappy.rect.centerx > obj.rect.right:
                        obj.passed = True
                        self.score += 1
                        self.sounds['point'].play()
    
        self.draw()
        self.check_failed()
        pygame.display.update()
        self.Clock.tick(self.fps)
        return game_over, self.score

def main():
    game = Game()
    while True:
        game_over, score = game.play_step()
        if game_over:
            break

    print('Final Score:', score)
    pygame.quit()
    sys.exit()
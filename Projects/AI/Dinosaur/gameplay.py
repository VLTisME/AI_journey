# This is the Dinosaur offline game of Google with horizontal movement  

import pygame  # type: ignore  
import os  
import sys
import random  

pygame.init()  

# GLOBAL VARIABLES  

# GLOBAL VARIABLES  
def resource_path(relative_path):  
    """ Get absolute path to resource, works for dev and when packaged with PyInstaller """  
    try:  
        # PyInstaller creates a temp folder and stores path in _MEIPASS  
        base_path = sys._MEIPASS  
    except AttributeError:  
        base_path = os.path.abspath(".")  

    return os.path.join(base_path, relative_path)

SCREEN_HEIGHT = 600  
SCREEN_WIDTH = 1100  
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))  

FPS = 120  # Set the frame rate to 120 FPS  
FPS_BASELINE = 30  # Original FPS to maintain game speed  

RUNNING = [  
    pygame.image.load(resource_path("assets/Dino/DinoRun1.png")),  
    pygame.image.load(resource_path("assets/Dino/DinoRun2.png")),  
]  
JUMPING = pygame.image.load(resource_path("assets/Dino/DinoJump.png"))  
DUCKING = [  
    pygame.image.load(resource_path("assets/Dino/DinoDuck1.png")),  
    pygame.image.load(resource_path("assets/Dino/DinoDuck2.png")),  
]  

BIRD = [  
    pygame.image.load(resource_path("assets/Bird/Bird1.png")),  
    pygame.image.load(resource_path("assets/Bird/Bird2.png")),  
]  
CLOUD = pygame.image.load(resource_path("assets/Other/Cloud.png"))  
BG = pygame.image.load(resource_path("assets/Other/Track.png"))  
SMALL_CACTUS = [  
    pygame.image.load(resource_path("assets/Cactus/SmallCactus1.png")),  
    pygame.image.load(resource_path("assets/Cactus/SmallCactus2.png")),  
    pygame.image.load(resource_path("assets/Cactus/SmallCactus3.png")),  
]  
LARGE_CACTUS = [  
    pygame.image.load(resource_path("assets/Cactus/LargeCactus1.png")),  
    pygame.image.load(resource_path("assets/Cactus/LargeCactus2.png")),  
    pygame.image.load(resource_path("assets/Cactus/LargeCactus3.png")),  
]


class Dinosaur:  
    X_POS_START = 80  # Starting X position  
    Y_POS = 310  
    Y_POS_DUCK = 340  
    JUMP_VEL = 8.5  
    MOVE_SPEED = 10  # Horizontal movement speed  

    def __init__(self):  
        self.duck_img = DUCKING  
        self.run_img = RUNNING  
        self.jump_img = JUMPING  

        self.dino_duck = False  
        self.dino_jump = False  
        self.dino_run = True  
        self.dino_force_down = False  

        self.jump_vel = self.JUMP_VEL  
        self.step_index = 0  
        self.image = self.run_img[0]  
        self.dino_rect = self.image.get_rect()  
        self.dino_rect.x = self.X_POS_START  
        self.dino_rect.y = self.Y_POS  

        self.move_speed = self.MOVE_SPEED  # Horizontal movement speed  
        self.max_x = SCREEN_WIDTH - self.image.get_width()  # Maximum X position  
        self.min_x = 0  # Minimum X position  

        # Create the initial mask  
        self.mask = pygame.mask.from_surface(self.image)  

    def update(self, userInput, delta_multiplier):  
        if self.dino_run:  
            self.run(delta_multiplier)  
        if self.dino_jump:  
            self.jump(delta_multiplier)  
        if self.dino_duck:  
            self.duck(delta_multiplier)  

        if self.step_index >= 10:  
            self.step_index = 0  

        # Horizontal movement  
        if userInput[pygame.K_RIGHT]:  
            self.dino_rect.x += self.move_speed * delta_multiplier  
        if userInput[pygame.K_LEFT]:  
            self.dino_rect.x -= self.move_speed * delta_multiplier  

        # Keep the dinosaur within screen bounds  
        self.dino_rect.x = max(self.min_x, min(self.dino_rect.x, self.max_x))  

        # State transitions  
        if userInput[pygame.K_UP] and not self.dino_jump:  
            self.dino_run = False  
            self.dino_jump = True  
            self.dino_duck = False  
        elif userInput[pygame.K_DOWN] and not self.dino_jump:  
            self.dino_duck = True  
            self.dino_run = False  
            self.dino_jump = False  
        elif userInput[pygame.K_DOWN] and self.dino_jump:  
            self.dino_duck = False  
            self.dino_run = False  
            self.dino_jump = True  
            self.dino_force_down = True  
        elif not (self.dino_jump or userInput[pygame.K_DOWN]):  
            self.dino_run = True  
            self.dino_jump = False  
            self.dino_duck = False  

    def run(self, delta_multiplier):  
        self.image = self.run_img[int(self.step_index) // 5]  
        self.dino_rect.y = self.Y_POS  
        self.step_index += 1 * delta_multiplier  
        self.mask = pygame.mask.from_surface(self.image)  # Update mask  --> Every time the image changes, you have to update the mask (because it nearly gets the perfect shape of the image, unlike the Rect)

    def jump(self, delta_multiplier):  
        self.image = self.jump_img  
        if self.dino_jump:  
            if self.dino_force_down:  
                self.dino_rect.y += 50 * delta_multiplier  
                self.jump_vel -= 0.8 * delta_multiplier  
            else:  
                self.dino_rect.y -= self.jump_vel * 4 * delta_multiplier  
                self.jump_vel -= 0.8 * delta_multiplier  
        if self.jump_vel < -self.JUMP_VEL or self.dino_rect.y >= self.Y_POS:  
            self.dino_rect.y = self.Y_POS  
            self.dino_jump = False  
            self.dino_force_down = False  
            self.jump_vel = self.JUMP_VEL  

        self.step_index += 1 * delta_multiplier  
        self.mask = pygame.mask.from_surface(self.image)  # Update mask  

    def duck(self, delta_multiplier):  
        self.image = self.duck_img[int(self.step_index) // 5]  
        self.dino_rect.y = self.Y_POS_DUCK  
        self.step_index += 1 * delta_multiplier  
        self.mask = pygame.mask.from_surface(self.image)  # Update mask  

    def draw(self, SCREEN):  
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))  


class Cloud:  
    def __init__(self):  
        self.image = CLOUD  
        self.x = SCREEN_WIDTH + random.randint(50, 100)  
        self.y = random.randint(50, 100)  
        self.width = self.image.get_width()  

    def update(self, delta_multiplier):  
        self.x -= game_speed * delta_multiplier  
        if self.x < -self.width:  
            self.x = SCREEN_WIDTH + random.randint(50, 100)  
            self.y = random.randint(50, 100)  

    def draw(self, SCREEN):  
        SCREEN.blit(self.image, (self.x, self.y))  


class Obstacles:  
    def __init__(self, image, type):  
        self.image = image  
        self.type = type  
        self.rect = self.image[self.type].get_rect()  
        self.rect.x = SCREEN_WIDTH  
        self.mask = pygame.mask.from_surface(self.image[self.type])  # Create mask  

    def update(self, delta_multiplier):  
        self.rect.x -= game_speed * delta_multiplier  
        if self.rect.x < -self.rect.width:  
            obstacles.pop()  

    def draw(self, SCREEN):  
        SCREEN.blit(self.image[self.type], self.rect)  
        # No need to update mask unless the image changes  


class SmallCactus(Obstacles):  
    def __init__(self, image, type):  # u dont want to inherit __init__ from the parent
        super().__init__(image, type) # uhh u changed ur mind and call all the __init__ codes in the parent
        self.rect.y = 325 # seems like parent doesnt have this.... lets add it here
        # self.mask = pygame.mask.from_surface(self.image[self.type])  # Update mask  - but somehow you want to update the mask again, so you do it here again
        # but it is no need to reinitialize the mask, because it is already initialized in the parent class and the image doesnt change

class LargeCactus(Obstacles):  
    def __init__(self, image, type):  
        super().__init__(image, type)  
        self.rect.y = 300  
        # self.mask = pygame.mask.from_surface(self.image[self.type])  # Update mask  


class Bird(Obstacles):  
    def __init__(self, image):  
        self.type = 0  
        super().__init__(image, self.type)  
        self.rect.y = random.choice([200, 250, 300])  
        self.index = 0  
        # Initialize mask  
        self.image_type = self.image[self.type]  
        self.mask = pygame.mask.from_surface(self.image_type)  

    def draw(self, SCREEN):  
        if self.index >= 9:  
            self.index = 0  
        current_image = self.image[int(self.index) // 5]  
        SCREEN.blit(current_image, self.rect)  
        self.index += 1 * FPS_BASELINE / FPS  
        # Update mask with current image  
        self.mask = pygame.mask.from_surface(current_image)  


def background(delta_multiplier):  
    global x_pos_bg, y_pos_bg  
    background_width = BG.get_width()  
    SCREEN.blit(BG, (x_pos_bg, y_pos_bg))  
    SCREEN.blit(BG, (background_width + x_pos_bg, y_pos_bg))  
    x_pos_bg -= game_speed * delta_multiplier  
    if x_pos_bg - game_speed * delta_multiplier <= -background_width:  
        x_pos_bg = 0  


def score():  
    global points, game_speed, font, cnt  
    if cnt % 5 == 0:  
        points += 1  
    if points % 100 == 0:  
        game_speed += 1  
    game_speed = min(game_speed, 70)  
    text = font.render(f"Points: {points}", True, (0, 0, 0))  
    SCREEN.blit(text, (900, 50))  


def main():  
    global game_speed, x_pos_bg, y_pos_bg, points, font, cnt, obstacles  
    x_pos_bg = 0  
    y_pos_bg = 380  
    game_speed = 14  
    run = True  
    clock = pygame.time.Clock()  
    player = Dinosaur()  
    cloud = Cloud()  
    points = 0  
    cnt = 0  
    death_count = 0
    font = pygame.font.Font('freesansbold.ttf', 20)  

    obstacles = []  

    while run:  
        clock.tick(FPS)  # Control the frame rate  
        delta_time = clock.get_time() / 1000  
        delta_multiplier = delta_time * FPS_BASELINE  

        for event in pygame.event.get():  
            if event.type == pygame.QUIT:  
                run = False  

        userInput = pygame.key.get_pressed()  

        # Update background, score, and cloud  
        SCREEN.fill((225, 225, 225))  
        background(delta_multiplier)  
        cnt += 1  
        score()  
        cloud.update(delta_multiplier)  
        cloud.draw(SCREEN)  

        # Add new obstacles if needed  
        if len(obstacles) == 0:  
            if random.randint(0, 2) == 0:  
                obstacles.append(SmallCactus(SMALL_CACTUS, random.randint(0, 2)))  
            elif random.randint(0, 2) == 1:  
                obstacles.append(LargeCactus(LARGE_CACTUS, random.randint(0, 2)))  
            elif random.randint(0, 2) == 2:  
                obstacles.append(Bird(BIRD))  

        # Update player and obstacles positions  
        player.update(userInput, delta_multiplier)  
        for obstacle in obstacles:  
            obstacle.update(delta_multiplier)  

        # Collision detection using masks  
        for obstacle in obstacles:  
            # First check for rectangle collision to optimize performance  
            if player.dino_rect.colliderect(obstacle.rect):  
                offset = (obstacle.rect.x - player.dino_rect.x, obstacle.rect.y - player.dino_rect.y)  
                collision_point = player.mask.overlap(obstacle.mask, offset)  
                if collision_point:
                    death_count += 1
                    pygame.time.delay(2000)  
                    menu(death_count)

        # Drawing everything  
        player.draw(SCREEN)  
        for obstacle in obstacles:  
            obstacle.draw(SCREEN)  

        # Debugging: Drawing collision rectangles (optional)  
        # pygame.draw.rect(SCREEN, (255, 0, 0), player.dino_rect, 2)  
        # for obstacle in obstacles:  
        #     pygame.draw.rect(SCREEN, (0, 0, 255), obstacle.rect, 2)  

        pygame.display.update()

    pygame.quit()  


def menu(death_count):
    global points
    run = True
    cnt = 0
    clock = pygame.time.Clock()
    while run:
        clock.tick(FPS)
        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render(f"Press Any Key to Continue", True, (0, 0, 0))
        SCREEN.blit(text, (400, 300))
        if death_count > 0:
            text = font.render(f"Your Score: {points}", True, (0, 0, 0))
            SCREEN.blit(text, (400, 350))
        SCREEN.blit(RUNNING[int(cnt // 5)], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
        cnt += 1
        if cnt >= 10:
            cnt = 0
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                main()

menu(death_count = 0)
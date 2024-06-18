import pygame
import numpy as np

def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)

class Car:
    def __init__(self, trajectory):
        RED_CAR = scale_image(pygame.image.load("C:/Users/Lenovo/OneDrive - Università degli Studi di Milano/MAGISTRALE/REINFORCEMENT LEARNING/Project/mdp/img/red-car.png"), 0.55)
        self.trajectory = trajectory 
        self.angle = 270
        self.x, self.y = (0, 0)
        self.img = RED_CAR
        self.frames_to_stay_per_image = 5  # Number of frames to view each image
        self.current_frames_stayed = 0
        self.current_image_index = 0
        self.current_frame = 0

    def position_to_coord(self, state):
        self.x = state % 12
        self.y = state // 12
        return (self.x * 20.5, self.y * 20.5)

    def draw_traj(self, win, background):
        if self.current_frame < len(self.trajectory) * self.frames_to_stay_per_image:
            state_index = self.current_frame // self.frames_to_stay_per_image # divisione intera
            state = self.trajectory[state_index]

            for img, pos in background:
                win.blit(img, pos)

            if state_index > 0:
                prev_state = self.trajectory[state_index - 1]
                prev_pos = self.position_to_coord(prev_state)
                curr_pos = self.position_to_coord(state)

                # Interpolation between positions for smoother movement
                interp_factor = (self.current_frame % self.frames_to_stay_per_image) / self.frames_to_stay_per_image
                interp_x = prev_pos[0] + interp_factor * (curr_pos[0] - prev_pos[0])
                interp_y = prev_pos[1] + interp_factor * (curr_pos[1] - prev_pos[1])

                # Check if it crosses prohibited cells
                # Draws a line from the previous position to the current position
                pygame.draw.line(win, (255, 0, 0), prev_pos, (interp_x, interp_y), 5)

                # Draw the car
                win.blit(self.img, (interp_x, interp_y))
            else:
                pos = self.position_to_coord(state)
                win.blit(self.img, pos)

            pygame.display.update()
            self.current_frame += 1


    def is_crossing_cell(self, start_pos, end_pos, cell):
        x1, y1 = start_pos[0] // 20.5, start_pos[1] // 20.5
        x2, y2 = end_pos[0] // 20.5, end_pos[1] // 20.5
        cx, cy = cell
        if (x1 <= cx <= x2 or x2 <= cx <= x1) and (y1 <= cy <= y2 or y2 <= cy <= y1):
            return True
        return False

def draw(trajectory):
    pygame.init()
    GRASS = scale_image(pygame.image.load("C:/Users/Lenovo/OneDrive - Università degli Studi di Milano/MAGISTRALE/REINFORCEMENT LEARNING/Project/mdp/img/grass.jpg"), 2.0)
    TRACK = scale_image(pygame.image.load("C:/Users/Lenovo/OneDrive - Università degli Studi di Milano/MAGISTRALE/REINFORCEMENT LEARNING/Project/mdp/img/track.jpg"), 1)
    FINISH = scale_image(pygame.image.load("C:/Users/Lenovo/OneDrive - Università degli Studi di Milano/MAGISTRALE/REINFORCEMENT LEARNING/Project/mdp/img/finish.png"), 0.74)
    FPS = 10  # make sure the program is not running faster than 10 frame per second
    WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption("Racing game")
    clock = pygame.time.Clock()
    background = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, (0, 81))]

    player_car = Car(trajectory)
    run = True

    while run: 
        clock.tick(FPS)
        player_car.draw_traj(WIN, background)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

    pygame.quit()
    
    

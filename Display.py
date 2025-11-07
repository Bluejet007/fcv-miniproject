import pygame
from pygame.locals import DOUBLEBUF, HWSURFACE

class Display:
    def __init__(self, W, H):
        pygame.init()

        self.screen = pygame.display.set_mode((W, H), HWSURFACE | DOUBLEBUF)
        # Only create a surface once
        self.surface = pygame.Surface((W, H)).convert()

    def display2D(self, img):
        # Remove alpha channel or homogeneous channel
        img_rgb = img[:, :, :3]

        pygame.surfarray.blit_array(self.surface, img_rgb.swapaxes(0,1))
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
  
    def clear(self):
        pygame.quit()
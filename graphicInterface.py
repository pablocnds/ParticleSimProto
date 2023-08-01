import pygame

import colors


class GraphicalInterface:
    def __init__(self, screen_size, scale=1, zero_pos=(0.5,0.5), window_name="app", set_fps=0):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption(window_name)
        self.set_fps = set_fps
        self.scale = scale
        self.offset_x = self.screen.get_width() * zero_pos[0]
        self.offset_y = self.screen.get_height() * zero_pos[1]
        self._clock = pygame.time.Clock()
        self.done = False
    

    def _manageEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                self._closeWindow()


    def _closeWindow(self):
        pygame.display.quit()
        pygame.quit()


    def _drawPoints(self, positions, radius=10):
        for p in positions:
            pygame.draw.ellipse(self.screen, colors.RED, 
                    [p[0]*self.scale+self.offset_x, p[1]*self.scale+self.offset_y, radius, radius])


    def isDone(self):
        return self.done


    def nextFrame(self, points):
        self._manageEvents()
        if self.done:
            return
        
        self.screen.fill(colors.GRAY)
        self._drawPoints(points)
        self._clock.tick(self.set_fps)
        pygame.display.flip()
    

    def dt_last_frame(self):
        return self._clock.tick()
    

    def __enter__(self):
        return self
    

    def __exit__(self, exc, value, tb):
        self._closeWindow()


    def __del__(self):
        self._closeWindow()

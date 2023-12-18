import pygame
import sys
import gym
from gym import spaces
import numpy as np
import random

# Определение пользовательского окружения Gym
class GridEnv(gym.Env):
    def __init__(self, gridsize=10):
        super(GridEnv, self).__init__()

        self.gridsize = gridsize
        self.obserspace = spaces.Discrete(self.gridsize)
        self.actspace = spaces.Discrete(4)  # 0: Вверх, 1: Вниз, 2: Влево, 3: Вправо

        self.agentposition = np.array([0, 0])
        self.visitedcells = set()

        self.reset()

    def reset(self):
        # Сброс состояния среды
        self.agentposition = np.array([0, 0])
        self.visitedcells = set()
        self._update_state()
        return self.agentposition

    def step(self, action):
        # Шаг агента в окружении в соответствии с выбранным действием
        if action == 0:  # Вверх
            self.agentposition[0] = max(0, self.agentposition[0] - 1)
        elif action == 1:  # Вниз
            self.agentposition[0] = min(self.gridsize - 1, self.agentposition[0] + 1)
        elif action == 2:  # Влево
            self.agentposition[1] = max(0, self.agentposition[1] - 1)
        elif action == 3:  # Вправо
            self.agentposition[1] = min(self.gridsize - 1, self.agentposition[1] + 1)

        self.visitedcells.add(tuple(self.agentposition))
        self._update_state()

        done = len(self.visitedcells) == self.gridsize**2  # Проверка, посетил ли агент все клетки
        reward = 0
        info = {}

        return self.agentposition, reward, done, info

    def render(self, screen):
        # Отображение текущего состояния среды в графическом режиме с использованием Pygame
        cellsize = 30

        for i in range(self.gridsize):
            for j in range(self.gridsize):
                rect = pygame.Rect(j * cellsize, i * cellsize, cellsize, cellsize)

                if (i, j) == tuple(self.agentposition):
                    pygame.draw.rect(screen, (0, 255, 0), rect)  # Зеленый цвет для агента
                elif (i, j) in self.visitedcells:
                    pygame.draw.rect(screen, (128, 128, 128), rect)  # Серый цвет для посещенных клеток
                else:
                    pygame.draw.rect(screen, (255, 255, 255), rect)  # Белый цвет для непосещенных клеток

        pygame.display.flip()

    def _update_state(self):
        # Обновление состояния агента
        state = np.array(self.agentposition)
        return state

# Пример использования среды
env = GridEnv()
pygame.init()

cellsize = 30
screensize = env.gridsize * cellsize
screen = pygame.display.set_mode((screensize, screensize))
pygame.display.set_caption("GridEnv")

clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((255, 255, 255))  # Белый фон
    env.render(screen)
    pygame.display.flip()

    # Генерация случайного действия для движения агента
    action = random.choice([0, 1, 2, 3])
    state, reward, done, _ = env.step(action)

    if done:
        print("Агент посетил все клетки.")
        pygame.quit()
        sys.exit()

    clock.tick(5)  # Регулировка скорости симуляции


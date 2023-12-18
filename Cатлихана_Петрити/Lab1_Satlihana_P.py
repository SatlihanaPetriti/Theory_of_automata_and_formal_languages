import gym  
import pygame 
import sys  
import numpy as np  

# Определение класса CustomEnv, наследующегося от класса gym.Env
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Параметры окружения
        self.WDW_SIZE = 10
        self.START_POINT = (0, 0)

        # Определение пространства состояний и действий
        self.observation_space = gym.spaces.Discrete(self.WDW_SIZE * self.WDW_SIZE)
        self.action_space = gym.spaces.Discrete(4)

        # Инициализация переменных состояния и позиции агента
        self.state = None
        self.agent_position = self.START_POINT

        # Инициализация графического интерфейса с использованием библиотеки Pygame
        pygame.init()
        self.WINDOW_SIZE = 400
        self.cell_size = self.WINDOW_SIZE // self.WDW_SIZE
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Random Movement")

    # Метод сброса среды в начальное состояние
    def reset(self):
        self.state = self.START_POINT[0] * self.WDW_SIZE + self.START_POINT[1]
        self.agent_position = self.START_POINT
        return self.state

    # Метод выполнения одного шага в среде
    def step(self, action):
        current_position = (self.state // self.WDW_SIZE, self.state % self.WDW_SIZE)
        new_position = self.state_after_action(current_position, action)

        reward = -1
        done = False

        self.state = new_position[0] * self.WDW_SIZE + new_position[1] if new_position is not None else self.state
        self.agent_position = new_position

        if self.state == self.WDW_SIZE * self.WDW_SIZE - 1:
            done = True

        return self.state, reward, done, {}

    # Метод отрисовки текущего состояния среды
    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))

        for i in range(self.WDW_SIZE + 1):
            pygame.draw.line(self.screen, (200, 200, 200), (0, i * self.cell_size), (self.WINDOW_SIZE, i * self.cell_size))
            pygame.draw.line(self.screen, (200, 200, 200), (i * self.cell_size, 0), (i * self.cell_size, self.WINDOW_SIZE))

        pygame.draw.circle(self.screen, (0, 255, 0),
                           (self.agent_position[1] * self.cell_size + self.cell_size // 2,
                            self.agent_position[0] * self.cell_size + self.cell_size // 2),
                           self.cell_size // 2)

        pygame.display.flip()

    # Метод для определения состояния после выполнения действия
    def state_after_action(self, state, action):
        if action == 0 and state[0] > 0:
            return (state[0] - 1, state[1])
        elif action == 1 and state[0] < self.WDW_SIZE - 1:
            return (state[0] + 1, state[1])
        elif action == 2 and state[1] > 0:
            return (state[0], state[1] - 1)
        elif action == 3 and state[1] < self.WDW_SIZE - 1:
            return (state[0], state[1] + 1)
        else:
            return state

# Функция для случайного перемещения агента в среде
def random_movement(env):
    episodes = 1

    for episode in range(episodes):
        state = env.reset()

        while True:
            action = np.random.choice([0, 1, 2, 3])  # Выбор случайного действия из пространства действий
            next_state, reward, done, _ = env.step(action)  # Выполнение выбранного действия

            env.render()  # Отрисовка текущего состояния среды
            pygame.time.wait(100)  # Задержка для визуализации

            state = next_state

            if done:
                print("Агент достиг каждой ячейки.")
                break

# Создание экземпляра среды
env = CustomEnv()

# Запуск случайного перемещения агента в среде
random_movement(env)

# Завершение работы Pygame и выход из программы
pygame.quit()
sys.exit()

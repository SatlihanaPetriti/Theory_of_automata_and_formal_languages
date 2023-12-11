import gym
import numpy as np
import pygame
import sys
import time

# Определение класса CustomEnv, унаследованного от gym.Env
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Параметры среды
        self.WDW_SIZE = 10  # Размер среды (10x10 сетка)
        self.NR_OBSTACLES = 3  # Количество препятствий
        self.START_POINT = (0, 0)  # Начальная точка агента
        self.GOAL_POINT = (self.WDW_SIZE - 1, self.WDW_SIZE - 1)  # Целевая точка

        # Определение пространства состояний и действий
        self.observation_space = gym.spaces.Discrete(self.WDW_SIZE * self.WDW_SIZE)
        self.action_space = gym.spaces.Discrete(4)

        # Инициализация Q-таблицы
        self.q_table = np.zeros((self.WDW_SIZE * self.WDW_SIZE, 4))

        # Дополнительная инициализация при необходимости
        self.state = None
        self.agent_position = self.START_POINT
        self.target_position = self.GOAL_POINT

        # Инициализация Pygame
        pygame.init()
        self.WINDOW_SIZE = 400  # Размер окна Pygame
        self.cell_size = self.WINDOW_SIZE // self.WDW_SIZE  # Размер ячейки среды в окне
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Q-обучение с OpenAI Gym")

        # Позиции препятствий
        self.obstacle_positions = [(2, 3), (5, 5), (8, 7)]

    # Метод сброса среды в начальное состояние
    def reset(self):
        self.state = self.START_POINT[0] * self.WDW_SIZE + self.START_POINT[1]
        self.agent_position = self.START_POINT
        return self.state

    # Метод выполнения одного шага в среде
    def step(self, action):
        current_position = (self.state // self.WDW_SIZE, self.state % self.WDW_SIZE)

        new_position = self.state_after_action(current_position, action)

        done = (new_position == self.GOAL_POINT)
        reward = -1 if not done else 10  # Награда за достижение цели (-1 за каждый шаг, 10 за достижение цели)

        self.state = new_position[0] * self.WDW_SIZE + new_position[1] if new_position is not None else self.state
        self.agent_position = new_position

        return self.state, reward, done, {}

    # Метод визуализации среды
    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))  # Белый фон

        # Отрисовка сетки
        for i in range(self.WDW_SIZE + 1):
            pygame.draw.line(self.screen, (200, 200, 200), (0, i * self.cell_size), (self.WINDOW_SIZE, i * self.cell_size))
            pygame.draw.line(self.screen, (200, 200, 200), (i * self.cell_size, 0), (i * self.cell_size, self.WINDOW_SIZE))

        # Отрисовка препятствий
        for pos in self.obstacle_positions:
            pygame.draw.rect(self.screen, (255, 0, 0), (pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size))

        # Отрисовка целевой точки
        pygame.draw.circle(self.screen, (255, 255, 0),
                           (self.GOAL_POINT[1] * self.cell_size + self.cell_size // 2,
                            self.GOAL_POINT[0] * self.cell_size + self.cell_size // 2),
                           self.cell_size // 2)

        # Отрисовка текущей позиции агента
        pygame.draw.circle(self.screen, (0, 255, 0),
                           (self.agent_position[1] * self.cell_size + self.cell_size // 2,
                            self.agent_position[0] * self.cell_size + self.cell_size // 2),
                           self.cell_size // 2)

        pygame.display.flip()

    # Метод вычисления следующего состояния после выполнения действия
    def state_after_action(self, state, action):
        if action == 0 and state[0] > 0 and (state[0] - 1, state[1]) not in self.obstacle_positions:
            return (state[0] - 1, state[1])
        elif action == 1 and state[0] < self.WDW_SIZE - 1 and (state[0] + 1, state[1]) not in self.obstacle_positions:
            return (state[0] + 1, state[1])
        elif action == 2 and state[1] > 0 and (state[0], state[1] - 1) not in self.obstacle_positions:
            return (state[0], state[1] - 1)
        elif action == 3 and state[1] < self.WDW_SIZE - 1 and (state[0], state[1] + 1) not in self.obstacle_positions:
            return (state[0], state[1] + 1)
        else:
            return state

# Функция обработки событий Pygame
def pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

# Функция выбора действия агента с учетом вероятности исследования
def choose_action(state, exploration_prob):
    if np.random.rand() < exploration_prob:
        return np.random.choice([0, 1, 2, 3])  # Случайный выбор действия с вероятностью исследования
    else:
        return np.argmax(env.q_table[state, :])  # Выбор действия с наивысшим Q-значением для данного состояния

# Функция Q-обучения
def q_learning(env):
    learning_rate = 0.1  # Коэффициент обучения
    discount_factor = 0.9  # Коэффициент дисконтирования
    initial_exploration_prob = 0.8  # Начальная вероятность исследования
    episodes = 500  # Количество эпизодов

    for episode in range(episodes):
        state = env.reset()  # Начальное состояние среды
        exploration_prob = initial_exploration_prob / (episode + 1)  # Уменьшение вероятности исследования с течением времени

        start_time = time.time()

        while True:
            action = choose_action(state, exploration_prob)
            next_state, reward, done, _ = env.step(action)

            # Обновление Q-значения с использованием формулы Q-обучения
            env.q_table[state, action] = (1 - learning_rate) * env.q_table[state, action] + \
                                          learning_rate * (reward + discount_factor * np.max(env.q_table[next_state, :]))

            env.render()
            pygame_events()
            pygame.time.wait(100)  # Пауза для визуализации

            state = next_state

            if done:
                end_time = time.time()
                time_taken = end_time - start_time
                print(f"Эпизод {episode + 1} | Время: {time_taken:.2f} секунды | Вероятность исследования: {exploration_prob:.2f}")
                break

# Создание экземпляра пользовательской среды
env = CustomEnv()

# Запуск Q-обучения
q_learning(env)

# Закрытие окна Pygame
pygame.quit()
sys.exit()

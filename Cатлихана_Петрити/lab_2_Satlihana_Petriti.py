import numpy as np
import matplotlib.pyplot as plt
import pygame
import random
import time

# Настройка окружения
WDW_SIZE = 10 # размер сетки (WDW_SIZE)
NR_OBSTACLES = 3 # количество препятствий (NR_OBSTACLES) 
START_POINT = (0, 0)# начальная точка (START_POINT)
GOAL_POINT = (WDW_SIZE - 1, WDW_SIZE - 1) # конечная точка (GOAL_POINT)
 
# Параметры Q-обучения
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.2
episodes = 500

# Инициализация Q-таблицы
q_table = np.zeros((WDW_SIZE, WDW_SIZE, 4))  # 4 действия: вверх, вниз, влево, вправо

# Добавление препятствий в окружение
obstacle_positions = [(2, 3), (5, 5), (8, 7)]
for pos in obstacle_positions:
    q_table[pos[0], pos[1], :] = -np.inf  # Сделать препятствия непроходимыми

# Константы для действий
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Инициализация Pygame
pygame.init()

# Настройка окна
WINDOW_SIZE = 400
cell_size = WINDOW_SIZE // WDW_SIZE
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Окружение для Q-обучения")

# Эта функция отвечает за рисование окружения. Она включает в себя сетку, препятствия, цель и текущее положение агента.
def draw_environment(agent_pos):
    screen.fill((255, 255, 255))  # Белый фон

    # Рисование сетки
    for i in range(WDW_SIZE + 1):
        pygame.draw.line(screen, (200, 200, 200), (0, i * cell_size), (WINDOW_SIZE, i * cell_size))
        pygame.draw.line(screen, (200, 200, 200), (i * cell_size, 0), (i * cell_size, WINDOW_SIZE))

    # Рисование препятствий
    for pos in obstacle_positions:
        pygame.draw.rect(screen, (255, 0, 0), (pos[1] * cell_size, pos[0] * cell_size, cell_size, cell_size))

    # Рисование цели (цвет на желтый)
    pygame.draw.circle(screen, (255, 255, 0),
                       (GOAL_POINT[1] * cell_size + cell_size // 2, GOAL_POINT[0] * cell_size + cell_size // 2),
                       cell_size // 2)

    # Рисование агента
    pygame.draw.circle(screen, (0, 255, 0),
                       (agent_pos[1] * cell_size + cell_size // 2, agent_pos[0] * cell_size + cell_size // 2),
                       cell_size // 2)

    pygame.display.flip()

# Эта функция обрабатывает события Pygame. 
# она проверяет наличие события QUIT (закрытие окна) и соответствующим образом завершает работу программы.
def pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

# Функция для ожидания некоторого времени
def wait():
    pygame.time.wait(100)

# Функция для выбора действия на основе значения Q
def choose_action(state):
    if random.uniform(0, 1) < exploration_prob:
        return random.choice([UP, DOWN, LEFT, RIGHT])
    else:
        # Эта строка возвращает действие с наибольшим значением Q для данного состояния
        return np.argmax(q_table[state[0], state[1], :])

# Эта функция реализует алгоритм Q-learning. 
# Она выполняет итерацию по эпизодам, обновляя значения Q на основе взаимодействий агента с окружающей средой.
def q_learning():
    for episode in range(episodes):
        start_time = pygame.time.get_ticks()  # Запись времени начала эпизода
        state = START_POINT
        while True:
            action = choose_action(state)
            new_state = state_after_action(state, action)
            reward = -1

            if new_state == GOAL_POINT:
                reward = 10
                end_time = pygame.time.get_ticks()  # Запись времени окончания при достижении цели
                time_taken = (end_time - start_time) / 1000.0  # Преобразование миллисекунд в секунды
                print(f"Цель {episode + 1}: {time_taken} секунд")
            #Эта строка обновляет Q-значение для текущей пары состояние-действие, используя формулу Q-learning
            q_table[state[0], state[1], action] = (1 - learning_rate) * q_table[state[0], state[1], action] + \
                                                  learning_rate * (
                                                          reward + discount_factor * np.max(
                                                      q_table[new_state[0], new_state[1], :]))

            state = new_state
            draw_environment(state)
            wait()
            pygame_events()

            if state == GOAL_POINT:
                break

# Функция для получения следующего состояния после выполнения действия
def state_after_action(state, action):
    if action == UP and state[0] > 0 and q_table[state[0] - 1, state[1], action] != -np.inf:
        return (state[0] - 1, state[1])
    elif action == DOWN and state[0] < WDW_SIZE - 1 and q_table[state[0] + 1, state[1], action] != -np.inf:
        return (state[0] + 1, state[1])
    elif action == LEFT and state[1] > 0 and q_table[state[0], state[1] - 1, action] != -np.inf:
        return (state[0], state[1] - 1)
    elif action == RIGHT and state[1] < WDW_SIZE - 1 and q_table[state[0], state[1] + 1, action] != -np.inf:
        return (state[0], state[1] + 1)
    else:
        return state
    # Эта строка возвращает новое состояние на основе действия. 
    # Если действие допустимо, оно обновляет состояние; в противном случае оно остается в текущем состоянии.  

# Запуск алгоритма Q-обучения
# Агент учится перемещаться от начальной точки к цели, избегая препятствий.
q_learning()


import pygame
import random
import numpy as np

# Define constants
width, height = 500, 500 # Ширина и высота окна отображения
grid_size = 10# Размер сетки
cell_size = width // grid_size # Размер ячейки на сетке
posis_goal = (grid_size - 1, grid_size - 1)  # Позиция цели
posis_agent = (0, 0)  # Позиция агента (начальная позиция)
posis_obstac = [(2, 3), (7, 3), (7, 7)]# Позиции препятствий
movement = ["UP", "DOWN", "LEFT", "RIGHT"] # Возможные действия агента
rate_learning = 0.1       # Скорость обучения для обновления Q-таблицы
factor_disc = 0.9    # Фактор дисконтирования для учета будущих наград
epsilon = 0.1     # Параметр для стратегии epsilon-greedy
nr_episodes = 1000   # Количество эпизодов обучения

#Код инициализирует Q-таблицу нулями, где каждая запись представляет Q-значение
#для определенной пары состояние-действие.
# Инициализация Q-таблицы
q_table = np.zeros((grid_size, grid_size, len(movement)))

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((width, height)) # Создание окна отображения
pygame.display.set_caption("Q-Learning Environment") # Установка заголовка окна

# Отрисовка сетки
def grid_draw():
    for i in range(1, grid_size):
        pygame.draw.line(screen, pygame.Color("black"), (i * cell_size, 0), (i * cell_size, height))
        pygame.draw.line(screen, pygame.Color("black"), (0, i * cell_size), (width, i * cell_size))

# Отрисовка агента
def agent_draw(pos):
    pygame.draw.circle(screen, pygame.Color("black"), (pos[0] * cell_size + cell_size // 2, pos[1] * cell_size + cell_size // 2), cell_size // 2)

# Отрисовка цели
def goal_draw(pos):
    pygame.draw.rect(screen, pygame.Color("red"), (pos[0] * cell_size, pos[1] * cell_size, cell_size, cell_size))

# Отрисовка препятствий
def obstacle_draw(obstacle_positions):
    for pos in obstacle_positions:
        pygame.draw.circle(screen, pygame.Color("yellow"), (pos[0] * cell_size + cell_size // 2, pos[1] * cell_size + cell_size // 2), cell_size // 2)


#Функция q_learningupdate реализует правило обновления Q-learning. 
# #Она вычисляет новое значение Q на основе текущего значения Q, 
# #полученного вознаграждения и максимального значения Q для следующего состояния.
# Обновление Q-таблицы по алгоритму Q-обучения
def q_learningupdate(state, action, reward, next_state):
    current_q = q_table[state[0]][state[1]][movement.index(action)] # Текущее значение Q-функции для данного состояния и действия
    best_next_q = np.max(q_table[next_state[0]][next_state[1]])# Максимальное значение Q-функции для следующего состояния
    new_q = current_q + rate_learning * (reward + factor_disc * best_next_q - current_q)  # Обновление Q-значения
    q_table[state[0]][state[1]][movement.index(action)] = new_q    # Присвоение нового значения в Q-таблицу

#Функция getnextaction выбирает следующее действие, используя эпсилон-жадную стратегию. 
# Она выбирает случайное действие с вероятностью ЭПСИЛОН или выбирает действие с максимальным значением Q в противном случае.
# Получение следующего действия с использованием стратегии epsilon-greedy
def getnextaction(state):
    if random.uniform(0, 1) < epsilon: # Случайное действие с вероятностью epsilon
        return random.choice(movement)
    else:
        return movement[np.argmax(q_table[state[0]][state[1]])] # Действие с максимальным Q-значением с вероятностью (1 - epsilon)

# Основной цикл
running = True  # Флаг для работы основного цикла
clock = pygame.time.Clock() # Инициализация объекта Clock для управления временем в игре

for episode in range(nr_episodes): # Цикл по эпизодам обучения
    state = posis_agent # Установка начального состояния агента


    while state != posis_goal:   # Цикл до достижения цели в текущем эпизоде
        screen.fill(pygame.Color("white")) # Очистка экрана
        grid_draw()  # Отрисовка сетки
        obstacle_draw(posis_obstac)  # Отрисовка препятствий
        goal_draw(posis_goal)   # Отрисовка цели
        agent_draw(state)  # Отрисовка агента
        pygame.display.flip()  # Обновление отображения на экране

        action = getnextaction(state)  # Получение следующего действия агента

        next_state = state
        if action == "UP" and state[1] > 0 and (state[0], state[1] - 1) not in posis_obstac:
            next_state = (state[0], state[1] - 1)
        elif action == "DOWN" and state[1] < grid_size - 1 and (state[0], state[1] + 1) not in posis_obstac:
            next_state = (state[0], state[1] + 1)
        elif action == "LEFT" and state[0] > 0 and (state[0] - 1, state[1]) not in posis_obstac:
            next_state = (state[0] - 1, state[1])
        elif action == "RIGHT" and state[0] < grid_size - 1 and (state[0] + 1, state[1]) not in posis_obstac:
            next_state = (state[0] + 1, state[1])

        reward = -1   # Награда за каждый шаг (по умолчанию)

        if next_state == posis_goal:
            reward = 100  # Награда за достижение цели
            print("Goal reached in episode:", episode)

        q_learningupdate(state, action, reward, next_state) # Обновление Q-таблицы по алгоритму Q-обучения
        state = next_state  # Переход к следующему состоянию

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(10)   # Регулировка скорости симуляции

pygame.quit() # Завершение Pygame
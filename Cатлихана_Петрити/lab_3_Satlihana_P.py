import gym
import numpy as np
import pygame
import sys
import time
from stable_baselines3 import PPO

# Определение кастомной среды, унаследованной от gym.Env
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Параметры среды
        self.WDW_SIZE = 10  # Размер среды (10x10 сетка)
        self.START_POINT = (0, 0)  # Начальное положение агента
        self.GOAL_POINT = (self.WDW_SIZE - 1, self.WDW_SIZE - 1)  # Целевая позиция

        # Определение пространства состояний и действий для первого агента
        self.observation_space = gym.spaces.Discrete(self.WDW_SIZE * self.WDW_SIZE)
        self.action_space = gym.spaces.Discrete(4)

        # Инициализация Q-таблицы для первого агента
        self.q_table_agent1 = np.zeros((self.WDW_SIZE * self.WDW_SIZE, 4))

        # Определение пространства состояний и действий для второго агента
        self.observation_space_agent2 = gym.spaces.Discrete(self.WDW_SIZE * self.WDW_SIZE)
        self.action_space_agent2 = gym.spaces.Discrete(4)

        # Инициализация Q-таблицы для второго агента
        self.q_table_agent2 = np.zeros((self.WDW_SIZE * self.WDW_SIZE, 4))

        # Дополнительная инициализация, если необходимо
        self.state_agent1 = None
        self.state_agent2 = None
        self.agent_position_agent1 = self.START_POINT
        self.agent_position_agent2 = self.START_POINT
        self.target_position_agent1 = self.GOAL_POINT
        self.target_position_agent2 = self.GOAL_POINT

        middle_row = self.WDW_SIZE // 2
        middle_col = self.WDW_SIZE // 2
        self.target_path = [(i, j) for i in range(middle_row - 2, middle_row + 3) for j in range(middle_col - 2, middle_col + 3)]
        self.target_path_index = 0

        # Инициализация Pygame
        pygame.init()
        self.WINDOW_SIZE = 400  # Размер окна Pygame
        self.cell_size = self.WINDOW_SIZE // self.WDW_SIZE  # Размер ячейки сетки в окне
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Q-learning and PPO with OpenAI Gym")

    # Метод сброса среды к начальному состоянию
    def reset(self):
        # Сброс состояния и целевой позиции для первого агента
        self.state_agent1 = self.START_POINT[0] * self.WDW_SIZE + self.START_POINT[1]
        self.target_position_agent1 = self.target_path[0]
        self.agent_position_agent1 = self.START_POINT

        # Сброс состояния и целевой позиции для второго агента
        self.state_agent2 = self.START_POINT[0] * self.WDW_SIZE + self.START_POINT[1]
        self.target_position_agent2 = self.target_path[0]
        self.agent_position_agent2 = self.START_POINT

        # Сброс индекса пути для обоих агентов
        self.target_path_index = 0

        return self.state_agent1, self.state_agent2

    # Метод выполнения шага в среде
    def step(self, action_agent1, action_agent2):
        # Шаг для первого агента
        current_position_agent1 = (self.state_agent1 // self.WDW_SIZE, self.state_agent1 % self.WDW_SIZE)
        self.target_position_agent1 = self.target_path[self.target_path_index]
        self.target_path_index = (self.target_path_index + 1) % len(self.target_path)
        new_position_agent1 = self.state_after_action(current_position_agent1, action_agent1)

        done_agent1 = (new_position_agent1 == self.target_position_agent1)
        reward_agent1 = -1 if not done_agent1 else 10

        self.state_agent1 = new_position_agent1[0] * self.WDW_SIZE + new_position_agent1[1] if new_position_agent1 is not None else self.state_agent1
        self.agent_position_agent1 = new_position_agent1

        # Шаг для второго агента
        current_position_agent2 = (self.state_agent2 // self.WDW_SIZE, self.state_agent2 % self.WDW_SIZE)
        self.target_position_agent2 = self.target_path[self.target_path_index]
        self.target_path_index = (self.target_path_index + 1) % len(self.target_path)
        new_position_agent2 = self.state_after_action(current_position_agent2, action_agent2)

        done_agent2 = (new_position_agent2 == self.target_position_agent2)
        reward_agent2 = -1 if not done_agent2 else 10

        self.state_agent2 = new_position_agent2[0] * self.WDW_SIZE + new_position_agent2[1] if new_position_agent2 is not None else self.state_agent2
        self.agent_position_agent2 = new_position_agent2

        return (self.state_agent1, self.state_agent2), (reward_agent1, reward_agent2), (done_agent1, done_agent2), {}

    # Метод отображения среды
    def render(self, mode='human'):
        # Очистка экрана
        self.screen.fill((255, 255, 255))
        
        # Отрисовка сетки
        for i in range(self.WDW_SIZE + 1):
            pygame.draw.line(self.screen, (200, 200, 200), (0, i * self.cell_size), (self.WINDOW_SIZE, i * self.cell_size))
            pygame.draw.line(self.screen, (200, 200, 200), (i * self.cell_size, 0), (i * self.cell_size, self.WINDOW_SIZE))

        # Отрисовка цели для первого агента (желтый круг)
        pygame.draw.circle(self.screen, (255, 255, 0),
                           (self.target_position_agent1[1] * self.cell_size + self.cell_size // 2,
                            self.target_position_agent1[0] * self.cell_size + self.cell_size // 2),
                           self.cell_size // 2)

        # Отрисовка агента для первого агента (зеленый круг)
        pygame.draw.circle(self.screen, (0, 255, 0),
                           (self.agent_position_agent1[1] * self.cell_size + self.cell_size // 2,
                            self.agent_position_agent1[0] * self.cell_size + self.cell_size // 2),
                           self.cell_size // 2)

        # Отрисовка агента для второго агента (красный круг)
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (self.agent_position_agent2[1] * self.cell_size + self.cell_size // 2,
                            self.agent_position_agent2[0] * self.cell_size + self.cell_size // 2),
                           self.cell_size // 2)

        pygame.display.flip()

    # Метод определения состояния после действия
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

# Функция для выравнивания состояния
def flatten_state(state):
    return np.array(state).flatten()

# Обработка событий Pygame
def pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

# Функция выбора действия агента
def choose_action(state, exploration_prob, q_table):
    if np.random.rand() < exploration_prob:
        return np.random.choice([0, 1, 2, 3])
    else:
        return np.argmax(q_table[state, :])


# Функция обучения с использованием Q-learning
def q_learning(env):
    learning_rate = 0.1
    discount_factor = 0.9
    initial_exploration_prob = 0.8
    episodes = 10

    q_learning_total_rewards = []

    # Итерация по эпизодам
    for episode in range(episodes):
        state = env.reset()
        exploration_prob = initial_exploration_prob / (episode + 1)
        start_time = time.time()
        total_reward_agent1 = 0

        # Цикл для выполнения шагов внутри эпизода
        while True:
            # Выбор действий для обоих агентов с учетом исследования
            action_agent1 = choose_action(state[0], exploration_prob, env.q_table_agent1)
            action_agent2 = choose_action(state[1], exploration_prob, env.q_table_agent2)
            
            # Выполнение шага в среде
            next_state, reward, done, _ = env.step(action_agent1, action_agent2)

            # Обновление Q-таблицы для обоих агентов
            env.q_table_agent1[state[0], action_agent1] = (1 - learning_rate) * env.q_table_agent1[state[0], action_agent1] + \
                                                           learning_rate * (reward[0] + discount_factor * np.max(env.q_table_agent1[next_state[0], :]))

            env.q_table_agent2[state[1], action_agent2] = (1 - learning_rate) * env.q_table_agent2[state[1], action_agent2] + \
                                                           learning_rate * (reward[1] + discount_factor * np.max(env.q_table_agent2[next_state[1], :]))

            total_reward_agent1 += reward[0]

            # Отображение среды, обработка событий Pygame и задержка
            env.render()
            pygame_events()
            pygame.time.wait(100)

            state = next_state

            # Проверка завершения эпизода для обоих агентов
            if done[0] or done[1]:
                end_time = time.time()
                time_taken = end_time - start_time
                q_learning_total_rewards.append(total_reward_agent1)
                
                # Вывод результатов эпизода Q-learning
                print(f"Q-learning Агенты | Эпизод {episode + 1} | Время: {time_taken:.2f} секунд | Вероятность исследования: {exploration_prob:.2f} | Общий вознаграждение (Агент 1): {total_reward_agent1}")
                break

    return q_learning_total_rewards


# Функция обучения с использованием PPO
def train_ppo_agent(env, num_episodes=10, learning_rate=0.00025, ent_coef=0.01, discount_factor=0.99):
    # Настройка гиперпараметров для PPO
    model_agent1 = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate, ent_coef=ent_coef, gamma=discount_factor, net_arch=[64, 64])
    model_agent2 = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate, ent_coef=ent_coef, gamma=discount_factor, net_arch=[64, 64])

    ppo_total_rewards = []

    # Итерация по эпизодам для PPO
    for episode in range(num_episodes):
        state = env.reset()
        done = (False, False)
        total_reward_agent2 = 0

        # Цикл для выполнения шагов внутри эпизода для PPO
        while not done[0] and not done[1]:
            # Предсказание действий PPO для обоих агентов
            action_agent1, _ = model_agent1.predict(flatten_state(state[0]), deterministic=False)
            action_agent2, _ = model_agent2.predict(flatten_state(state[1]), deterministic=False)
            
            # Выполнение шага в среде
            next_state, reward, done, _ = env.step(action_agent1, action_agent2)

            # Обучение PPO на каждом шаге
            model_agent1.learn(total_timesteps=1)
            model_agent2.learn(total_timesteps=1)

            total_reward_agent2 += reward[1]
            state = next_state

            # Отображение среды и обработка событий Pygame
            env.render()
            pygame_events()

        # Вывод общего вознаграждения PPO-агента
        print(f"PPO Агенты | Эпизод {episode + 1} | Общее вознаграждение (Агент 2): {total_reward_agent2}")

        # Проверка достижения цели
        if done[0] or done[1]:
            ppo_total_rewards.append(total_reward_agent2)
            print(f"PPO Агенты | Эпизод {episode + 1} | Агент {'1' if done[0] else '2'} достиг цели!")

            # Сброс и поиск, возможно, более быстрого пути
            state = env.reset()
            done = (False, False)

    env.close()
    return ppo_total_rewards

# Основная часть программы
if __name__ == "__main__":
    env = CustomEnv()

    # Обучение агентов с использованием Q-learning
    q_learning_rewards = q_learning(env)

    # Обучение агентов с использованием PPO
    ppo_rewards = train_ppo_agent(env)

    pygame.quit()
    sys.exit()


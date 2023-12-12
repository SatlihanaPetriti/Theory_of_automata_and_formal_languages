import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Определение агента Q-обучения
class QLearningAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_values = np.zeros(action_space.n)

    def select_action(self, state):
        return np.argmax(self.q_values)

    def update_q_values(self, state, action, reward, next_state, done):
        # Базовое обновление Q-обучения
        if not done:
            next_max_q_value = np.max(self.q_values[next_state])
            target = reward + self.gamma * next_max_q_value
        else:
            target = reward

        delta = target - self.q_values[state][action]
        self.q_values[state][action] += delta

# Определение агента PPO
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(300, 128)
        self.fc2 = nn.Linear(128, 4)  # Предполагается 4 действия

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=0)

class PPOAgent:
    def __init__(self):
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.gamma = 0.99
        self.eps_clip = 0.2

    def select_action(self, state):
        state = torch.from_numpy(state).float().view(-1)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update_policy(self, log_probs, rewards):
        # Базовое обновление политики PPO
        returns = sum(rewards)

        policy_loss = -log_probs * returns
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

# Модифицированный класс окружения для работы с двумя разными агентами
class MultiAgentEnv(gym.Env):
    def __init__(self, agent1, agent2):
        super(MultiAgentEnv, self).__init__()
        self.agent1 = agent1
        self.agent2 = agent2
        self.actspace = gym.spaces.Discrete(4)  # Предполагается 4 действия
        self.agent1pos = [0, 0]
        self.agent2pos = [9, 9]
        self.targetpos = [5, 5]
        self.targetmovrange = 1
        self.viewer = None

    def resetep(self):
        self.agent1pos = [0, 0]
        self.agent2pos = [9, 9]
        self.retargetpos()
        return [self.getobserv(), self.getobserv()]

    def step(self, act):
        act_agent1, _ = act[0], None
        act_agent2, log_prob_agent2 = self.agent2.select_action(self.getobserv())

        self.agent1pos = self.updpos(self.agent1pos, act_agent1)
        self.agent2pos = self.updpos(self.agent2pos, act_agent2)
        self.mvtarget()

        disagent1 = np.linalg.norm(np.array(self.agent1pos) - np.array(self.targetpos))
        disagent2 = np.linalg.norm(np.array(self.agent2pos) - np.array(self.targetpos))

        done = disagent1 < 1.0 or disagent2 < 1.0

        rewagent1 = 1.0 if disagent1 < 1.0 else 0.0
        rewagent2 = 1.0 if disagent2 < 1.0 else 0.0

        return [self.getobserv(), self.getobserv()], [rewagent1, rewagent2], done, log_prob_agent2

    def getobserv(self):
        observ = np.zeros((10, 10, 3), dtype=np.uint8)
        observ[self.agent1pos[0], self.agent1pos[1], 0] = 255
        observ[self.agent2pos[0], self.agent2pos[1], 1] = 255
        observ[self.targetpos[0], self.targetpos[1], 2] = 255
        return observ

    def updpos(self, pos, action):
        if action == 0:
            pos[0] = max(0, pos[0] - 1)
        elif action == 1:
            pos[0] = min(9, pos[0] + 1)
        elif action == 2:
            pos[1] = max(0, pos[1] - 1)
        elif action == 3:
            pos[1] = min(9, pos[1] + 1)
        return pos

    def retargetpos(self):
        self.targetpos = [np.random.randint(0, 10), np.random.randint(0, 10)]

    def mvtarget(self):
        new_target_pos = [
            np.clip(self.targetpos[0] + np.random.randint(-self.targetmovrange, self.targetmovrange + 1), 0, 9),
            np.clip(self.targetpos[1] + np.random.randint(-self.targetmovrange, self.targetmovrange + 1), 0, 9)
        ]
        self.targetpos = new_target_pos

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = plt.imshow(self.getobserv())
        else:
            self.viewer.set_data(self.getobserv())
        plt.pause(0.1)

# Создание агента Q-обучения
q_learning_agent = QLearningAgent(gym.spaces.Discrete(4))

# Создание агента PPO
ppo_agent = PPOAgent()

# Создание мультиагентного окружения с обоими агентами
multi_agent_env = MultiAgentEnv(q_learning_agent, ppo_agent)

nrepisod = 5

for episode in range(nrepisod):
    obs = multi_agent_env.resetep()
    done = False

    while not done:
        action_agent1 = q_learning_agent.select_action(obs[0])
        action_agent2, log_prob_agent2 = ppo_agent.select_action(obs[1])
        act = [action_agent1, action_agent2]

        obs, rewards, done, log_prob_agent2 = multi_agent_env.step(act)
        multi_agent_env.render()

    if rewards[0] > rewards[1]:
        print(f"Эпизод {episode + 1}: Агент 1 достиг цели первым.")
    elif rewards[1] > rewards[0]:
        print(f"Эпизод {episode + 1}: Агент 2 достиг цели первым.")
    else:
        print(f"Эпизод {episode + 1}: Оба агента достигли цели одновременно.")

plt.show()

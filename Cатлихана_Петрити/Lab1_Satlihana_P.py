import random

class Grid:
    def __init__(self, size):
        # Инициализация сетки заданного размера
        self.size = size
        self.grid10 = [['.' for _ in range(size)] for _ in range(size)]

class Agent:
    def __init__(self, grid10):
        # Инициализация агента с указанием сетки
        self.grid10 = grid10
        # Установка случайной начальной позиции агента
        self.row = random.randint(0, grid10.size - 1)
        self.column = random.randint(0, grid10.size - 1)

    def move(self):
        # Выбор случайного направления движения
        direction = random.choice(['up', 'down', 'left', 'right'])

        # Проверка и выполнение движения в выбранном направлении
        if direction == 'up' and self.row > 0:
            self.row -= 1
        elif direction == 'down' and self.row < self.grid10.size - 1:
            self.row += 1
        elif direction == 'left' and self.column > 0:
            self.column -= 1
        elif direction == 'right' and self.column < self.grid10.size - 1:
            self.column += 1

# Создание сетки размера 10x10
grid10 = Grid(10)

# Инициализация агента с передачей ему сетки
agent = Agent(grid10)

# Множество для отслеживания посещенных клеток
cellvisit = set()

# Цикл, пока не посетят все клетки сетки
while len(cellvisit) < grid10.size * grid10.size:
    # Добавление текущей позиции агента в множество посещенных клеток
    cellvisit.add((agent.row, agent.column))
    # Перемещение агента
    agent.move()

# Вывод сетки с отметкой пути агента ('S'), посещенных клеток ('.'), и не посещенных клеток ('0')
for row in range(grid10.size):
    for column in range(grid10.size):
        if (row, column) == (agent.row, agent.column):
            print('S', end=' ')
        elif (row, column) in cellvisit:
            print('.', end=' ')
        else:
            print('0', end=' ')
    print()

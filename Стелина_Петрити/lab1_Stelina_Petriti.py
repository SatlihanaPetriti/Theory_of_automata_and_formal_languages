import random
# Создать сетку 10x10, заполненную нулями
grid10x10 = [[0 for _ in range(10)] for _ in range(10)]

# Разместить агента в случайной начальной позиции
rowagenta = random.randint(0, 9)
colagenta = random.randint(0, 9)

# Функция для случайного перемещения агента
def dvizheniye_agenta():
    global rowagenta, colagenta  # объявить как глобальную
    direc = random.choice(['up', 'down', 'left', 'right'])

    if direc == 'up' and rowagenta > 0:
        rowagenta -= 1
    elif direc == 'down' and rowagenta < 9:
        rowagenta += 1
    elif direc == 'left' and colagenta > 0:
        colagenta -= 1
    elif direc == 'right' and colagenta < 9:
        colagenta += 1

# Двигать агента случайным образом до достижения каждой ячейки
poseshchennye_cells = set()

while len(poseshchennye_cells) < 100:
    dvizheniye_agenta()
    poseshchennye_cells.add((rowagenta, colagenta))

# Вывести конечную сетку с путем агента
for row in range(10):
    for col in range(10):
        if (row, col) == (rowagenta, colagenta):
            print('+', end=' ')
        elif (row, col) in poseshchennye_cells:
            print('.', end=' ')
        else:
            print('0', end=' ')
    print()

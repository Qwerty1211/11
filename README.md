# 11
import torch
import torch.nn as nn
import torch.optim as optim

# Определяем модель
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Входной слой
        self.fc2 = nn.Linear(50, 1)   # Выходной слой
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  
                x = self.fc2(x)
        return x

model = SimpleNN()

# Данные для обучения (входные и целевые значения)
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# Функция потерь и оптимизатор
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Цикл обучения
for epoch in range(100):
    optimizer.zero_grad()  # Обнуление градиентов
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()        # Вычисление градиентов
    optimizer.step()       # Обновление весов
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
        
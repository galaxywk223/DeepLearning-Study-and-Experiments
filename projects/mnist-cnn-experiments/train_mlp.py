import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

# 加载数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP()

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 数据加载器
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 测试准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {100 * correct / total:.2f}%")

import matplotlib.pyplot as plt  # 新增导入
# 新增可视化部分
model.eval()
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)

# 预测
outputs = model(images)
_, preds = torch.max(outputs, 1)

# 将数据转移到CPU并反归一化
images = images.cpu().numpy()
labels = labels.cpu().numpy()
preds = preds.cpu().numpy()

# 反归一化处理并调整形状
images = images * 0.3081 + 0.1307  # 恢复原始像素范围
images = images.reshape(-1, 28, 28)  # 调整形状为28x28

# 可视化前5张图片
num_images = 5
fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
for i in range(num_images):
    ax = axes[i]
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f'True: {labels[i]}\nPred: {preds[i]}', fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 数据增强和预处理
def get_transforms():
    return transforms.Compose([
        transforms.RandomRotation(5),  # 添加数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


# 数据加载封装
def get_dataloaders(batch_size=64):  # 增大batch_size
    transform = get_transforms()

    train_set = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# 模型定义优化
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),  # 添加批归一化
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),  # 减少全连接层维度
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加Dropout
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# 训练和评估函数
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# 可视化辅助函数
def visualize_predictions(model, loader, num_images=5):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    images = images.cpu().numpy()
    images = images * 0.3081 + 0.1307  # 反归一化
    images = images.squeeze()

    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'True: {labels[i]}\nPred: {preds[i]}', fontsize=9)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


# 主训练流程
def main():
    train_loader, test_loader = get_dataloaders()

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # 使用AdamW
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # 学习率调度

    best_acc = 0.0

    for epoch in range(15):  # 增加epoch数
        train_loss = train(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step(test_acc)

        print(f"Epoch {epoch + 1:02d}:")
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best Test Accuracy: {best_acc * 100:.2f}%")

    # 可视化预测结果
    visualize_predictions(model, test_loader)

    # 加载最佳模型进行演示
    model.load_state_dict(torch.load("best_model.pth"))
    final_acc = evaluate(model, test_loader, criterion)[1]
    print(f"Final Model Accuracy: {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()

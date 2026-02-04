import torch
from torchvision import transforms, datasets
from torch import nn
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = CNN()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 32

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {100 * correct / total:.2f}%")

import matplotlib.pyplot as plt

model.eval()
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)

outputs = model(images)
_, preds = torch.max(outputs, 1)

images = images.cpu().numpy()
labels = labels.cpu().numpy()
preds = preds.cpu().numpy()

images = images * 0.3081 + 0.1307
images = images.reshape(-1, 28, 28)

num_images = 5
fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
for i in range(num_images):
    ax = axes[i]
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f'True: {labels[i]}\nPred: {preds[i]}', fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

single_image, single_label = test_dataset[0]
single_image = single_image.unsqueeze(0).to(device)

outputs = {}
x = single_image
outputs['input'] = x.detach().cpu()

# Conv1
x = model.conv1(x)
outputs['conv1'] = x.detach().cpu()
x = model.relu(x)
outputs['relu1'] = x.detach().cpu()
x = model.pool(x)
outputs['pool1'] = x.detach().cpu()

# Conv2
x = model.conv2(x)
outputs['conv2'] = x.detach().cpu()
x = model.relu(x)
outputs['relu2'] = x.detach().cpu()
x = model.pool(x)
outputs['pool2'] = x.detach().cpu()

# 全连接层
x = x.view(x.size(0), -1)
x = model.fc1(x)
outputs['fc1'] = x.detach().cpu()
x = model.relu(x)
outputs['relu_fc1'] = x.detach().cpu()
x = model.fc2(x)
outputs['fc2'] = x.detach().cpu()

pred = torch.argmax(x, dim=1).item()

# 反归一化输入图像
input_image = outputs['input'].squeeze().numpy()
input_image = input_image * 0.3081 + 0.1307


def visualize_layer(output, layer_name, num_channels=8):
    if len(output.shape) == 4:  # (B, C, H, W)
        channels = output[0].detach().numpy()
        num_rows = int(np.sqrt(num_channels))
        num_cols = (num_channels + num_rows - 1) // num_rows

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        for i in range(num_channels):
            ax = axes.flat[i]
            img = channels[i]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Ch.{i}')
            ax.axis('off')
        plt.suptitle(f'{layer_name} ({channels.shape[0]} channels)')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Layer {layer_name} output shape not supported for visualization.")


plt.figure(figsize=(3, 3))
plt.imshow(input_image,cmap='gray')
plt.title(f'Original Image: {single_label}')
plt.axis('off')
plt.show()

visualize_layer(outputs['conv1'], 'Conv1 Layer', num_channels=16)
visualize_layer(outputs['pool1'], 'MaxPool1 Layer', num_channels=16)
visualize_layer(outputs['conv2'], 'Conv2 Layer', num_channels=32)
visualize_layer(outputs['pool2'], 'MaxPool2 Layer', num_channels=32)

print(f"模型预测结果: {pred}, 真实标签: {single_label}")

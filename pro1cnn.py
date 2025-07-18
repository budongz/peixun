import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # 设置无头模式，必须在其他matplotlib导入前设置
import matplotlib.pyplot as plt


# 检查GPU可用性
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# CIFAR-10类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义卷积神经网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 (32x32x3的图像)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 卷积层(16x16x16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 卷积层(8x8x32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout层 (p=0.3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

# 创建模型实例
model = Net()
model.to(device)  # 将模型移动到GPU（如果可用）

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
num_epochs = 20
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    epoch_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss += loss.item()
        
        # 每2000个batch打印一次统计信息
        if i % 2000 == 1999:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(trainloader)}], Loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    
    # 记录平均训练损失
    train_losses.append(epoch_loss / len(trainloader))

    # 每个epoch结束后在测试集上评估
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}] Test Accuracy: {accuracy:.2f}%')

print('Finished Training')

# 在测试集上评估模型并输出每个类别的准确率
model.eval()
all_labels = []
all_preds = []
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

import matplotlib
matplotlib.use('Agg')  # 设置无头模式，必须在其他matplotlib导入前设置
import matplotlib.pyplot as plt

# 1. 保存训练损失和测试准确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('training_metrics.png')  # 保存训练指标图
plt.close()  # 关闭图形释放内存

# 2. 保存测试样本预测结果可视化
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# 获取测试样本并预测
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)
images, labels, preds = images.cpu(), labels.cpu(), preds.cpu()

# 创建并保存预测结果图
plt.figure(figsize=(12, 8))
for idx in range(12):
    plt.subplot(3, 4, idx+1)
    imshow(images[idx])
    plt.title(f'True: {classes[labels[idx]]}\nPred: {classes[preds[idx]]}', fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_predictions.png')  # 保存样本预测图
plt.close()

# 3. 保存每个类别的准确率柱状图
class_acc = [100 * class_correct[i] / class_total[i] for i in range(10)]
plt.figure(figsize=(10, 5))
plt.bar(classes, class_acc)
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Per-class Accuracy')
plt.xticks(rotation=45)
for i, acc in enumerate(class_acc):
    plt.text(i, acc+1, f'{acc:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('class_accuracy.png')  # 保存类别准确率图
plt.close()



# 打印分类报告和每个类别的准确率（保持不变）
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

print("\nPer-class Accuracy:")
for i in range(10):
    print(f'Accuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:.2f}%')
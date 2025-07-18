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
matplotlib.use('Agg')  # 设置无头模式
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

# 定义带残差连接的卷积神经网络
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_channels = 16
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 残差块
        self.layer1 = self.make_layer(16, 2, stride=1)
        self.layer2 = self.make_layer(32, 2, stride=2)
        self.layer3 = self.make_layer(64, 2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64, 500)  # 输入维度改为64
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.3)
    
    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始卷积
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 残差块
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # 全局平均池化到1x1
        out = F.avg_pool2d(out, out.size()[2:])
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接层
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# 创建模型实例
model = ResNet()
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练模型
num_epochs = 20
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss += loss.item()
        
        if i % 2000 == 1999:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(trainloader)}], Loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    
    train_losses.append(epoch_loss / len(trainloader))
    scheduler.step()
    
    # 测试集评估
    model.eval()
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

# 评估模型
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

# 1. 保存训练指标图
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
plt.savefig('resnet_training_metrics.png')
plt.close()

# 2. 保存样本预测图
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)
images, labels, preds = images.cpu(), labels.cpu(), preds.cpu()

plt.figure(figsize=(12, 8))
for idx in range(12):
    plt.subplot(3, 4, idx+1)
    imshow(images[idx])
    plt.title(f'True: {classes[labels[idx]]}\nPred: {classes[preds[idx]]}', fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.savefig('resnet_sample_predictions.png')
plt.close()

# 3. 保存类别准确率图
class_acc = [100 * class_correct[i] / class_total[i] for i in range(10)]
plt.figure(figsize=(10, 5))
plt.bar(classes, class_acc)
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('ResNet Per-class Accuracy')
plt.xticks(rotation=45)
for i, acc in enumerate(class_acc):
    plt.text(i, acc+1, f'{acc:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('resnet_class_accuracy.png')
plt.close()



# 打印报告
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

print("\nPer-class Accuracy:")
for i in range(10):
    print(f'Accuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:.2f}%')
分析下面的代码，帮我该代码中用rearrange展开patch的方式改为用卷积的方式展开，维度不变，其他代码也不用变

import torch

from einops import rearrange, repeat

from einops.layers.torch import Rearrange

from torch import nn

from torchvision import transforms

from torchvision.datasets import CIFAR10

from torch.utils.data import DataLoader

from torch.optim import AdamW

from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):

        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads

        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(

            nn.Linear(inner_dim, dim),

            nn.Dropout(dropout)

        )

    def forward(self, x):

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class FFN(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):

        super().__init__()

        self.net = nn.Sequential(

            nn.LayerNorm(dim),

            nn.Linear(dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, dim),

            nn.Dropout(dropout)

        )

    def forward(self, x):

        return self.net(x)

class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim_ratio, dropout):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([])

        mlp_dim = mlp_dim_ratio * dim

        for _ in range(depth):

            self.layers.append(nn.ModuleList([

                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),

                FFN(dim=dim, hidden_dim=mlp_dim, dropout=dropout)

            ]))

    def forward(self, x):

        for attn, ffn in self.layers:

            x = attn(x) + x

            x = ffn(x) + x

        return self.norm(x)

class ViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, pool='cls', channels=3, dim_head=64, dropout=0.):

        super().__init__()

        image_height, image_width = image_size, image_size

        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            
        nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),

        Rearrange('b c h w -> b (h w) c'),  # 可选：用 einops 或 view

        nn.LayerNorm(dim)
)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim_ratio, dropout)

        self.pool = pool

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):

        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)

        x = self.transformer(x)

        cls_token = x[:, 0]

        pooled_output = cls_token

        classification_result = self.mlp_head(pooled_output)

        return classification_result

# 调整ViT模型定义的image_size和patch_size

model = ViT(

    image_size=32,

    patch_size=4,

    num_classes=10,

    dim=128,  # 减小嵌入维度

    depth=12,   # 减少层数

    heads=8,

    mlp_dim_ratio=2,

    dropout=0.1

)

# 检查可用的 GPU 数量

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# CIFAR-10 数据集加载和预处理

preprocess = transforms.Compose([

    transforms.RandomHorizontalFlip(),

    transforms.RandomCrop(32, padding=4),

    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 添加颜色抖动

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),

])

train_dataset = CIFAR10(root='data/', train=True, transform=preprocess, download=True)

test_dataset = CIFAR10(root='data/', train=False, transform=preprocess, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 增加批次大小

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(model.parameters(), lr=0.0001)  # 使用较小的学习率

scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

def train(dataloader, model, criterion, optimizer, device, epoch):

    model.train()

    total_loss = 0

    with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{20}', unit='batch') as pbar:

        for batch_idx, (images, labels) in enumerate(dataloader):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            current_loss = total_loss / (batch_idx + 1)

            pbar.set_postfix({'loss': current_loss})

            pbar.update(1)

    avg_loss = total_loss / len(dataloader)

    print(f'Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}')

    scheduler.step()
    return avg_loss

def test(dataloader, model, device):

    model.eval()

    total_correct = 0

    total_samples = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total_correct += (predicted == labels).sum().item()

            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    return accuracy


import matplotlib.pyplot as plt

# 用于记录每个 epoch 的 loss 和 accuracy
train_losses = []
test_accuracies = []

# 用于保存最佳模型
best_acc = 0.0
best_model_path = "best_vit_model.pth"

# 开始训练循环并记录指标
for epoch in range(20):
    train_loss = train(train_dataloader, model, criterion, optimizer, device, epoch)
    accuracy = test(test_dataloader, model, device)

    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
    # 保存 loss 和 accuracy
    train_losses.append(train_loss)
    test_accuracies.append(accuracy)

    # 保存最佳模型
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_acc * 100:.2f}%")

# 可视化训练过程
plt.figure(figsize=(12, 5))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot([acc * 100 for acc in test_accuracies], label='Test Accuracy', color='green')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig("training_curve.png")  # 保存图像
plt.close()

# 输出最佳测试集准确率
print(f"\n Best Test Accuracy: {best_acc * 100:.2f}%")
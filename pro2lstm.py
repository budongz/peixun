import torch
from torch.utils.data import DataLoader, Dataset
import os
import re
import numpy as np
import pickle
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
 
data_base_path = 'data/aclImdb'
 
train_batch_size = 64
test_batch_size = 500
max_len = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 # 分词的API
def tokenize(text):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    return [i.strip() for i in text.split()]


# 自定义的数据集
class ImdbDataset(Dataset):
    def __init__(self, mode):
        super(ImdbDataset, self).__init__()
        if mode == "train":
            text_path = [os.path.join(data_base_path, i) for i in ["train/neg", "train/pos"]]
        else:
            text_path = [os.path.join(data_base_path, i) for i in ["test/neg", "test/pos"]]

        self.total_file_path_list = []
        for i in text_path:
            self.total_file_path_list.extend([os.path.join(i, j) for j in os.listdir(i)])

    def __getitem__(self, idx):
        cur_path = self.total_file_path_list[idx]
        cur_filename = os.path.basename(cur_path)
        # 修改这里，使标签为0或1
        label = int(cur_filename.split("_")[-1].split(".")[0]) 
        label = 0 if label <= 4 else 1  # 1-4星为负面(0)，7-10星为正面(1)，忽略5-6星中性评价
        text = tokenize(open(cur_path, encoding="utf-8").read().strip())
        return label, text

    def __len__(self):
        return len(self.total_file_path_list)


# 自定义的collate_fn方法
def collate_fn(batch):
    batch = list(zip(*batch))
    labels = torch.tensor(batch[0], dtype=torch.int32)
    texts = batch[1]
    texts = torch.tensor([ws.transform(i, max_len) for i in texts])
    del batch
    return labels.long(), texts.long()


# 获取数据的方法
def get_dataloader(train=True):
    if train:
        mode = 'train'
    else:
        mode = "test"
    dataset = ImdbDataset(mode)
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# Word2Sequence
class Word2Sequence:
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.fited = False
        self.count = {}

    def to_index(self, word):
        return self.dict.get(word, self.UNK)

    def to_word(self, index):
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_TAG

    def __len__(self):
        return len(self.dict)

    def fit(self, sentence):
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=None, max_count=None, max_feature=None):
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}

        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}

        if max_feature is not None:
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_feature])

        for word in self.count:
            self.dict[word] = len(self.dict)

        self.inversed_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)
        if max_len is not None and len(sentence) > max_len:
            sentence = sentence[:max_len]
        for index, word in enumerate(sentence):
            r[index] = self.to_index(word)
        return np.array(r, dtype=np.int64)

    def inverse_transform(self, indices):
        sentence = []
        for i in indices:
            word = self.to_word(i)
            sentence.append(word)
        return sentence


# 建立词表
def fit_save_word_sequence():
    word_to_sequence = Word2Sequence()
    train_path = [os.path.join(data_base_path, i) for i in ["train/neg", "train/pos"]]
    total_file_path_list = []
    for i in train_path:
        total_file_path_list.extend([os.path.join(i, j) for j in os.listdir(i)])
    for cur_path in tqdm(total_file_path_list, ascii=True, desc="fitting"):
        word_to_sequence.fit(tokenize(open(cur_path, encoding="utf-8").read().strip()))
    word_to_sequence.build_vocab()
    pickle.dump(word_to_sequence, open("model/ws.pkl", "wb"))


ws = pickle.load(open("./model/ws.pkl", "rb"))


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# 手动实现LSTM单元
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入门参数
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # 遗忘门参数
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        # 候选记忆参数
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        
        # 输出门参数
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # 使用Xavier初始化
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
    
    def forward(self, x, hx=None):
        if hx is None:
            h_t = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
            c_t = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_t, c_t = hx
        
        # 输入门
        i_t = torch.sigmoid(F.linear(x, self.W_ii, self.b_i) + F.linear(h_t, self.W_hi))
        
        # 遗忘门
        f_t = torch.sigmoid(F.linear(x, self.W_if, self.b_f) + F.linear(h_t, self.W_hf))
        
        # 候选记忆
        g_t = torch.tanh(F.linear(x, self.W_ig, self.b_g) + F.linear(h_t, self.W_hg))
        
        # 输出门
        o_t = torch.sigmoid(F.linear(x, self.W_io, self.b_o) + F.linear(h_t, self.W_ho))
        
        # 更新细胞状态
        c_t = f_t * c_t + i_t * g_t
        
        # 更新隐藏状态
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

# 手动实现LSTM层
class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(MultiLayerLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 创建多层LSTM单元
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_layers.append(LSTMCell(input_dim, hidden_size))
        
        # 层间dropout
        if dropout > 0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, x, hx=None):
        batch_size, seq_len, _ = x.size()
        
        # 初始化隐藏状态
        if hx is None:
            h_t = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                   for _ in range(self.num_layers)]
            c_t = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                   for _ in range(self.num_layers)]
        else:
            h_t, c_t = hx
        
        # 存储所有时间步的输出
        layer_outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            new_h_t = []
            new_c_t = []
            
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.lstm_layers[layer](
                    x_t, (h_t[layer], c_t[layer]))
                
                # 更新下一层的输入
                x_t = h_t[layer]
                
                # 应用dropout(除了最后一层)
                if self.dropout_layer is not None and layer < self.num_layers - 1:
                    x_t = self.dropout_layer(x_t)
                
                new_h_t.append(h_t[layer])
                new_c_t.append(c_t[layer])
            
            h_t, c_t = new_h_t, new_c_t
            layer_outputs.append(h_t[-1])  # 只保存最后一层的输出
        
        # 将输出堆叠为 [batch_size, seq_len, hidden_size]
        output = torch.stack(layer_outputs, dim=1)
        
        # 返回所有时间步的输出和最后的隐藏状态
        return output, (h_t, c_t)

class ManualLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.3):
        super(ManualLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=ws.PAD)
        self.lstm = MultiLayerLSTM(embedding_dim, hidden_size, num_layers, dropout)
        self.fc = nn.Linear(hidden_size, 2)  # 二分类
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 嵌入层
        embedded = self.dropout(self.embedding(x))  # [batch_size, seq_len, embedding_dim]
        
        # LSTM层
        lstm_out, _ = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_size]
        
        # 取最后一个时间步的输出
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out

# 模型参数
vocab_size = len(ws)
embedding_dim = 128
hidden_size = 128
num_layers = 2
dropout = 0.3

model = ManualLSTMClassifier(vocab_size, embedding_dim, hidden_size, num_layers, dropout).to(device)


# 训练函数
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (labels, texts) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        labels, texts = labels.to(device), texts.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 测试函数
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for labels, texts in tqdm(test_loader, desc="Evaluating"):
            labels, texts = labels.to(device), texts.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 训练准备
train_loader = get_dataloader(train=True)
test_loader = get_dataloader(train=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 训练循环
num_epochs = 10
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    print(f'Epoch {epoch}:')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

# 可视化训练过程
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('model/manual_lstm_training_metrics.png')
plt.show()

print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
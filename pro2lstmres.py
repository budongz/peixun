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
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ManualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.0):
        super(ManualLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # 更稳定的参数初始化
        for layer in range(num_layers):
            for direction in ['forward', 'backward'] if bidirectional else ['forward']:
                for gate in ['i', 'f', 'o', 'c']:
                    # 输入变换权重
                    W = torch.Tensor(input_size if layer == 0 else hidden_size * (2 if bidirectional else 1), 
                                   hidden_size)
                    U = torch.Tensor(hidden_size, hidden_size)
                    
                    # Xavier初始化
                    nn.init.xavier_uniform_(W, gain=nn.init.calculate_gain('sigmoid'))
                    nn.init.xavier_uniform_(U, gain=nn.init.calculate_gain('sigmoid'))
                    
                    # 遗忘门偏置初始化为1
                    if gate == 'f':
                        b = torch.ones(hidden_size)
                    else:
                        b = torch.zeros(hidden_size)
                    
                    self.register_parameter(f'{direction}_l{layer}_W_{gate}', nn.Parameter(W))
                    self.register_parameter(f'{direction}_l{layer}_U_{gate}', nn.Parameter(U))
                    self.register_parameter(f'{direction}_l{layer}_b_{gate}', nn.Parameter(b))
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, hidden_states=None):
        seq_len, batch_size, _ = x.size()
        
        if hidden_states is None:
            h = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                          batch_size, self.hidden_size, device=x.device)
            c = torch.zeros_like(h)
        else:
            h, c = hidden_states
        
        layer_outputs = []
        
        for layer in range(self.num_layers):
            layer_input = x if layer == 0 else layer_outputs[layer-1]
            
            if self.dropout > 0 and layer > 0:
                layer_input = self.dropout_layer(layer_input)
            
            if self.bidirectional:
                # 前向
                h_forward = h[layer*2]
                c_forward = c[layer*2]
                forward_output = self._lstm_step(layer_input, h_forward, c_forward, layer, 'forward')
                
                # 后向
                h_backward = h[layer*2+1]
                c_backward = c[layer*2+1]
                backward_output = self._lstm_step(layer_input.flip(0), h_backward, c_backward, layer, 'backward')
                backward_output = backward_output.flip(0)
                
                layer_output = torch.cat([forward_output, backward_output], dim=-1)
                h_layer = torch.stack([forward_output[-1], backward_output[-1]])
                c_layer = torch.stack([c_forward, c_backward])
            else:
                h_layer = h[layer]
                c_layer = c[layer]
                layer_output = self._lstm_step(layer_input, h_layer, c_layer, layer, 'forward')
                h_layer = layer_output[-1].unsqueeze(0)
                c_layer = c_layer.unsqueeze(0)
            
            # 添加残差连接 (从当前层的输入到输出)
            if layer > 0:  # 从第二层开始添加残差连接
                # 确保维度匹配
                if layer_output.size(-1) != layer_input.size(-1):
                    # 如果维度不匹配，使用线性投影
                    residual = nn.Linear(layer_input.size(-1), layer_output.size(-1), device=x.device)(layer_input)
                else:
                    residual = layer_input
                layer_output = layer_output + residual
            
            layer_outputs.append(layer_output)
            
            # 更新隐藏状态
            if self.bidirectional:
                h = torch.cat([h[:layer*2], h_layer, h[layer*2+2:]])
                c = torch.cat([c[:layer*2], c_layer, c[layer*2+2:]])
            else:
                h = torch.cat([h[:layer], h_layer, h[layer+1:]])
                c = torch.cat([c[:layer], c_layer, c[layer+1:]])
        
        return layer_outputs[-1], (h, c)
    
    def _lstm_step(self, x, h_prev, c_prev, layer, direction):
        seq_len, batch_size, _ = x.size()
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]
            
            # 获取参数
            W_i = getattr(self, f'{direction}_l{layer}_W_i')
            W_f = getattr(self, f'{direction}_l{layer}_W_f')
            W_o = getattr(self, f'{direction}_l{layer}_W_o')
            W_c = getattr(self, f'{direction}_l{layer}_W_c')
            
            U_i = getattr(self, f'{direction}_l{layer}_U_i')
            U_f = getattr(self, f'{direction}_l{layer}_U_f')
            U_o = getattr(self, f'{direction}_l{layer}_U_o')
            U_c = getattr(self, f'{direction}_l{layer}_U_c')
            
            b_i = getattr(self, f'{direction}_l{layer}_b_i')
            b_f = getattr(self, f'{direction}_l{layer}_b_f')
            b_o = getattr(self, f'{direction}_l{layer}_b_o')
            b_c = getattr(self, f'{direction}_l{layer}_b_c')
            
            # 更稳定的计算方式
            i_t = torch.sigmoid((x_t @ W_i) + (h_prev @ U_i) + b_i)
            f_t = torch.sigmoid((x_t @ W_f) + (h_prev @ U_f) + b_f)
            o_t = torch.sigmoid((x_t @ W_o) + (h_prev @ U_o) + b_o)
            c_tilde = torch.tanh((x_t @ W_c) + (h_prev @ U_c) + b_c)
            
            c_t = f_t * c_prev + i_t * c_tilde
            h_t = o_t * torch.tanh(c_t)
            
            outputs.append(h_t)
            h_prev = h_t
            c_prev = c_t
        
        return torch.stack(outputs)

class IMDBModel(nn.Module):
    def __init__(self):
        super(IMDBModel, self).__init__()
        self.hidden_size = 64
        self.embedding_dim = 200
        self.num_layer = 2
        self.bidirectional = True
        self.bi_num = 2 if self.bidirectional else 1
        self.dropout = 0.5
        
        # 限制embedding范围
        self.embedding = nn.Embedding(len(ws), self.embedding_dim, padding_idx=ws.PAD)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        self.lstm = ManualLSTM(self.embedding_dim, self.hidden_size,
                             num_layers=self.num_layer, 
                             bidirectional=self.bidirectional,
                             dropout=self.dropout)
        
        # 更稳定的输出层
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * self.bi_num, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        h_0, c_0 = self.init_hidden_state(x.size(1))
        _, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        if self.bidirectional:
            out = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        else:
            out = h_n[-1, :, :]
            
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

    def init_hidden_state(self, batch_size):
        h_0 = torch.zeros(self.num_layer * self.bi_num, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layer * self.bi_num, batch_size, self.hidden_size).to(device)
        return h_0, c_0

# 初始化模型和优化器
imdb_model = IMDBModel().to(device)
optimizer = optim.Adam(imdb_model.parameters(), lr=0.0005, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    imdb_model.train()
    train_dataloader = get_dataloader(train=True)
    
    for idx, (target, input) in enumerate(train_dataloader):
        target = target.to(device)
        input = input.to(device)
        
        optimizer.zero_grad()
        output = imdb_model(input)
        loss = F.nll_loss(output, target)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(imdb_model.parameters(), max_norm=0.5)
        
        loss.backward()
        optimizer.step()
        
        if idx % 10 == 0:
            print(f'Epoch {epoch} Batch {idx} Loss: {loss.item():.6f}')
            
            # 检查NaN
            for name, param in imdb_model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN in {name}")
                    break

def test():
    imdb_model.eval()
    test_loss = 0
    correct = 0
    test_dataloader = get_dataloader(train=False)
    
    with torch.no_grad():
        for target, input in test_dataloader:
            target = target.to(device)
            input = input.to(device)
            output = imdb_model(input)
            test_loss += F.nll_loss(output, target, reduction='sum')
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum()
    
    test_loss /= len(test_dataloader.dataset)
    accuracy = 100. * correct / len(test_dataloader.dataset)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} ({accuracy:.2f}%)\n')

if __name__ == '__main__':
    # 训练和测试
    test()
    for i in range(3):
        train(i)
        print(f"训练第{i+1}轮的测试结果-----------------------------------------------------------------------------------------")
        test()
import torch
import torch.nn as nn
import numpy as np

# CNN 模型定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 20 * 20, 128)  # 假设 env_size = 20
        self.fc2 = nn.Linear(128, 4)  # 输出4个动作

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 数据预处理，将状态和动作转换为适合 CNN 的输入格式
def preprocess_data(states, actions, action_map={"W": 0, "S": 1, "A": 2, "D": 3}):
    # 扩展状态的维度为 (num_samples, 1, env_size, env_size)
    states = np.expand_dims(states, axis=1)
    
    #TODO why ?
    # 将动作转换为数字标签
    actions = np.array([action_map[a] for a in actions])

    # 转换为张量
    return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.long)

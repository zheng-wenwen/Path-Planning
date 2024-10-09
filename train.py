import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import CNN, preprocess_data
from Astar import astar, get_action
from env import env, generate_random_environment, update_view, move_agent, env_size, directions
from path_planning_cnn_master.model.ppcnet import PPCNet

# 生成相对终点位置
def get_relative_goal_positions(states, goals):
    return [[(g[0] - s[0]) / env_size, (g[1] - s[1]) / env_size] for s, g in zip(states, goals)]

# 生成专家数据，包括相对终点位置
def generate_expert_data(num_samples=100):
    """
        Args:
            num_samples

        Returns:
            states:存储path坐标,len = num_samples
            goals: 目标点坐标,len = num_samples
            starts:起点坐标,len = num_samples
            envs:环境,障碍物2, 非障碍物0, size=20*20, len = num_samples
        """
    states = []
    actions = []
    goals = []
    starts= []
    envs = []
    

    for _ in range(num_samples):
        #TODO 1.put outside?
        #TODO 2. generate_random_environment 可以随机生成起点和goal，并且能绕过障碍物，下面随机生成起始点和目标点可能会撞倒障碍物
        #TODO 3. astar只能生成路径
        # 随机生成环境 
        environment , start_pos, goal_pos = generate_random_environment()
        

        # 随机生成起始点和目标点
        #start_pos = [np.random.randint(0, env_size), np.random.randint(0, env_size)]
        #goal_pos = [np.random.randint(0, env_size), np.random.randint(0, env_size)]

        # 使用 A* 算法生成从起点到目标点的路径
        path  = astar(environment, list(start_pos), list(goal_pos))

        if path:  # 确保有路径生成
            # 记录状态、动作和目标位置
            states.append(path)
            #actions.extend(path_actions)
            #goals.extend(get_relative_goal_positions(path, goal_pos))  # 记录每一步的相对目标位置
            goals.append(goal_pos)
            starts.append(start_pos)
            envs.append(environment)

    return states, goals, starts ,envs

# 训练模型并记录损失
def train_model(model, train_loader, epochs=10, lr=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels, goals in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, goals)  # 将状态和相对终点位置传递给模型
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'第 {epoch + 1} 轮，损失: {avg_loss:.3f}')

    return loss_history

# 绘制损失曲线
def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history, label='训练损失')
    plt.title('训练损失随时间变化')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 生成专家数据
    states, goals, starts , envs = generate_expert_data()
    import ipdb
    ipdb.set_trace()

    # 生成相对目标位置
    goal_positions = get_relative_goal_positions(states, goals)
    
    #TODO there is no goal_positions in preprocess_data
    # 预处理数据，加入相对目标位置
    states_tensor, actions_tensor, goal_tensor = preprocess_data(states, actions, goal_positions)

    # 创建数据集和数据加载器
    #TODO 1.这里很混乱，你原先的是input=state(path),label = action，goal。我改后建议为input= env,label = state(path) ,goal,start
    #TODO 2.修改数据集表示，由坐标拓展为图像。
    #TODO 3.适配ppcnet
    
    dataset = TensorDataset(states_tensor, actions_tensor, goal_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化CNN模型
    #model = CNN()

    #init PPCnet
    model = PPCNet(gaussian_blur_kernel=3).to(device)

    # 训练模型
    loss_history = train_model(model, train_loader, epochs=10)

    # 保存模型
    model_save_path = 'trained_model_with_goal.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'模型已保存至 {model_save_path}')

    # 绘制损失曲线
    plot_loss(loss_history)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import CNN, preprocess_data
from Astar import astar, get_action
from env import env, generate_random_environment, update_view, move_agent, env_size, directions
from path_planning_cnn_master.model.ppcnet import PPCNet
from dataset import Dataset
from torch.nn.modules.loss import MSELoss, L1Loss
import matplotlib.pyplot as plt 
import cv2
from path_planning_cnn_master.lr_scheduler.scheduler import CosineAnnealingWarmRestartsDecay
from torch.optim import Adam
from focalloss import FocalLoss
from l1loss import l1_loss

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
def train_model(model, train_loader,loss_f, epochs=10, lr=0.001):
    
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200)
    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=150, decay=0.995)
    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, goals, starts, labels in train_loader:
            
            
            optimizer.zero_grad()
            #outputs = model(inputs, goals)  # 将状态和相对终点位置传递给模型
            #loss = criterion(outputs, labels)
            out = model(inputs, starts, goals)
            
            loss = loss_f(out.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'第 {epoch + 1} 轮，损失: {avg_loss:.5f}')

    return loss_history



  
def visualize_and_save(data, goals, starts, filename, cmap=None):
    # 类别到颜色的映射  
    COLOR_MAP = {  
    0: [0, 0, 0],  # 背景 - 黑色  
    1: [0, 255, 0],  # 路径 - 绿色  
    2: [255, 0, 0]  # 障碍物 - 红色  
    }    
    # 将数据转换为彩色图像  
    height, width = data.shape  
    image = np.zeros((height, width, 3), dtype=np.uint8)  
      
    for y in range(height):  
        for x in range(width):  
            category = data[y, x]
            if category>=0 and category<0.5:
                image[y, x]= [0,0,0]
            elif category>=0.5 and category<=1:
                image[y,x] = [0,255,0]
            elif category ==2.0:
                image[y,x]= [255,0,0]

            """if category in COLOR_MAP:  
                image[y, x] = COLOR_MAP[category]"""  
      
    # 标记 goals 和 starts  
    
    cv2.circle(image, (int(goals[1]), int(goals[0])), 5, [0, 0, 255], -1)  # 蓝色圈表示 goals  
    
    cv2.circle(image, (int(starts[1]), int(starts[0])), 5, [255, 0, 255], -1)  # 洋红色圈表示 starts  
      
    # 保存图像  
    plt.imsave(filename, image)  
    plt.close()  

def test_model(model, data_loader):
    save_path = './result/'
    model.eval()
    idx = 0
    for inputs, goals, starts, labels in data_loader:
        out = model(inputs, starts, goals)
  

        out = out.squeeze(0).squeeze(0)  # 去除 batch 维度 
        input_data =  inputs.squeeze(0).squeeze(0)
        label_data = labels.squeeze(0)

        goal_points = goals.squeeze(0)
        start_points = starts.squeeze(0)
          
        # 可视化并保存 input, label 和 output  
        """import ipdb
        ipdb.set_trace()"""
        visualize_and_save(input_data, goal_points, start_points, save_path + str(idx) + "_input.png")  
        visualize_and_save(label_data, goal_points, start_points, save_path + str(idx) + "_label.png")  
        visualize_and_save(out.detach().cpu().numpy(), goal_points, start_points, save_path + str(idx) + "_output.png")  
        idx = idx+1
        


    return 0


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



def train():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 生成专家数据
    states, goals, starts , envs = generate_expert_data(num_samples=500)
    
    

    # 生成相对目标位置
    #goal_positions = get_relative_goal_positions(states, goals)
    
    #TODO there is no goal_positions in preprocess_data
    # 预处理数据，加入相对目标位置
    #states_tensor, actions_tensor, goal_tensor = preprocess_data(states, actions, goal_positions)

    # 创建数据集和数据加载器
    #TODO 1.这里很混乱，你原先的是input=state(path),label = action，goal。我改后建议为input= env,label = state(path) ,goal,start
    #TODO 2.修改数据集表示，由坐标拓展为图像。
    #TODO 3.适配ppcnet
    
    dataset = Dataset(states, goals, starts, envs)
    train_loader = DataLoader(dataset,batch_size=8,shuffle=True)
    #dataset = TensorDataset(states_tensor, actions_tensor, goal_tensor)
    #train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化CNN模型
    #model = CNN()

    #init PPCnet
    model = PPCNet(gaussian_blur_kernel=3).to(device)
    #loss_f = L1Loss().to(device)
    #loss_f = FocalLoss(alpha=1, gamma=2, reduction='mean').to(device)  
    loss_f = l1_loss 
    # 训练模型
    loss_history = train_model(model, train_loader,loss_f, epochs=500, lr=1e-5)

    # 保存模型
    model_save_path = 'trained_model_with_goal.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'模型已保存至 {model_save_path}')
    
    # 绘制损失曲线
    #plot_loss(loss_history)

    #test
    #test_loader = DataLoader(dataset,batch_size=1,shuffle=True)
    #test_model(model, test_loader)


def test():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 生成专家数据
    states, goals, starts , envs = generate_expert_data(num_samples=10)
    
    dataset = Dataset(states, goals, starts, envs)
    data_loader = DataLoader(dataset,batch_size=1,shuffle=True)

    #init PPCnet
    model = PPCNet(gaussian_blur_kernel=3).to(device)
    model.load_state_dict(torch.load('trained_model_with_goal.pth'))  
    # test模型
    test_model(model, data_loader)






if __name__ == '__main__':
    train()
    test()

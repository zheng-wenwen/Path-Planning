#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# In[7]:


# 环境配置
env_size = 20
agent_view = 9

# 初始化环境：0 代表可行区域，2 代表障碍物，1 代表终点
env = np.zeros((env_size, env_size))
new_env = np.full(env.shape, -1)  # 用来显示每步的图像
goal_position = (18, 18)
env[goal_position] = 1

# 随机放置一些障碍物
np.random.seed(42)
num_obstacles = 30
obstacles = np.random.choice(env_size * env_size, num_obstacles, replace=False)
for ob in obstacles:
    env[ob // env_size, ob % env_size] = 2

# 初始化 agent 位置
agent_position = [5, 5]

# 定义方向：上、下、左、右
directions = {
    "W": (1, 0),  # 上
    "S": (-1, 0),   # 下
    "A": (0, -1),  # 左
    "D": (0, 1)    # 右
}
def generate_random_environment(obstacle_ratio=0.2):
    global env, agent_position, goal_position
    # 重置环境
    env = np.zeros((env_size, env_size), dtype=int)

    # 随机生成障碍物
    num_obstacles = int(env_size * env_size * obstacle_ratio)
    obstacle_positions = set()
    while len(obstacle_positions) < num_obstacles:
        pos = (np.random.randint(0, env_size), np.random.randint(0, env_size))
        obstacle_positions.add(pos)

    for pos in obstacle_positions:
        env[pos] = 2

    # 随机生成 agent 位置，确保不在障碍物上
    while True:
        agent_position = (np.random.randint(0, env_size), np.random.randint(0, env_size))
        if env[agent_position] == 0:
            env[agent_position] = 0
            break

    # 随机生成目标位置，确保不在障碍物或 agent 上
    while True:
        goal_position = (np.random.randint(0, env_size), np.random.randint(0, env_size))
        if env[goal_position] == 0 and goal_position != agent_position:
            env[goal_position] = 1
            break

    return env, agent_position, goal_position
# 更新 agent 可视范围内的不可见区域 (-1 代表不可见)，并考虑障碍物阻挡视野
def update_view(env, agent_pos):
    view_range = agent_view // 2
    x, y = agent_pos
    visible_env = np.full_like(env, -1)  # 初始化为不可见区域
    
    # 视野范围内扫描
    for i in range(max(0, x - view_range), min(env_size, x + view_range + 1)):
        for j in range(max(0, y - view_range), min(env_size, y + view_range + 1)):
            if i == x and j == y:
                visible_env[i, j] = env[i, j]  # agent 的当前位置总是可见
            else:
                # 判断是否有障碍物阻挡
                if is_visible(env, agent_pos, (i, j)):
                    visible_env[i, j] = env[i, j]  # 如果没有障碍物阻挡则可见
                else:
                    visible_env[i, j] = -1  # 如果有障碍物，则后面的格子不可见
    return visible_env

# 判断目标格子是否可见（在所有方向上检查障碍物）
def is_visible(env, agent_pos, target_pos):
    # 使用 Bresenham 算法来计算两点之间的直线
    def bresenham(x0, y0, x1, y1):
        """生成从 (x0, y0) 到 (x1, y1) 的直线路径"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    # 获取 agent 和目标位置的坐标
    ax, ay = agent_pos
    tx, ty = target_pos

    # 生成从 agent 到目标格子的直线路径
    path = bresenham(ax, ay, tx, ty)

    # 检查路径中是否有障碍物
    for x, y in path:
        if env[x, y] == 2:  # 如果有障碍物
            return False
    return True

# 可视化环境，添加颜色区分和边界
def visualize_env(visible_env, agent_pos):
    fig, ax = plt.subplots()
    # 显示完整地图的障碍物
    for i in range(env_size):
        for j in range(env_size):
            if env[i, j] == 2:  # 所有障碍物始终用红色表示
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='red'))  # 障碍物

    # 绘制 agent 的可视范围边框
    view_range = agent_view // 2
    for i in range(max(0, agent_pos[0] - view_range), min(env_size, agent_pos[0] + view_range + 1)):
        for j in range(max(0, agent_pos[1] - view_range), min(env_size, agent_pos[1] + view_range + 1)):
            if visible_env[i, j] == 0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='yellow'))  # 可见区域
            elif visible_env[i, j] == -1:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='gray'))    # 不可见区域
            elif visible_env[i, j] == 1:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='green'))   # 终点位置
    # 显示完整地图的障碍物
    for i in range(env_size):
        for j in range(env_size):
            if env[i, j] == 2:  # 障碍物始终用红色表示
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='red'))  # 障碍物
            elif visible_env[i, j] == 1:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='green'))   # 终点位置
    # 绘制网格线
    ax.set_xticks(np.arange(0, env_size, 1))
    ax.set_yticks(np.arange(0, env_size, 1))
    ax.grid(color='black', linestyle='-', linewidth=1)  # 边界线变细
    
    # 仅显示整数的坐标
    ax.set_xticklabels(np.arange(0, env_size, 1))
    ax.set_yticklabels(np.arange(0, env_size, 1))
    
    # 确保每个格子是正方形
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 绘制 agent 位置，使用实心蓝色圆点表示 agent 当前位置
    ax.plot(agent_pos[1] + 0.5, agent_pos[0] + 0.5, 'o', color='blue', markersize=12)  # 蓝色实心圆点表示 agent
    
    # 绘制 agent 的可视范围边框
    #rect = plt.Rectangle((agent_pos[1] - view_range, agent_pos[0] - view_range), agent_view, agent_view,
                         #fill=False, edgecolor='blue', linewidth=2)  # 蓝色边框表示视野范围
    #ax.add_patch(rect)
    
    plt.xlim(0, env_size)
    plt.ylim(0, env_size)
    
    plt.show()

# 移动 agent
def move_agent(env, agent_pos, direction):
    new_x = agent_pos[0] + directions[direction][0]
    new_y = agent_pos[1] + directions[direction][1]
    
    # 检查是否超出边界或遇到障碍物
    if 0 <= new_x < env_size and 0 <= new_y < env_size and env[new_x, new_y] != 2:
        agent_pos[0], agent_pos[1] = new_x, new_y
    else:
        print("撞到了边界或障碍物！")
    
    return agent_pos


# In[8]:


# 主循环
if __name__ == '__main__':
    env, agent_position, goal_position=generate_random_environment(obstacle_ratio=0.3)
    visible_env = update_view(env, agent_position)
    visualize_env(visible_env, agent_position)
"""
while True:
    move = input("输入方向 (W 上, S 下, A 左, D 右)，输入 'exit' 退出: ").upper()
    if move == 'EXIT':
        break
    if move in directions:
        agent_position = move_agent(env, agent_position, move)
        visible_env = update_view(env, agent_position)
        visualize_env(visible_env, agent_position)
    else:
        print("无效输入，请输入有效方向。")
"""

# In[ ]:





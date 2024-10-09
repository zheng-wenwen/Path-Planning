import torch
import numpy as np
from model import CNN, preprocess_data
from 强化学习.env import env, update_view, move_agent, env_size, directions

# 确保终点不在障碍物上
def get_valid_goal(env):
    while True:
        goal_pos = tuple(np.random.randint(0, env_size, size=2))
        if env[goal_pos] != 2:
            return goal_pos

# 测试模型在新环境中的表现
def test_model(model, test_env):
    model.eval()

    start_pos = [np.random.randint(0, env_size), np.random.randint(0, env_size)]
    goal_pos = get_valid_goal(test_env)

    agent_pos = start_pos
    steps = []

    while tuple(agent_pos) != goal_pos:
        visible_env = update_view(test_env, agent_pos)
        visible_env = torch.tensor(visible_env, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 适配 CNN 输入
        output = model(visible_env)
        action_index = torch.argmax(output, dim=1).item()

        action_map_reverse = {0: "W", 1: "S", 2: "A", 3: "D"}
        action = action_map_reverse[action_index]
        
        agent_pos = move_agent(test_env, agent_pos, action)
        steps.append((agent_pos, action))
        print(f"Agent moved to {agent_pos} with action {action}")

    print("Agent reached the goal!")
    return steps

if __name__ == '__main__':
    # 加载模型
    model = CNN()
    model.load_state_dict(torch.load('trained_model.pth'))
    print('Model loaded from trained_model.pth')

    # 测试模型
    test_env = np.copy(env)  # 使用新的环境进行测试
    steps = test_model(model, test_env)
    
    # 可视化 agent 的路径
    import matplotlib.pyplot as plt
    positions, actions = zip(*steps)
    x_coords = [pos[1] for pos in positions]
    y_coords = [pos[0] for pos in positions]

    plt.plot(x_coords, y_coords, marker='o')
    plt.title('Agent Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.scatter(x_coords[0], y_coords[0], c='red', label='Start', marker='o')
    plt.scatter(x_coords[-1], y_coords[-1], c='green', label='Goal', marker='o')
    plt.legend()
    plt.show()

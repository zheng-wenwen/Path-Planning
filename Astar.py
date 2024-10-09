import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from queue import PriorityQueue
from env import env, agent_position, goal_position, update_view, visualize_env, move_agent, env_size, directions

# 定义动作
actions = {
    "W": (1, 0),   # 上
    "S": (-1, 0),  # 下
    "A": (0, -1),  # 左
    "D": (0, 1)    # 右
}

# A* 算法找到最优路径
def astar(env, start, goal):
    start = tuple(start)
    goal = tuple(goal)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {tuple(start): 0}


    f_score = {tuple(start): heuristic(start, goal)}

    while not open_set.empty():
        _, current = open_set.get()
        

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            
            return path[::-1]

        for direction in directions.values():
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if 0 <= neighbor[0] < env_size and 0 <= neighbor[1] < env_size and env[neighbor] != 2:
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))
    
    return []

# 计算相对目标位置
def get_relative_goal_pos(agent_pos, goal_pos):
    return [(goal_pos[0] - agent_pos[0]) / env_size, (goal_pos[1] - agent_pos[1]) / env_size]

# 计算两个位置之间的动作
def get_action(current_pos, next_pos):
    delta_x = next_pos[0] - current_pos[0]
    delta_y = next_pos[1] - current_pos[1]

    for action, (dx, dy) in actions.items():
        if dx == delta_x and dy == delta_y:
            return action
    return None

# 动画显示 agent 的移动过程，保存每一步的图像、动作以及相对目标位置到 replay_buffer
def animate_agent_movement(env, agent_position, path, goal_position, replay_buffer):
    fig, ax = plt.subplots()

    def update(num):
        ax.clear()  # 清除上一步的图像
        agent_pos = path[num]
        visible_env = update_view(env, agent_pos)  # 更新 agent 的可视范围

        # 使用从 env.py 导入的 visualize_env 函数来绘制环境
        visualize_env(visible_env, agent_pos)

        if num < len(path) - 1:
            next_pos = path[num + 1]
            action = get_action(agent_pos, next_pos)  # 获取当前动作
            relative_goal_pos = get_relative_goal_pos(agent_pos, goal_position)  # 计算相对目标位置

            # 保存可视环境、动作和相对目标位置
            replay_buffer.append((visible_env.copy(), action, relative_goal_pos))

    ani = animation.FuncAnimation(fig, update, frames=len(path), repeat=False)
    plt.show()



if __name__ == '__main__':
    # 主程序逻辑
    replay_buffer = []  # 存储 (每一步的环境, 当前动作, 相对目标位置) 元组
    path = astar(env, tuple(agent_position), goal_position)

    if path:
        print("找到路径: ", path)
        animate_agent_movement(env, agent_position, path, goal_position, replay_buffer)
    else:
        print("未找到路径！")

    # 输出 replay_buffer 中的动作、相对目标位置和对应的环境信息
    for step, (env_snapshot, action, relative_goal_pos) in enumerate(replay_buffer):
        print(f"Step {step + 1}: Action = {action}, Relative Goal Position = {relative_goal_pos}")

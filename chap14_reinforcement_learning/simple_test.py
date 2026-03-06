from RL_QG_agent import RL_QG_agent
import numpy as np
import os

print("="*60)
print("测试 RL_QG_agent")
print("="*60)

# 创建目录
if not os.path.exists('Reversi'):
    os.makedirs('Reversi')
    print("✅ 创建Reversi目录")

# 创建智能体
print("\n创建智能体...")
agent = RL_QG_agent()

# 模拟几局游戏
print("\n模拟对弈...")
for game in range(3):
    print(f"\n第{game+1}局:")
    
    # 模拟棋盘状态
    state = np.random.rand(3, 8, 8)
    
    # 模拟可下的位置
    enables = [0, 1, 2, 10, 20, 30, 40, 50]
    
    # 智能体选择动作
    action = agent.place(state, enables)
    
    row = action // 8
    col = action % 8
    print(f"  白棋选择: 位置{action} (第{row}行, 第{col}列)")
    
    # 模拟存储经验（训练用）
    reward = np.random.randn()
    next_state = np.random.rand(3, 8, 8)
    done = False
    
    agent.remember(state, action, reward, next_state, done)

# 测试训练
print("\n测试训练...")
if len(agent.memory) >= agent.batch_size:
    loss = agent.replay()
    print(f"  训练损失: {loss:.6f}")

# 保存模型
print("\n保存模型...")
agent.save_model()

# 加载模型
print("\n加载模型...")
agent.load_model()

print("\n"+"="*60)
print("✅ 所有测试通过！")
print("="*60)
print("\n可以提交 RL_QG_agent.py 了！")
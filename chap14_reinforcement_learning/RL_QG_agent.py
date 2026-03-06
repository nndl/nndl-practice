"""
强化学习黑白棋智能体 - DQN实现
使用深度Q网络学习玩黑白棋
"""

# RL_QG_agent.py
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import random
from collections import deque


class RL_QG_agent:
    """深度Q学习黑白棋智能体"""
    
    def __init__(self):
        """初始化智能体"""
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        
        # 创建模型目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # 超参数
        self.board_size = 8
        self.action_space = self.board_size * self.board_size  # 64个位置
        self.state_shape = (3, 8, 8)  # 3通道: 黑棋、白棋、可下位置
        
        # DQN参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Experience Replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # 初始化模型
        self.init_model()
    
    def init_model(self):
        """定义神经网络"""
        # TensorFlow 1.x 兼容
        tf.reset_default_graph()
        
        # Placeholders
        self.state_input = tf.placeholder(tf.float32, [None, 3, 8, 8], name='state_input')
        self.action_input = tf.placeholder(tf.int32, [None], name='action_input')
        self.target_q = tf.placeholder(tf.float32, [None], name='target_q')
        
        # 构建Q网络
        with tf.variable_scope('q_network'):
            # 将 (batch, 3, 8, 8) 转换为 (batch, 8, 8, 3)
            state_transposed = tf.transpose(self.state_input, [0, 2, 3, 1])
            
            # 卷积层1
            conv1 = tf.layers.conv2d(
                state_transposed,
                filters=64,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu,
                name='conv1'
            )
            
            # 卷积层2
            conv2 = tf.layers.conv2d(
                conv1,
                filters=128,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu,
                name='conv2'
            )
            
            # 卷积层3
            conv3 = tf.layers.conv2d(
                conv2,
                filters=128,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu,
                name='conv3'
            )
            
            # 展平
            flatten = tf.layers.flatten(conv3)
            
            # 全连接层
            fc1 = tf.layers.dense(
                flatten,
                units=256,
                activation=tf.nn.relu,
                name='fc1'
            )
            
            # 输出层 - Q值（64个动作）
            self.q_values = tf.layers.dense(
                fc1,
                units=self.action_space,
                activation=None,
                name='q_output'
            )
        
        # 损失函数
        # 选择执行动作对应的Q值
        action_one_hot = tf.one_hot(self.action_input, self.action_space)
        q_value_pred = tf.reduce_sum(self.q_values * action_one_hot, axis=1)
        
        # 均方误差损失
        self.loss = tf.reduce_mean(tf.square(self.target_q - q_value_pred))
        
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        # 创建Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        # 初始化变量
        self.sess.run(tf.global_variables_initializer())
        
        # Saver
        self.saver = tf.train.Saver(max_to_keep=5)
        
        print("DQN模型初始化完成")
        print(f"  状态形状: {self.state_shape}")
        print(f"  动作空间: {self.action_space}")
    
    def preprocess_state(self, state):
        """
        预处理状态
        state: (3, 8, 8) numpy数组
        """
        # 确保状态是float32类型
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        else:
            state = state.astype(np.float32)
        
        # 归一化
        state = state / np.max(state + 1e-8)
        
        return state
    
    def place(self, state, enables):
        """
        选择下棋位置
        state: 当前棋盘状态 (3, 8, 8)
        enables: 可行动作列表
        返回: action (0-63)
        """
        # 如果没有可行动作，返回pass
        if len(enables) == 0:
            return self.board_size ** 2 + 1
        
        # ε-greedy策略
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            action = random.choice(enables)
        else:
            # 利用：选择Q值最大的动作
            state = self.preprocess_state(state)
            state_batch = np.expand_dims(state, axis=0)  # (1, 3, 8, 8)
            
            q_values = self.sess.run(self.q_values, feed_dict={
                self.state_input: state_batch
            })[0]  # (64,)
            
            # 只考虑可行动作
            valid_q_values = [(enables[i], q_values[enables[i]]) for i in range(len(enables))]
            valid_q_values.sort(key=lambda x: x[1], reverse=True)
            
            action = valid_q_values[0][0]
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = []
        actions = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            state = self.preprocess_state(state)
            states.append(state)
            actions.append(action)
            
            if done:
                target = reward
            else:
                next_state = self.preprocess_state(next_state)
                next_state_batch = np.expand_dims(next_state, axis=0)
                
                # 计算下一状态的最大Q值
                next_q_values = self.sess.run(self.q_values, feed_dict={
                    self.state_input: next_state_batch
                })[0]
                
                target = reward + self.gamma * np.max(next_q_values)
            
            targets.append(target)
        
        states = np.array(states)
        actions = np.array(actions)
        targets = np.array(targets)
        
        # 训练
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.state_input: states,
            self.action_input: actions,
            self.target_q: targets
        })
        
        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def save_model(self):
        """保存模型"""
        save_path = self.saver.save(
            self.sess, 
            os.path.join(self.model_dir, 'parameter.ckpt')
        )
        print(f"模型已保存: {save_path}")
    
    def load_model(self):
        """加载模型"""
        try:
            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            if checkpoint:
                self.saver.restore(self.sess, checkpoint)
                print(f"模型已加载: {checkpoint}")
                # 加载后设置较小的epsilon（更多利用）
                self.epsilon = 0.1
            else:
                print("未找到已保存的模型，使用随机初始化")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("使用随机初始化的模型")
    
    def get_reward(self, env, color):
        """
        计算奖励
        color: 0=黑棋, 1=白棋
        """
        # 统计棋盘上的棋子数
        black_count = np.sum(env.state[0, :, :])
        white_count = np.sum(env.state[1, :, :])
        
        if color == 1:  # 白棋
            return white_count - black_count
        else:  # 黑棋
            return black_count - white_count
    
    def train_self_play(self, episodes=1000):
        """
        自我对弈训练
        """
        import gym
        
        env = gym.make('Reversi8x8-v0')
        
        print("=" * 60)
        print("开始自我对弈训练...")
        print(f"训练轮数: {episodes}")
        print("=" * 60)
        
        win_count = 0
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(100):
                # 白棋走棋（我们的智能体）
                enables = env.possible_actions
                
                if len(enables) == 0:
                    action_pos = self.board_size ** 2 + 1
                else:
                    action_pos = self.place(state, enables)
                
                action = [action_pos, 1]  # 白棋
                next_state, reward, done, info = env.step(action)
                
                # 存储经验
                self.remember(state, action_pos, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    # 计算最终得分
                    black_score = len(np.where(env.state[0, :, :] == 1)[0])
                    white_score = len(np.where(env.state[1, :, :] == 1)[0])
                    
                    if white_score > black_score:
                        win_count += 1
                    
                    if (episode + 1) % 10 == 0:
                        win_rate = win_count / 10
                        print(f"Episode {episode+1}/{episodes}, "
                              f"ε={self.epsilon:.3f}, "
                              f"胜率={win_rate:.2f}, "
                              f"白棋={white_score}, 黑棋={black_score}")
                        win_count = 0
                    
                    break
                
                # 黑棋走棋（随机对手）
                enables = env.possible_actions
                if len(enables) == 0:
                    action_pos = self.board_size ** 2 + 1
                else:
                    action_pos = random.choice(enables)
                
                action = [action_pos, 0]  # 黑棋
                state, _, done, _ = env.step(action)
                
                if done:
                    break
            
            # 经验回放
            if len(self.memory) >= self.batch_size:
                self.replay()
            
            # 定期保存模型
            if (episode + 1) % 100 == 0:
                self.save_model()
        
        print("=" * 60)
        print("训练完成！")
        print("=" * 60)
        self.save_model()


# 训练入口
if __name__ == '__main__':
    agent = RL_QG_agent()
    
    # 训练模型
    agent.train_self_play(episodes=500)
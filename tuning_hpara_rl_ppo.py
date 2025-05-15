# -*- coding: utf-8 -*-
"""
Created on Thu May 15 17:21:11 2025

@author: roy
"""

'''
Reinforcement learning algorithm:
離散動作空間
 Q-Learning（適合小量離散超參數）
 DQN（Deep Q-Network）
連續動作空間
 DDPG（Deep Deterministic Policy Gradient）
 PPO（Proximal Policy Optimization）
 SAC（Soft Actor-Critic）
'''

import gym,numpy as np,warnings,pandas as pd,random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

warnings.filterwarnings('ignore')

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 載入資料
X, y = data,target
random_state = random.randint(1, 100000)
kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

# 自訂強化學習環境
class RandomForestTuningEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 動作空間：四個超參數（連續轉整數使用）(設定好超參數範圍)
        self.action_space = spaces.Box(low=np.array([10,2,2,0.1]), high=np.array([1000,20,20,1.0]), dtype=np.float32) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)  # 空狀態
        self.best_score = -np.inf
        self.last_n = 0
        self.last_d = 0
        self.last_s = 0
        self.last_f = 0
        self.last_reward = 0

    def step(self, action):
        # 模型超參數
        n_estimators = int(action[0])
        max_depth = int(action[1])
        min_samples_split = int(action[2])
        max_features = float(action[3])
        # 模型設定
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            n_jobs=-1, # 多核心使用
            random_state=random_state
        )

        # 10-fold CV，使用 R square 作為分數
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        mean_score = scores.mean()
        
        # normalization 儲存本輪資訊作為下次觀察用
        self.last_n = n_estimators / 200
        self.last_d = max_depth / 20
        self.last_s = min_samples_split / 20
        self.last_f = max_features
        self.last_reward = mean_score
        
        obs = np.array([self.last_n,self.last_d,self.last_s,self.last_f,self.last_reward])
        reward = mean_score  # 越高越好
        done = True
        
        return obs, reward, done, {
            'params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'max_features':max_features,
                'score': mean_score
            }
        }

    def reset(self):
        self.last_n = 0
        self.last_d = 0
        self.last_s = 0
        self.last_f = 0
        self.last_reward = 0
        return np.array([0, 0, 0, 0, 0], dtype=np.float32)

# 建立環境與訓練 PPO 模型
env = RandomForestTuningEnv()
model = PPO("MlpPolicy", env, verbose=1)
eval_callback = EvalCallback(
    env,                      # 評估時使用的環境
    eval_freq=500,            # 每 500 個 timesteps 評估一次
    best_model_save_path="./best_model",  # 儲存最佳模型的位置
    verbose=1
)

model.learn(total_timesteps=20000, callback=eval_callback)

# 測試 PPO 模型並找出最佳參數
obs = env.reset()
best_params = None
best_score = -np.inf

for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if reward > best_score:
        best_score = reward
        best_params = info['params']

print("📌 最佳超參數:")
print(best_params)

best_model = PPO.load("./best_model/best_model")

import matplotlib.pyplot as plt

data = np.load("./best_model/evaluations.npz")

timesteps = data["timesteps"]
results = data["results"]  # 每次評估時的 reward (n_eval_episodes 個)
mean_rewards = results.mean(axis=1)
std_rewards = results.std(axis=1)

# 繪圖 learning curve
plt.figure(figsize=(10, 6))
plt.plot(timesteps, mean_rewards, label="Mean reward")
plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
plt.xlabel("Timesteps")
plt.ylabel("Reward (CV Score)")
plt.title("PPO Tuning Performance over Time")
plt.legend()
plt.grid()
plt.show()
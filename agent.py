import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from network import ppo_network
import numpy as np


# -- params of Advantage-ベルマン方程式
GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

#epsiron parameter
EPS_START = 0.5
EPS_END = 0.1
EPS_STEPS = 200

class ppo_agent:
    def __init__(self):
        self.brain = ppo_network()
        self.memory = []
        self.R = 0
        return

    #input state to output action
    def action(self,state):
        state_t=torch.from_numpy(state.astype(np.float32)).clone()
        action,v =self.brain.action(state_t)
        #todo add randomize
        action = action.to('cpu').detach().numpy().copy()
        self.action_ = action

        a = np.random.choice(2,p=action)
        self.state = state
        return a

    def greedy_action(self,state):
        #if frame >= EPS_STEPS:   
        eps = EPS_END
        #else:
        #    eps = EPS_START + frame* (EPS_END - EPS_START) / EPS_STEPS  

        if np.random.random() <= eps:
            return np.random.choice(2)
        else:
            return self.action(state)

    def record(self, s, a, r, s_):
        def get_sample(memory, n):  # advantageを考慮し、メモリからnステップ後の状態とnステップ後までのRを取得する関数
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]
            return s, a, self.R, s_
        #advantage push brain
 # one-hotコーディングにしたa_catsをつくり、、s,a_cats,r,s_を自分のメモリに追加
        a_cats = np.zeros(2)  # turn action into one-hot representation
        a_cats[a] = 1
        self.memory.append((s, a_cats, r, s_))

        # 前ステップの「時間割引Nステップ分の総報酬R」を使用して、現ステップのRを計算
        self.R = (self.R + r * GAMMA_N) / GAMMA     # r0はあとで引き算している、この式はヤロミルさんのサイトを参照

        # advantageを考慮しながら、LocalBrainに経験を入力する
        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)
                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0  # 次の試行に向けて0にしておく

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)
            self.R = self.R - self.memory[0][2]     # # r0を引き算
            self.memory.pop(0)

 #       self.memory.append([self.state,self.action_,reward,terminate,next_state])
        return 


    def compute_logpi(mean,stddev,action):
        a1 = -0.5 * np.log(2*np.pi)
        a2 = torch.log(stddev)
        a3 = torch.exp(-((action,mean)**2)/(2*stddev))
        return a1 + a2 + a3

    def update(self):

        self.brain.update()
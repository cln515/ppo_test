import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pdb

NONE_STATE = np.zeros(4)
MIN_BATCH = 5000
# -- params of Advantage-ベルマン方程式
GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN
EPSILON = 0.4
LOSS_V = 0.2 
LOSS_ENTROPY = 0.01

class ppo_model(nn.Module):

    def __init__(self, state_num, action_num):
        super(ppo_model, self).__init__()
#model policy
        self.p_fc1 = nn.Linear(state_num, 100) 
        self.p_fc2 = nn.Linear(100, action_num)
#model value
        self.v_fc1 = nn.Linear(state_num, 100) 
        self.v_fc2 = nn.Linear(100, 1)


    def feed(self, state):
        x = F.leaky_relu(self.p_fc1(state))
        self.prob = F.softmax(self.p_fc2(x))
        
        x = F.leaky_relu(self.v_fc1(state))
        self.v = self.v_fc2(x)
        
        return self.prob, self.v



class ppo_network:

    def __init__(self):
        super(ppo_network, self).__init__()

        self.newnet = ppo_model(4,2)
        self.oldnet = ppo_model(4,2)


        self.train_queue = [[], [], [], [], []] 
        self.old_prob = 1

        
        return

    def action(self, s):
        p,v = self.newnet.feed(s)
        return p,v

    def train_push(self, s, a, r, s_):
        self.train_queue[0].append(s)
        self.train_queue[1].append(a)
        self.train_queue[2].append(r)

        if s_ is None:
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
        else:
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)        

    def update(self):
        if len(self.train_queue[0]) < MIN_BATCH:    # データがたまっていない場合は更新しない
            return

        s, a, r, s_, s_mask = self.train_queue
        self.train_queue = [[], [], [], [], []]
        s = np.vstack(s)    # vstackはvertical-stackで縦方向に行列を連結、いまはただのベクトル転置操作
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        # Nステップあとの状態s_から、その先得られるであろう時間割引総報酬vを求めます
        s_torch = torch.from_numpy(s_.astype(np.float32)).clone()
        r_torch = torch.from_numpy(r.astype(np.float32)).clone()
        a_torch = torch.from_numpy(a.astype(np.float32)).clone()
        s_mask_torch = torch.from_numpy(s_mask.astype(np.float32)).clone()
        
        _, v = self.newnet.feed(s_torch)


        # N-1ステップあとまでの時間割引総報酬rに、Nから先に得られるであろう総報酬vに割引N乗したものを足します
        r_torch = r_torch + GAMMA_N * v * s_mask_torch  # set v to 0 where s_ is terminal state
        #feed_dict = {self.s_t: s, self.a_t: a, self.r_t: r}     # 重みの更新に使用するデータ


        #optimization input s a r
        r_t = r_torch.clone().detach()
        a_t = a_torch
        s_t = torch.from_numpy(s.astype(np.float32)).clone()

        self.optimizer = optim.Adam(self.newnet.parameters(),lr = 0.001)
        for i in range(200):
            self.optimizer.zero_grad()
            p,v = self.newnet.feed(s_t)
            o_p,o_v = self.oldnet.feed(s_t)
            advantage = r_t - v 
            adv_nograd = advantage.clone().detach()

            self.prob = p + 1e-10
            
            self.old_prob = o_p.clone().detach() + 1e-10
            r_theta = torch.div(self.prob , self.old_prob)

#            action_theta = torch.sum(torch.mul(r_theta, a_t), axis = 1, keepdims = True)
            r_clip = torch.clamp(r_theta, 1-EPSILON, 1+EPSILON)
            
            advantage_CPI = torch.mul(r_theta , adv_nograd)  
            clipped_advantage_CPI = torch.mul(r_clip , adv_nograd) 
            loss_CLIP = torch.min(advantage_CPI, clipped_advantage_CPI)

            loss_value = LOSS_V * torch.square(advantage)  # minimize value error
            entropy = LOSS_ENTROPY * torch.sum(p * torch.log(p + 1e-10), dim=(1), keepdim=True)  # maximize entropy (regularization)

            self.loss_total = torch.mean(-loss_CLIP + loss_value - entropy)
            self.loss_total.backward()
            self.optimizer.step()

        self.oldnet.load_state_dict(self.newnet.state_dict())
#        




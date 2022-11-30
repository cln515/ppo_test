import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ppo_network(nn.Module):

    def __init__(self, state_num, action_num):
        super(ppo_network, self).__init__()

#model policy
        self.p_fc1 = nn.Linear(state_num, 30) 
        self.p_fc2 = nn.Linear(30, action_num)


#model value
        self.v_fc1 = nn.Linear(state_num, 30) 
        self.v_fc2 = nn.Linear(30, 1) 
        return

    def feed(self, state):
        x = F.leaky_relu(self.p_fc1(state))
        self.prob = F.softmax(self.p_fc2(x))
        
        x = F.leaky_relu(self.v_fc1(state))
        self.v = F.softmax(self.v_fc2(x))
        
        return self.prob, self.v
        




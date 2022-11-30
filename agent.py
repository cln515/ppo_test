import torch
from network import ppo_network
import numpy as np

class ppo_agent:
    def __init__(self):
        self.brain = ppo_network(4,1)
        self.memory = []
        return

    #input state to output action
    def action(self,state):
        state_t=torch.from_numpy(state.astype(np.float32)).clone()
        action,v =self.brain.feed(state_t)
        #todo add randomize
        action = action.to('cpu').detach().numpy().copy()
        self.action_ = action

        self.state = state
        return action

    def record(self,reward,terminate,next_state):

        self.memory.append([self.state,self.action_,reward,terminate,next_state])
        return 

    def update(self):
        self.memory = []
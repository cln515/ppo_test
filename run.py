from cartpole import cartpole
from time import sleep
from agent import ppo_agent

if __name__ == "__main__":
    env = cartpole()
    obs_state = env.reset()
    agent = ppo_agent()
    step = 0
    total_reward = 0
    while True:
        action = agent.action(obs_state)
        obs_state, reward, terminate  = env.step(action)
        agent.record(obs_state,reward,terminate)
        total_reward += reward
        step += 1
        #sleep(0.001)
        # terminate
        if step >= 1000 or terminate:
            print ("reset;", total_reward, step)
            step = 0
            total_reward = 0
            agent.update()
            env.reset()



from cartpole import cartpole
from time import sleep
from agent import ppo_agent

if __name__ == "__main__":
    env = cartpole()
    p_obs_state = env.reset()
    agent = ppo_agent()
    step = 0
    total_reward = 0

    iteration = 0

    while True:
        if iteration % 10000 < 100:
            test = True
        else:
            test = False
        action = agent.action(p_obs_state)
        obs_state, reward, terminate  = env.step(action)

        if not test:
            agent.record(p_obs_state,action,reward,obs_state)
        total_reward += reward
        step += 1
        p_obs_state = obs_state
        
        if step >= 1000 or terminate:
            print ("reset;", total_reward, step, iteration)
            iteration += 1
            step = 0
            total_reward = 0
            if not test:
                agent.update()
            env.reset()



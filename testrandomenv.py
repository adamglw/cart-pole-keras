import gym
import random

if __name__ in "__main__":
    env = gym.make ('CartPole-v0', render_mode='human') 
    states = env.observation_space.shape[0] 
    actions = env.action_space.n
    episodes = 15
    for episode in range (1, episodes+1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = random.choice([0,1])  # take a random step left or right
            state, reward, done, info = env.step(action) # apply action to the environment, if fail or end of game, done = true. This is one episode
            score += reward # based on the step we take we get a reward
            env.render() # render environment to see cart in action
        print(f"Episode: {episode}, Score: {score}")
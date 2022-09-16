import gym
import random

if __name__ in "__main__":
    env = gym.make ('CartPole-v0', render_mode='human') # build cart pole environment
    states = env.observation_space.shape[0] # the 4 states
    actions = env.action_space.n # the 2 actions

    # Let's try a few games
    episodes = 15
    for episode in range (1, episodes+1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = random.choice([0,1])  # take a random step left or right
            state, reward, done, info = env.step(action) # apply action to the environment
            score += reward # update reward
            env.render() # render environment to see cart in action
        print(f"Episode: {episode}, Score: {score}")
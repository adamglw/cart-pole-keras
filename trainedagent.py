import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import trainmodel


# Test Random Environment

# import random

# env = gym.make ('CartPole-v0', render_mode='human') # use make method from openai gym library to build cart pole environment
# states = env.observation_space.shape[0] # extract the states from the observation space, these are: cart position, cart velocity, pole angle, pole angular velocity
# actions = env.action_space.n # extract the actions from the action space, these are, push cart right or push cart left

# episodes = 15
# for episode in range (1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = random.choice([0,1])  # take a random step left or right
#         state, reward, done, info = env.step(action) # apply action to the environment, if fail or end of game, done = true. This is one episode
#         score += reward # based on the step we take we get a reward
#         env.render() # render environment to see cart in action
#     print(f"Episode: {episode}, Score: {score}")

# if __name__ in "__main__":
#     # create a deep learning model with keras
#     import numpy as np
#     import tensorflow

#     def build_model(states, actions):
#         model = tensorflow.keras.Sequential() # instantiate the sequential model
#         # print(f"states: {states}")
#         model.add(tensorflow.keras.layers.Flatten(input_shape=(1, states))) # flat node of 4 different states of cart pole
#         model.add(tensorflow.keras.layers.Dense(24, activation='relu')) # first dense node with relu activation function
#         model.add(tensorflow.keras.layers.Dense(24, activation='relu')) # second dense node with relu activation function
#         model.add(tensorflow.keras.layers.Dense(actions, activation='linear')) # last dense node has the two possible actions (left/right, i.e. 0/1)
#         return model

#     # build agent with keras-RL
#     from rl.agents import DQNAgent # other agents are available, like SARSA
#     from rl.policy import BoltzmannQPolicy # policuy based learning with this policy (as opposed to value based)
#     from rl.memory import SequentialMemory # retain memory

#     def build_agent(model, actions): # model defined above and actions are left and right
#         policy = BoltzmannQPolicy() # set up policy
#         memory = SequentialMemory(limit=50000, window_length=1) # set up memory
#         dqn = DQNAgent(model=model, memory=memory, policy=policy, 
#                     nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2) # set up dqn agent
#         return dqn

#     model = build_model(states, actions)
#     model.summary()
#     dqn = build_agent(model, actions)
#     dqn.compile(tensorflow.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])
#     dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

#     scores = dqn.test(env, nb_episodes=100, visualize=False)
#     print(np.mean(scores.history['episode_reward']))

#     _ = dqn.test(env, nb_episodes=15, visualize=True)

#     dqn.save_weights('dqn_weights.h5f', overwrite=True)

    ###
if __name__ in "__main__":
    env = trainmodel.gym.make('CartPole-v0')
    actions = env.action_space.n
    states = env.observation_space.shape[0]
    model = trainmodel.build_model(states, actions)
    dqn = trainmodel.build_agent(model, actions)
    dqn.compile(trainmodel.tensorflow.keras.optimizers.Adam(lr=1e-3), metrics=['mae'])

    dqn.load_weights('dqn_weights.h5f')

    _ = dqn.test(env, nb_episodes=5, visualize=True)
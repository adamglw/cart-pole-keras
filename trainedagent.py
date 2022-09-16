import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import buildmodel

if __name__ in "__main__":
    env = buildmodel.gym.make('CartPole-v0')
    actions = env.action_space.n
    states = env.observation_space.shape[0]
    model = buildmodel.build_model(states, actions)
    dqn = buildmodel.build_agent(model, actions)
    dqn.compile(buildmodel.tensorflow.keras.optimizers.Adam(lr=1e-3), metrics=['mae'])

    dqn.load_weights('dqn_weights.h5f')

    _ = dqn.test(env, nb_episodes=5, visualize=True)
import gym
import numpy as np

class QLearningAgent:
    def __init__(self, env, gamma, alpha, epsilon):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    # TODO: implement Q-learning update rule here
    def learn(self, old_state, action, reward, new_state):
        self.Q[old_state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[old_state, action])

    # selects action according to epsilon-greedy policy
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state, :])
        
    # Update Q values when an episode is finished
    def finish_episode(self, state, action, reward):
        self.Q[state, action] += self.alpha * (reward - self.Q[state, action])

    # decay epsilon after each episode
    def decay_epsilon(self):
        self.epsilon *= 0.99

    # train the agent for n episodes
    def train(self, n_episodes):
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                new_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, new_state)
                state = new_state
            self.decay_epsilon()





def main():
    env = gym.make('CartPole-v1', render_mode='human')
    agent = QLearningAgent(env, 0.99, 0.5, 0.1)
    agent.train(100000)
    
    print(agent.Q)
    # TODO: print learned policy here
    print("Learned policy:")
    for i in range(env.observation_space.n):
        print(np.argmax(agent.Q[i, :]), end=" ")
    print()
    # print reward table
    print("Reward table:")
    for i in range(env.observation_space.n):
        for j in range(env.action_space.n):
            print("{0:5.1f}".format(env.P[i][j][0][2]), end=" ")
        print()

if __name__ == "main":
    main()
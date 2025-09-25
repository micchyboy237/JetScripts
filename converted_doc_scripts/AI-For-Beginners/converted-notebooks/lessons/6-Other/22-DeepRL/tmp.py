from jet.logger import logger
from tensorflow import keras
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)


env = gym.make("CartPole-v1")

class REINFORCE:
  def __init__(self, env, path=None):
    self.env=env #import env
    self.state_shape=env.observation_space.shape # the state space
    self.action_shape=env.action_space.n # the action space
    self.gamma=0.99 # decay rate of past observations
    self.alpha=1e-4 # learning rate in the policy gradient
    self.learning_rate=0.01 # learning rate in deep learning

    if not path:
      self.model=self._create_model() #build model
    else:
      self.model=self.load_model(path) #import model

    self.states=[]
    self.gradients=[]
    self.rewards=[]
    self.probs=[]
    self.discounted_rewards=[]
    self.total_rewards=[]

  def hot_encode_action(self, action):
    '''encoding the actions into a binary list'''

    action_encoded=np.zeros(self.action_shape, np.float32)
    action_encoded[action]=1

    return action_encoded

  def remember(self, state, action, action_prob, reward):
    '''stores observations'''
    encoded_action=self.hot_encode_action(action)
    self.gradients.append(encoded_action-action_prob)
    self.states.append(state)
    self.rewards.append(reward)
    self.probs.append(action_prob)

  def _create_model(self):
    ''' builds the model using keras'''
    model=keras.Sequential()

    model.add(keras.layers.Dense(24, input_shape=self.state_shape, activation="relu"))
    model.add(keras.layers.Dense(12, activation="relu"))

    model.add(keras.layers.Dense(self.action_shape, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(lr=self.learning_rate))

    return model

  def get_action(self, state):
    '''samples the next action based on the policy probabilty distribution
      of the actions'''

    state=state.reshape([1, state.shape[0]])
    action_probability_distribution=self.model.predict(state).flatten()
    action_probability_distribution/=np.sum(action_probability_distribution)

    action=np.random.choice(self.action_shape,1,
                            p=action_probability_distribution)[0]

    return action, action_probability_distribution

  def get_discounted_rewards(self, rewards):
    '''Use gamma to calculate the total reward discounting for rewards
    Following - \gamma ^ t * Gt'''

    discounted_rewards=[]
    cumulative_total_return=0
    for reward in rewards[::-1]:
      cumulative_total_return=(cumulative_total_return*self.gamma)+reward
      discounted_rewards.insert(0, cumulative_total_return)

    mean_rewards=np.mean(discounted_rewards)
    std_rewards=np.std(discounted_rewards)
    norm_discounted_rewards=(discounted_rewards-
                          mean_rewards)/(std_rewards+1e-7) # avoiding zero div

    return norm_discounted_rewards

  def update_policy(self):
    '''Updates the policy network using the NN model.
    This function is used after the MC sampling is done - following
    \delta \theta = \alpha * gradient + log pi'''

    states=np.vstack(self.states)

    gradients=np.vstack(self.gradients)
    rewards=np.vstack(self.rewards)
    discounted_rewards=self.get_discounted_rewards(rewards)
    gradients*=discounted_rewards
    gradients=self.alpha*np.vstack([gradients])+self.probs

    history=self.model.train_on_batch(states, gradients)

    self.states, self.probs, self.gradients, self.rewards=[], [], [], []

    return history

  def train(self, episodes, rollout_n=1, render_n=50):
    '''train the model
        episodes - number of training iterations
        rollout_n- number of episodes between policy update
        render_n - number of episodes between env rendering '''

    env=self.env
    total_rewards=np.zeros(episodes)

    for episode in range(episodes):
      state=env.reset()
      done=False
      episode_reward=0 #record episode reward

      while not done:
        action, prob=self.get_action(state)
        next_state, reward, done, _=env.step(action)
        self.remember(state, action, prob, reward)
        state=next_state
        episode_reward+=reward

        if done:
          if episode%rollout_n==0:
            history=self.update_policy()

      total_rewards[episode]=episode_reward
      if episode%10==0:
        logger.debug(f"{episode} -> {episode_reward}")

    self.total_rewards=total_rewards

r = REINFORCE(env)

r.train(200)

plt.plot(r.total_rewards)

logger.info("\n\n[DONE]", bright=True)
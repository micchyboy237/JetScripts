from PIL import Image
from jet.logger import logger
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
## CartPole Skating

> **Problem**: If Peter wants to escape from the wolf, he needs to be able to move faster than him. We will see how Peter can learn to skate, in particular, to keep balance, using Q-Learning.

First, let's install the gym and import required libraries:
"""
logger.info("## CartPole Skating")

# !pip install gym pygame


"""
## Create a cartpole environment
"""
logger.info("## Create a cartpole environment")

env = gym.make("CartPole-v1")
logger.debug(env.action_space)
logger.debug(env.observation_space)
logger.debug(env.action_space.sample())

"""
To see how the environment works, let's run a short simulation for 100 steps.
"""
logger.info("To see how the environment works, let's run a short simulation for 100 steps.")

env.reset()

for i in range(100):
   env.render()
   env.step(env.action_space.sample())
env.close()

"""
During simulation, we need to get observations in order to decide how to act. In fact, `step` function returns us back current observations, reward function, and the `done` flag that indicates whether it makes sense to continue the simulation or not:
"""
logger.info("During simulation, we need to get observations in order to decide how to act. In fact, `step` function returns us back current observations, reward function, and the `done` flag that indicates whether it makes sense to continue the simulation or not:")

env.reset()

done = False
while not done:
   env.render()
   obs, rew, done, info = env.step(env.action_space.sample())
   logger.debug(f"{obs} -> {rew}")
env.close()

"""
We can get min and max value of those numbers:
"""
logger.info("We can get min and max value of those numbers:")

logger.debug(env.observation_space.low)
logger.debug(env.observation_space.high)

"""
## State Discretization
"""
logger.info("## State Discretization")

def discretize(x):
    return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))

"""
Let's also explore other discretization method using bins:
"""
logger.info("Let's also explore other discretization method using bins:")

def create_bins(i,num):
    return np.arange(num+1)*(i[1]-i[0])/num+i[0]

logger.debug("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))

ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
nbins = [20,20,10,10] # number of bins for each parameter
bins = [create_bins(ints[i],nbins[i]) for i in range(4)]

def discretize_bins(x):
    return tuple(np.digitize(x[i],bins[i]) for i in range(4))

"""
Let's now run a short simulation and observe those discrete environment values.
"""
logger.info("Let's now run a short simulation and observe those discrete environment values.")

env.reset()

done = False
while not done:
   obs, rew, done, info = env.step(env.action_space.sample())
   logger.debug(discretize(obs))
env.close()

"""
## Q-Table Structure
"""
logger.info("## Q-Table Structure")

Q = {}
actions = (0,1)

def qvalues(state):
    return [Q.get((state,a),0) for a in actions]

"""
## Let's Start Q-Learning!
"""
logger.info("## Let's Start Q-Learning!")

alpha = 0.3
gamma = 0.9
epsilon = 0.90

def probs(v,eps=1e-4):
    v = v-v.min()+eps
    v = v/v.sum()
    return v

Qmax = 0
cum_rewards = []
rewards = []
for epoch in range(100000):
    obs = env.reset()
    done = False
    cum_reward=0
    while not done:
        s = discretize(obs)
        if random.random()<epsilon:
            v = probs(np.array(qvalues(s)))
            a = random.choices(actions,weights=v)[0]
        else:
            a = np.random.randint(env.action_space.n)

        obs, rew, done, info = env.step(a)
        cum_reward+=rew
        ns = discretize(obs)
        Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
    cum_rewards.append(cum_reward)
    rewards.append(cum_reward)
    if epoch%5000==0:
        logger.debug(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
        if np.average(cum_rewards) > Qmax:
            Qmax = np.average(cum_rewards)
            Qbest = Q
        cum_rewards=[]

"""
## Plotting Training Progress
"""
logger.info("## Plotting Training Progress")

plt.plot(rewards)

"""
From this graph, it is not possible to tell anything, because due to the nature of stochastic training process the length of training sessions varies greatly. To make more sense of this graph, we can calculate **running average** over series of experiments, let's say 100. This can be done conveniently using `np.convolve`:
"""
logger.info("From this graph, it is not possible to tell anything, because due to the nature of stochastic training process the length of training sessions varies greatly. To make more sense of this graph, we can calculate **running average** over series of experiments, let's say 100. This can be done conveniently using `np.convolve`:")

def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))

"""
## Varying Hyperparameters and Seeing the Result in Action

Now it would be interesting to actually see how the trained model behaves. Let's run the simulation, and we will be following the same action selection strategy as during training: sampling according to the probability distribution in Q-Table:
"""
logger.info("## Varying Hyperparameters and Seeing the Result in Action")

obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()

"""
## Saving result to an animated GIF

If you want to impress your friends, you may want to send them the animated GIF picture of the balancing pole. To do this, we can invoke `env.render` to produce an image frame, and then save those to animated GIF using PIL library:
"""
logger.info("## Saving result to an animated GIF")

obs = env.reset()
done = False
i=0
ims = []
while not done:
   s = discretize(obs)
   img=env.render(mode='rgb_array')
   ims.append(Image.fromarray(img))
   v = probs(np.array([Qbest.get((s,a),0) for a in actions]))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
   i+=1
env.close()
ims[0].save('images/cartpole-balance.gif',save_all=True,append_images=ims[1::2],loop=0,duration=5)
logger.debug(i)

logger.info("\n\n[DONE]", bright=True)
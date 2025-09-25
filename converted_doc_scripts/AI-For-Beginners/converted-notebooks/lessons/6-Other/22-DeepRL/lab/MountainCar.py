from jet.logger import logger
import gym
import os
import shutil


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
# # Training Mountain Car to Escape

Lab Assignment from [AI for Beginners Curriculum](https://github.com/microsoft/ai-for-beginners).

Your goal is to train the RL agent to control [Mountain Car](https://www.gymlibrary.ml/environments/classic_control/mountain_car/) in Ollama Environment.

Let's start by creating the environment:
"""
logger.info("# # Training Mountain Car to Escape")

env = gym.make('MountainCar-v0')

"""
Let's see how the random experiment looks like:
"""
logger.info("Let's see how the random experiment looks like:")

state = env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        break

"""
Now the notebook is all yours - fell free to adopt Policy Gradients and Actor-Critic algorithms from the lesson to this problem!
"""
logger.info("Now the notebook is all yours - fell free to adopt Policy Gradients and Actor-Critic algorithms from the lesson to this problem!")



env.close()

logger.info("\n\n[DONE]", bright=True)
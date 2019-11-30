import pickle
import neat
import retro
import os
import cv2
import numpy as np
import time

ENV_NAME = 'SuperMarioBros-Nes'
IMG_SCALE_FACTOR = 8

config_path = os.path.join(os.path.dirname(__file__), 'config-feedforward') 
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

genome_path = os.path.join(os.path.dirname(__file__), f"{ENV_NAME}.pkl") 
with open(genome_path, "rb") as f:
    genome = pickle.load(f)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
env = retro.make(ENV_NAME, state='Level1-1')
obs = env.reset()
inx, iny, _ = env.observation_space.shape
inx = int(inx/IMG_SCALE_FACTOR)
iny = int(iny/IMG_SCALE_FACTOR)
done = False
while not done:
    env.render()
    time.sleep(0.01)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, (inx, iny))
    imgarray = np.ndarray.flatten(obs)
    act = net.activate(imgarray)
    obs, rew, done, info = env.step(act)
    if info['lives'] < 2:
        done = True
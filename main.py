import retro
import cv2
import os
import neat
import numpy as np
import pickle 
IMG_SCALE_FACTOR = 8
ENV_NAME = 'SuperMarioBros-Nes'
env = retro.make(game=ENV_NAME, state='Level1-1')

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        obs = env.reset()
        act = env.action_space.sample()
        inx, iny, inc = env.observation_space.shape
        inx = int(inx/IMG_SCALE_FACTOR)
        iny = int(iny/IMG_SCALE_FACTOR)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        done = False
        counter = 0
        fitness = 0
        cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        while not done:
            env.render()
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = cv2.resize(obs, (inx, iny))
            cv2.imshow('main', obs)
            cv2.waitKey(1)
            imgarray = np.ndarray.flatten(obs)
            act = net.activate(imgarray)
            obs, rew, done, info = env.step(act)
            if rew > 0.0:
                fitness += rew
                counter = 0
            else:
                counter+=1
            genome.fitness = fitness
            if counter == 300:
                done = True
                print(genome_id, fitness)

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, n=50)
    with open(f"{ENV_NAME}.pkl", "wb") as f:
        pickle.dump(file=f, obj=winner)

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'config-feedforward')
    run(config_path, checkpoint_path)

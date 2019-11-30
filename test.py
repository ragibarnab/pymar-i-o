import retro
import neat
import time
import cv2

scale = 8
env = retro.make(game='SuperMarioBros-Nes')
obs = env.reset()
cv2.namedWindow("main", cv2.WINDOW_NORMAL)
while True:
    action = [0, 1, 0, 0, 0, 0, 0, 1, 0]
    obs, rew, done, info = env.step(action)
    env.render()
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, (int(obs.shape[0]/scale), int(obs.shape[1]/scale)))
    cv2.imshow('main', obs)
    time.sleep(1.0/30)   # 30 frames / second
    if done:
        obs = env.reset()
env.close()

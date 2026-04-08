import pickle
import random

import pygame
from env import TrafficEnv
from agent import QAgent


use_ai = True  # change to False for random

pygame.init()

screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Traffic AI")
font = pygame.font.Font(None, 30)

env = TrafficEnv()
agent = QAgent()

with open("qtable.pkl", "rb") as f:
    agent.q_table = pickle.load(f)

state = env.reset()

running = True
clock = pygame.time.Clock()

while running:
    clock.tick(5)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if use_ai:
        action = agent.choose_action(state)
    else:
        action = random.randint(0, 1)
    state, _, _, _ = env.step(action)

    screen.fill((0, 0, 0))

    pygame.draw.rect(screen, (100, 100, 100), (200, 0, 200, 600))
    pygame.draw.rect(screen, (100, 100, 100), (0, 200, 600, 200))

    for i in range(env.lane1):
        pygame.draw.circle(screen, (255, 0, 0), (250, 500 - i * 5), 5)

    for i in range(env.lane2):
        pygame.draw.circle(screen, (0, 255, 0), (500 - i * 5, 250), 5)

    if env.emergency == 1:
        pygame.draw.circle(screen, (0, 0, 255), (300, 500), 10)
        em_text = font.render("EMERGENCY MODE", True, (0, 0, 255))
        screen.blit(em_text, (10, 70))

    text = font.render(
        f"Light: {'Lane1' if env.light == 0 else 'Lane2'}",
        True, (255, 255, 255)
    )
    screen.blit(text, (10, 10))

    action_text = font.render(f"AI Action: {action}", True, (255, 255, 0))
    screen.blit(action_text, (10, 40))

    pygame.display.update()

pygame.quit()

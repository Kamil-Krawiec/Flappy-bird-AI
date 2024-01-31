import os
import pickle

import neat
import pygame

from Objects.base import Base
from Objects.bird import Bird
from Objects.pipe import Pipe
from .config import const
import pandas as pd
from .functions import *

pygame.font.init()

WIN_WIDTH = const['WIN_WIDTH']
WIN_HEIGHT = const['WIN_HEIGHT']
CLOCK_TICK = const['CLOCK_TICK']
PIPE_DISTANCE = const['PIPE_DISTANCE']
FONT = const['FONT']
FITNESS_THRESHOLD = const['FITNESS_THRESHOLD']
GENERATIONS = const['GENERATIONS']

BG_IMG_PATH = os.path.join("./imgs", "bg.png")

STAT_FONT = pygame.font.SysFont(FONT, 25)


class FlappyBirdGame:
    def __init__(self):

        self.pipes = None
        self.birds = None
        self.nets = None
        self.ge = None
        self.stats = {
            "Generation": [],
            "Max Score": [],
            "Max Fitness": [],
        }
        self.gen = 0
        self.max_score = 0
        self.best_fitness = 0
        self.stat_font = pygame.font.SysFont("comicsans", 25)
        self.bg_img = pygame.transform.scale2x(pygame.image.load(BG_IMG_PATH))

    def update_pipes(self, score):

        rem = []
        for pipe in self.pipes:
            for x, bird in enumerate(self.birds):
                if pipe.collide(bird):
                    self.ge[x].fitness -= 1
                    self.birds.pop(x)
                    self.nets.pop(x)
                    self.ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    self.pipes.append(Pipe(PIPE_DISTANCE))
                    score += 1

                    for g in self.ge:
                        g.fitness += 5

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        for r in rem:
            self.pipes.remove(r)

        return score

    def update_birds(self, pipe_ind):
        for x, bird in enumerate(self.birds):
            bird.move()
            self.ge[x].fitness += 0.1
            output = self.nets[x].activate((bird.y,
                                            abs(bird.y - self.pipes[pipe_ind].height),
                                            abs(bird.y - self.pipes[pipe_ind].bottom),
                                            bird.vel,
                                            abs(bird.x - self.pipes[pipe_ind].x)))
            if output[0] > 0.5:
                bird.jump()

    def draw_window(self, win, base, score, fitness):
        win.blit(self.bg_img, (0, 0))
        for pipe in self.pipes:
            pipe.draw(win)

        self.max_score = max(self.max_score, score)
        self.best_fitness = max(self.best_fitness, fitness)

        self.draw_stats(win, score, fitness)

        for bird in self.birds:
            bird.draw(win)

        base.draw(win)
        # pygame.display.update()

    def draw_stats(self, win, score, fitness):
        # RIGHT TOP
        text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
        win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

        text = STAT_FONT.render("Max Score: " + str(self.max_score), 1, (255, 255, 255))
        win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 55))

        text = STAT_FONT.render("Max fitness: " + str(self.best_fitness), 1, (255, 255, 255))
        win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 95))

        # LEFT TOP
        text = STAT_FONT.render("Gen: " + str(self.gen), 1, (255, 255, 255))
        win.blit(text, (10, 10))

        text = STAT_FONT.render("Alive: " + str(len(self.birds)), 1, (255, 255, 255))
        win.blit(text, (10, 55))

        text = STAT_FONT.render("Curr fitness: " + str(round(fitness, 3)), 1, (255, 255, 255))
        win.blit(text, (10, 95))

    def fitness_function(self, genomes, config):
        self.gen += 1

        self.nets = []
        self.ge = []
        self.birds = []

        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            self.nets.append(net)
            self.birds.append(Bird(230, 350))
            g.fitness = 0
            self.ge.append(g)

        base = Base(730)
        self.pipes = [Pipe(PIPE_DISTANCE)]

        score = 0
        win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        clock = pygame.time.Clock()
        run = True

        while run and self.birds:
            clock.tick(CLOCK_TICK)
            self.handle_events()

            pipe_ind = 0

            if len(self.birds) > 0:
                if len(self.pipes) > 1 and self.birds[0].x > self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width():
                    pipe_ind = 1
            else:
                break

            self.update_birds(pipe_ind)
            score = self.update_pipes(score)
            self.update_offscreen_birds()

            fitness = max([round(g.fitness, 3) for g in self.ge]) if self.ge else 0

            if not run or any(genome.fitness >= FITNESS_THRESHOLD for genome in self.ge):
                break

            self.draw_window(win, base, score, fitness)
            base.move()

        self.stats["Generation"].append(self.gen)
        self.stats["Max Score"].append(self.max_score)
        self.stats["Max Fitness"].append(self.best_fitness)

    def update_offscreen_birds(self):
        for x, bird in enumerate(self.birds):

            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                self.birds.pop(x)
                self.nets.pop(x)
                self.ge.pop(x)

    def run(self, config_path, load_winner=False):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    config_path)

        if load_winner:
            with open('../winner.pickle', 'rb') as file:
                winner = pickle.load(file)
            self.fitness_function([(1, winner)], config)
        else:
            p = neat.Population(config)
            winner = p.run(self.fitness_function, GENERATIONS)

            with open('../winner.pickle', 'wb') as file:
                pickle.dump(winner, file)

        save_stats(self.stats)

    @staticmethod
    def handle_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

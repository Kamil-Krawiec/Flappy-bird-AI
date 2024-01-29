import os

import neat
import pygame

from base import Base
from bird import Bird
from pipe import Pipe

pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800

GEN = 0
MAX_SCORE = 0
BEST_FITNESS = 0


PIPE_DISTANCE = 650
STAT_FONT = pygame.font.SysFont("comicsans", 25)

BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))


def draw_window(win, birds, base, pipes, score,fitness):
    win.blit(BG_IMG, (0, 0))
    global MAX_SCORE
    global GEN
    global BEST_FITNESS

    MAX_SCORE = max(MAX_SCORE,score)
    BEST_FITNESS = max(BEST_FITNESS,fitness)

    for pipe in pipes:
        pipe.draw(win)

    # RIGHT TOP
    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Max Score: " + str(MAX_SCORE), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 55))

    text = STAT_FONT.render("Max fitness: " + str(BEST_FITNESS), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 95))

    # LEFT TOP
    text = STAT_FONT.render("Gen: " + str(GEN), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    text = STAT_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
    win.blit(text, (10, 55))

    text = STAT_FONT.render("Curr fitness: " + str(round(fitness,3)), 1, (255, 255, 255))
    win.blit(text, (10, 95))





    for bird in birds:
        bird.draw(win)
    base.draw(win)
    pygame.display.update()


def fitness(genomes, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(PIPE_DISTANCE)]

    score = 0

    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0

        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y,
                                       abs(bird.y - pipes[pipe_ind].height),
                                       abs(bird.y - pipes[pipe_ind].bottom),
                                       bird.vel,
                                       abs(bird.x - pipes[pipe_ind].x)))

            if output[0] > 0.5:
                bird.jump()

        rem = []

        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    pipes.append(Pipe(PIPE_DISTANCE))
                    score += 1

                    for g in ge:
                        g.fitness += 5

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):

            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        fitness = max([round(g.fitness,3) for g in ge]) if len(ge) > 0 else fitness

        draw_window(win, birds, base, pipes, score,fitness)
        base.move()


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(fitness, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config")

    run(config_path)

from Games.Flappy_bird_OOP import FlappyBirdGame
import os

if __name__ == "__main__":
    game = FlappyBirdGame()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config")
    game.run(config_path, load_winner=False)

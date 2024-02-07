import configparser
import os

import pandas as pd


def save_stats(game_stats):
    config_file_path = './neat-config'

    # Read config and game stats
    neat_config_data = read_neat_config(config_file_path)
    combined_df = concatenate_stats_and_config(game_stats, neat_config_data)

    # Save to file
    save_stats_to_file(combined_df, "game_stats.csv")


def save_stats_to_file(combined_df, filename):
    if os.path.isfile(filename):
        # Read the existing data to find the current max index
        existing_df = pd.read_csv(filename)
        if 'Game Index' in existing_df.columns:
            max_index = existing_df['Game Index'].max()
        else:
            max_index = 0

        # Increment the index for the new records
        combined_df.insert(0, 'Game Index', max_index + 1)
        combined_df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # If creating a new file, start index at 1

        combined_df.insert(0, 'Game Index', 1)
        combined_df.to_csv(filename, index=False)


def concatenate_stats_and_config(game_stats, config_stats):
    game_stats_df = pd.DataFrame.from_dict(game_stats, orient='columns')
    config_stats_df = pd.DataFrame([config_stats])

    # Replicate the config_stats_df to match the length of game_stats_df
    replicated_config_df = pd.concat([config_stats_df] * len(game_stats_df), ignore_index=True)

    # Concatenate along the columns (axis=1)
    combined_df = pd.concat([game_stats_df, replicated_config_df], axis=1)
    return combined_df


def read_neat_config(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Extracting relevant parameters
    neat_config = {
        "Population Size": config.getint('NEAT', 'pop_size'),
        "Hidden layers": config.getint('DefaultGenome', 'num_hidden'),
        "Activation Function": config.get('DefaultGenome', 'activation_default'),
        "Node delete probability": config.getfloat('DefaultGenome', 'node_delete_prob'),
        "Node add probability": config.getfloat('DefaultGenome', 'node_add_prob'),
    }

    return neat_config

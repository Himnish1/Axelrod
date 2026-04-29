#!/usr/bin/env python
"""
A demonstration script for using the mode parameter in the Match class.
This script shows how to use the mode parameter to control whether save_state() is called.
"""

import argparse
import axelrod as axl
from axelrod import all_strategies
from random import randint
import random
from tqdm import tqdm

def setup_opponents(num_opponents=5):
    """Generate random indices to select random players from all_strategies"""
    players_indices = []
    for i in range(num_opponents):
        number = randint(0, len(all_strategies)-35)
        if number in [15, 23, 101, 144]:
            i = i - 1
            continue
        players_indices.append(number)
    players = [axl.all_strategies[i]() for i in players_indices]
    print("Selected players:", players)
    return players

def train_mode():
    """Run matches in train mode to save the state of the QLearner"""
    print("Running in training mode...")
    players = setup_opponents()
    for player in tqdm(players):
        match = axl.Match(
            [axl.RiskyQLearner(), player],
            prob_end=0.001,
            p_A=random.choice([0, 0.25, 0.5, 0.75, 1]),
            mode='train'
        )
        match.play()
    print("Training complete. State has been saved.")

def test_mode():
    """Run matches in test mode without saving the state of the QLearner"""
    print("Running in testing mode...")
    players = setup_opponents()
    wins = 0
    for player in tqdm(players):
        match = axl.Match(
            [axl.RiskyQLearner(), player],
            prob_end=0.001,
            p_A=random.choice([0, 0.25, 0.5, 0.75, 1]),
            mode='test'
        )
        match.play()
        if match.final_score()[0] > match.final_score()[1]:
            wins += 1
    print(f"Testing complete. Won {wins} out of {len(players)} matches.")

def main():
    """Parse command-line arguments and run the appropriate mode"""
    parser = argparse.ArgumentParser(description='Demonstrate the mode parameter in Match')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='Mode to run: train or test')
    args = parser.parse_args()

    if args.mode == 'train':
        train_mode()
    else:  # args.mode == 'test'
        test_mode()

if __name__ == "__main__":
    main()

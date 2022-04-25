import argparse
import json
import os
import torch
import pprint

class Configuration(object):
    """Configuration parameters modified via command line"""
    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        parser = argparse.ArgumentParser()

        #General
        parser.add_argument('--data_workers', type=int, default=4, help='Number of parallel threads for data loading.')
        parser.add_argument('--print_every', type=int, default=200, help='Print stats to console every so many iters.')
        parser.add_argument('--eval_every', type=int, default=400, help='Evaluate validation set every so many iters.')
        parser.add_argument('--tag', default='reduce_plateau_scheduler', help='A custom tag for this experiment.')
        parser.add_argument('--seed', type=int, default=42, help='Random number generator seed.')
        parser.add_argument('--use_cuda', type=bool, default=True)
        parser.add_argument('--new_model', type=bool, default=True)

        #Data

        # Learning configurations.
        parser.add_argument('--lr', type=float, default=5.0e-4, help='Learning rate.')
        parser.add_argument('--n_epochs', type=int, default=700, help='Number of epochs.')
        parser.add_argument('--bs_train', type=int, default=32, help='Batch size for the training set.')
        parser.add_argument('--bs_eval', type=int, default=24, help='Batch size for valid/test set.')
        parser.add_argument('--linear_size', type=int, default=256, help='size of each model layer')
        parser.add_argument('--num_stage', type=int, default=18, help='# layers in linear model')
        parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        parser.add_argument('--lr_gamma', type=float, default=0.96)
        parser.add_argument('--scheduler', type=str, default='reduce_plateau')

        # model
        parser.add_argument('--encoder_type', type=str, default='gcn')
        parser.add_argument('--hidden_size', type=int, default=32)
        parser.add_argument('--output_size', type=int, default=16)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.5)

        config, unknown = parser.parse_known_args()
        # config = parser.parse_args()
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)







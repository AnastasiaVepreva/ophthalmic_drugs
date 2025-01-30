#!/usr/bin/env python
import argparse
import time
import os
from tensorboardX import SummaryWriter
from train_agent import train_agent


parser = argparse.ArgumentParser(description="Main script for running the model")
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--scoring-function', action='store', dest='scoring_function',
                    choices=['activity_model', 'tanimoto', 'no_sulphur', 'docking_score'],
                    default='tanimoto',
                    help='What type of scoring function to use.')
parser.add_argument('--scoring-function-decorate',  dest='scoring_function_decorate', default=False)
parser.add_argument('--scoring-function-kwargs', type=str, dest='scoring_function_kwargs')

parser.add_argument('--learning-rate', action='store', dest='learning_rate',
                    type=float, default=0.0005)
parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                    default=3000)
parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                    default=64)
parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                    default=20)
parser.add_argument('--experience', action='store', dest='experience_replay', type=int,
                    default=0, help='Number of experience sequences to sample each step. '\
                    '0 means no experience replay.')
parser.add_argument('--num-processes', action='store', dest='num_processes',
                    type=int, default=0,
                    help='Number of processes used to run the scoring function. "0" means ' \
                    'that the scoring function will be run in the main process.')
parser.add_argument('--prior', action='store', dest='restore_prior_from',
                    default='data/Prior.ckpt',
                    help='Path to an RNN checkpoint file to use as a Prior')
parser.add_argument('--agent', action='store', dest='restore_agent_from',
                    default='data/Prior.ckpt',
                    help='Path to an RNN checkpoint file to use as a Agent.')
parser.add_argument('--optimizer', action='store', dest='restore_optimizer_from',
                    default=None,
                    help='Path to an Agent optimizer checkpoint file')
parser.add_argument('--save-dir', action='store', dest='save_dir',
                    help='Path where results and model are saved. Default is data/results/run_<datetime>.')
parser.add_argument('--seed', type=int, default=42,
                    help='Path where results and model are saved. Default is data/results/run_<datetime>.')

if __name__ == "__main__":

    arg_dict = vars(parser.parse_args())

    if arg_dict['scoring_function_kwargs']:
        kwarg_str = arg_dict.pop('scoring_function_kwargs')
        kwarg_list = kwarg_str.split(' ')
        if not len(kwarg_list) % 2 == 0:
            raise ValueError("Scoring function kwargs must be given as pairs, "\
                             "but got a list with odd length.")
        # unsafe cast used only for private usage!
        keys = kwarg_list[::2]
        values = kwarg_list[1::2]
        values = [v if v.startswith('/home/jovyan') else eval(v) for v in values]
        kwarg_dict = {i:j for i, j in zip(keys, values)}
        arg_dict['scoring_function_kwargs'] = kwarg_dict
    else:
        arg_dict['scoring_function_kwargs'] = dict()

    print(arg_dict)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    save_dir = arg_dict['save_dir']
    logs_dir = os.path.join(save_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(logs_dir)
    writer.add_text('train', str(arg_dict), 0)
    arg_dict['writer'] = writer

    train_agent(**arg_dict)

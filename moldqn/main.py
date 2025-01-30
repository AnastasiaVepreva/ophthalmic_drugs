#!/usr/bin/env python
import argparse
import os
from train import train
import utils


parser = argparse.ArgumentParser(description="Main script for running the model")


parser.add_argument('--use_dense_rewards', type=utils.str2bool, default=True)
parser.add_argument('--start_molecule', type=str, default=None)
parser.add_argument('--epsilon_start', type=float, default=1.0)
parser.add_argument('--epsilon_end', type=float, default=0.01)
parser.add_argument('--epsilon_decay', type=int, default=2000)
parser.add_argument('--optimizer', type=str, default="Adam")
parser.add_argument('--polyak', type=float, default=0.995)
parser.add_argument('--atom_types', type=utils.str2strs, default=["C", "O", "N"])
parser.add_argument('--max_steps_per_episode', type=int, default=40)
parser.add_argument('--allow_removal', type=utils.str2bool, default=True)
parser.add_argument('--allow_no_modification', type=utils.str2bool, default=True)
parser.add_argument('--allow_bonds_between_rings', type=utils.str2bool, default=False)
parser.add_argument('--allowed_ring_sizes', type=utils.str2ints, default=[3, 4, 5, 6])
parser.add_argument('--replay_buffer_size', type=int, default=1000000)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--fingerprint_radius', type=int, default=3)
parser.add_argument('--fingerprint_length', type=int, default=2048)
parser.add_argument('--discount_factor', type=float, default=0.9)
parser.add_argument('--exp_root', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--episodes', type=int, default=0)
parser.add_argument('--iterations', type=int, default=200000)
parser.add_argument('--update_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_updates_per_it', type=int, default=1)
parser.add_argument('--eps_threshold', type=float, default=1.0)
parser.add_argument('--num_mols', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)

# Docking
parser.add_argument('--exhaustiveness', type=int, default=1)
parser.add_argument('--num_modes', type=int, default=10)
parser.add_argument('--num_sub_proc', type=int, default=1)
parser.add_argument('--n_conf', type=int, default=1)
parser.add_argument('--error_val', type=float, default=99.9)
parser.add_argument('--timeout_dock', type=int, default=100)
parser.add_argument('--timeout_gen3d', type=int, default=40)
parser.add_argument('--alpha', type=int, default=0.1)
parser.add_argument('--receptor_file', type=str, required=True)
parser.add_argument('--box_center', type=utils.str2floats, required=True)
parser.add_argument('--box_size', type=utils.str2floats, required=True)
parser.add_argument('--vina_program', type=str, required=True)

parser.add_argument('--local_rank', type=int, default=0)


def update_args(args):
    exp_dir = os.path.join(args.exp_root, args.name)
    docking_config = {
        'exhaustiveness': args.exhaustiveness,
        'num_sub_proc': args.num_sub_proc,
        'num_modes': args.num_modes,
        'timeout_gen3d': args.timeout_gen3d,
        'timeout_dock': args.timeout_dock,
        'seed': args.seed,
        'n_conf': args.n_conf,
        'error_val': args.error_val,
        'alpha': args.alpha,
        'receptor_file': args.receptor_file,
        'box_center': args.box_center,
        'box_size': args.box_size,
        'vina_program': args.vina_program,
        'temp_dir': os.path.join(exp_dir, 'tmp')
    }
    args.ckpt_dir = os.path.join(exp_dir, 'ckpt')
    args.tb_dir = os.path.join(exp_dir, "logs")
    args.mol_dir = os.path.join(exp_dir, 'mols')
    args.docking_config = docking_config
    return args


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    args = parser.parse_args()
    args = update_args(args)
    train(args)
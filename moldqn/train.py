import os
import json
from itertools import chain
import torch
from agent import Agent, DockingRewardMolecule
import math
import utils
import numpy as np
from tensorboardX import SummaryWriter


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    utils.set_seed(args.seed)

    environment = DockingRewardMolecule(
        discount_factor=args.discount_factor,
        config=args.docking_config,
        atom_types=set(args.atom_types),
        init_mol=args.start_molecule,
        allow_removal=args.allow_removal,
        allow_no_modification=args.allow_no_modification,
        allow_bonds_between_rings=args.allow_bonds_between_rings,
        allowed_ring_sizes=set(args.allowed_ring_sizes),
        max_steps=args.max_steps_per_episode,
    )

    # DQN Inputs and Outputs:
    # input: appended action (fingerprint_length + 1) .
    # Output size is (1).

    agent = Agent(args.fingerprint_length + 1, 1, device, args.replay_buffer_size, args.optimizer, args.learning_rate)
    writer = SummaryWriter(args.tb_dir)
    writer.add_text('args', str(vars(args)), 0)
    os.makedirs(args.mol_dir, exist_ok=True)

    environment.initialize()

    eps_threshold = args.eps_threshold
    batch_losses = []
    start_idx = 0
    next_states = []
    episodes = args.episodes
    for it in range(args.iterations):
        
        steps_left = args.max_steps_per_episode - environment.num_steps_taken

        # Compute a list of all possible valid actions. (Here valid_actions stores the states after taking the possible actions)
        valid_actions = list(environment.get_valid_actions())

        # Append each valid action to steps_left and store in observations.
        observations = np.vstack(
            [
                np.append(
                    utils.get_fingerprint(
                        act, args.fingerprint_length, args.fingerprint_radius
                    ),
                    steps_left,
                )
                for act in valid_actions
            ]
        )  # (num_actions, fingerprint_length)

        observations_tensor = torch.Tensor(observations)
        # Get action through epsilon-greedy policy with the following scheduler.
        # eps_threshold = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * \
        #     math.exp(-1. * it / args.epsilon_decay)

        a = agent.get_action(observations_tensor, eps_threshold)

        # Find out the new state (we store the new state in "action" here. Bit confusing but taken from original implementation)
        action = valid_actions[a]
        # Take a step based on the action
        result = environment.step(action)

        action_fingerprint = np.append(
            utils.get_fingerprint(action, args.fingerprint_length, args.fingerprint_radius),
            steps_left,
        )

        next_state, reward, done = result
        next_states.append(next_state)

        # Compute number of steps left
        steps_left = args.max_steps_per_episode - environment.num_steps_taken

        # Append steps_left to the new state and store in next_state
        next_state = utils.get_fingerprint(
            next_state, args.fingerprint_length, args.fingerprint_radius
        )  # (fingerprint_length)

        action_fingerprints = np.vstack(
            [
                np.append(
                    utils.get_fingerprint(
                        act, args.fingerprint_length, args.fingerprint_radius
                    ),
                    steps_left,
                )
                for act in environment.get_valid_actions()
            ]
        )  # (num_actions, fingerprint_length + 1)

        # Update replay buffer (state: (fingerprint_length + 1), action: _, reward: (), next_state: (num_actions, fingerprint_length + 1),
        # done: ()

        agent.replay_buffer.add(
            obs_t=action_fingerprint,  # (fingerprint_length + 1)
            action=0,  # No use
            reward=reward,
            obs_tp1=action_fingerprints,  # (num_actions, fingerprint_length + 1)
            done=result.terminated,
        )

        if done:
            final_reward = reward
            if episodes != 0 and len(batch_losses) != 0:
                writer.add_scalar("episode_loss", np.array(batch_losses).mean(), episodes)
            if episodes != 0 and episodes % 2 == 0 and len(batch_losses) != 0:
                print(
                    "mean loss in episode {} is {}".format(
                        episodes, np.array(batch_losses).mean()
                    )
                )
            episodes += 1
            eps_threshold *= 0.99907
            batch_losses = []
            environment.initialize()

        if it % args.update_interval == 0 and agent.replay_buffer.__len__() >= args.batch_size:
            buffer = agent.replay_buffer
            end_idx = buffer._next_idx
            ids = utils.get_ids(len(buffer), start_idx, end_idx)
            obs_ts, actions, rewards, obs_tp1s, dones = buffer._encode_sample(ids)
            if args.use_dense_rewards:
                rewards = environment._reward_batch(next_states)
                scales = utils.get_scales(dones, environment.num_steps_taken, environment.max_steps, discount_factor=environment.discount_factor)
                rewards *= scales
            else:
                returns = environment._reward_batch([next_state for next_state, done in zip(next_states, dones) if done])
                rewards[dones] = returns
            if dones.any():
                writer.add_scalar('reward', rewards[dones].mean(), it)
            with open (os.path.join(args.mol_dir, 'train.csv'), 'a') as f:
                for smi, r, d in zip(next_states, rewards, dones):
                    f.write(f'{smi},{r},{d},{it}\n')
            for idx, obs_t, action, reward, obs_tp1, done in zip(ids, obs_ts, actions, rewards, obs_tp1s, dones):
                buffer._storage[idx] = (obs_t, action, reward, obs_tp1, done)
            start_idx = end_idx
            next_states = []
            for update in range(args.num_updates_per_it):
                loss = agent.update_params(args.batch_size, args.gamma, args.polyak)
                loss = loss.item()
                batch_losses.append(loss)

        if it % args.save_interval == 0:
            utils.save_agent(agent, args.ckpt_dir, it)

    utils.save_agent(agent, args.ckpt_dir, it)
    samples = sample(agent, environment, max_steps_per_episode=args.max_steps_per_episode, 
                    fingerprint_length=args.fingerprint_length, fingerprint_radius=args.fingerprint_radius, 
                    eps_threshold=eps_threshold, num_mols=args.num_mols)

    with open(os.path.join(args.mol_dir, "gen.json"), 'wt') as f:
        json.dump(samples, f, indent=4)


def sample(agent, environment, max_steps_per_episode=40, fingerprint_length=2048, fingerprint_radius=3, eps_threshold=0.01, num_mols=1000):
    samples = []
    for _ in range(num_mols):
        done = False
        environment.initialize()
        while not done:
            steps_left = max_steps_per_episode - environment.num_steps_taken

            # Compute a list of all possible valid actions. (Here valid_actions stores the states after taking the possible actions)
            valid_actions = list(environment.get_valid_actions())

            # Append each valid action to steps_left and store in observations.
            observations = np.vstack(
                [
                    np.append(
                        utils.get_fingerprint(
                            act, fingerprint_length, fingerprint_radius
                        ),
                        steps_left,
                    )
                    for act in valid_actions
                ]
            )  # (num_actions, fingerprint_length)

            observations_tensor = torch.Tensor(observations)
            # Get action through epsilon-greedy policy.

            a = agent.get_action(observations_tensor, eps_threshold)

            # Find out the new state (we store the new state in "action" here. Bit confusing but taken from original implementation)
            action = valid_actions[a]
            # Take a step based on the action
            next_state, _, done = environment.step(action)

        samples.append(next_state)

    return samples


if __name__ == "__main__":
    train()    
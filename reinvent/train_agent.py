#!/usr/bin/env python

import random
import torch
import pickle
import numpy as np
import time
import os
import json
from shutil import copyfile

from model import RNN
from data_structs import Vocabulary, Experience
from scoring_functions import get_scoring_function, valid
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique


def load_routine(state_dict):
    names = [
        'gru_1.bias_ih',
        'gru_1.bias_hh',
        'gru_2.bias_ih',
        'gru_2.bias_hh',
        'gru_3.bias_ih',
        'gru_3.bias_hh'
    ]
    for name in names:
        state_dict[name] = state_dict[name].squeeze(0)

    return state_dict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_agent(restore_prior_from='data/Prior.ckpt',
                restore_agent_from='data/Prior.ckpt',
                restore_optimizer_from=None,
                scoring_function='tanimoto',
                scoring_function_kwargs=None,
                scoring_function_decorate=True,
                save_dir=None, learning_rate=0.0005,
                batch_size=64, n_steps=3000,
                num_processes=0, sigma=60,
                experience_replay=0,
                writer=None,
                save_frequency=1000, seed=42, **kwargs):

    set_seed(seed)
    voc = Vocabulary(init_from_file="data/Voc")

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)


    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        map_location = None
    else:
        map_location=lambda storage, loc: storage

    Prior.rnn.load_state_dict(load_routine(torch.load('data/Prior.ckpt', map_location=map_location)))
    Agent.rnn.load_state_dict(load_routine(torch.load(restore_agent_from, map_location=map_location)))


    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0005)
    if restore_optimizer_from:
        optimizer.load_state_dict(torch.load(restore_optimizer_from, map_location=map_location))

    # Scoring_function
    scoring_function = get_scoring_function(scoring_function=scoring_function, decorate=scoring_function_decorate, 
                                            num_processes=num_processes, **scoring_function_kwargs)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    print("Model initialized, starting training...")

    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)

        non_unique_seqs = seqs
        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        score = scoring_function(smiles)

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience)>4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(len(smiles)):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))

        non_unique_smiles = seq_to_smiles(non_unique_seqs, voc)
        unique_log_items = dict(zip(smiles, zip(score, agent_likelihood, agent_likelihood, augmented_likelihood)))
        non_unique_log_items = {smile: unique_log_items[smile] for smile in non_unique_smiles}
        with open(os.path.join(save_dir, 'logs', 'train.csv'), 'a') as f:
            for smile, log_items in non_unique_log_items.items():
                f.write(','.join(map(str, (smile, *log_items, time_elapsed, step))) + '\n')

        
        writer.add_scalar('Valid', np.mean(list(map(valid, non_unique_smiles))).item(), step)
        writer.add_scalar('Avg Score', np.mean(score).item(), step)
        writer.add_scalar('Max Score', np.max(score).item(), step)
        writer.add_scalar('Agent', np.mean(agent_likelihood).item(), step)
        writer.add_scalar('Prior', np.mean(prior_likelihood).item(), step)
        writer.add_scalar('Augmented', np.mean(augmented_likelihood).item(), step)
        writer.add_text('Smiles', ','.join(non_unique_smiles), step)

        if step % save_frequency == 0:
            torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent_{}.ckpt'.format(str(step).zfill(5))))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, 'Optimizer_{}.ckpt'.format(str(step).zfill(5))))

    copyfile('train_agent.py', os.path.join(save_dir, "train_agent.py"))

    experience.print_memory(os.path.join(save_dir, "memory"))
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

    seqs, _, _ = Agent.sample(1000)
    smiles = seq_to_smiles(seqs, voc)
    with open(os.path.join(save_dir, "sample.json"), 'wt') as f:
        json.dump(smiles, f)

if __name__ == "__main__":
    train_agent()

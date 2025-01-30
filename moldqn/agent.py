import os
import glob
from subprocess import run
from functools import lru_cache
from multiprocessing import Pool
import torch
import torch.nn as nn
import numpy as np
import torch.optim as opt
import utils
from dqn import MolDQN
from rdkit import Chem
from rdkit.Chem import QED
from environment import Molecule
from replay_buffer import ReplayBuffer


def init(env):
    os.environ = env


class QEDRewardMolecule(Molecule):
    """The molecule whose reward is the QED."""

    def __init__(self, discount_factor, **kwargs):
        """Initializes the class.

    Args:
      discount_factor: Float. The discount factor. We only
        care about the molecule at the end of modification.
        In order to prevent a myopic decision, we discount
        the reward at each step by a factor of
        discount_factor ** num_steps_left,
        this encourages exploration with emphasis on long term rewards.
      **kwargs: The keyword arguments passed to the base class.
    """
        super(QEDRewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor

    def _reward(self):
        """Reward of a state.

    Returns:
      Float. QED of the current state.
    """
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        qed = QED.qed(molecule)
        return qed * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class DockingRewardMolecule(Molecule):
    """The molecule whose reward is the Docking score."""

    def __init__(self, discount_factor, config, max_cache_size=5000, **kwargs):
        """Initializes the class.

    Args:
      discount_factor: Float. The discount factor. We only
        care about the molecule at the end of modification.
        In order to prevent a myopic decision, we discount
        the reward at each step by a factor of
        discount_factor ** num_steps_left,
        this encourages exploration with emphasis on long term rewards.
      **kwargs: The keyword arguments passed to the base class.
    """
        super(DockingRewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor
        self.config = config
        self.temp_dir = config['temp_dir']
        self.seed = config['seed']
        self.max_cache_size = max_cache_size
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.results = dict()

    def _reward(self):
        """Reward of a state.

    Returns:
      Float. Docking score of the current state.
    """
        smile = self._state
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            return 0.0
        return 0.0

    def _reward_batch(self, smiles_list):
        smiles_set = list(set(smiles_list).difference(self.results.keys()))

        if smiles_set:
            binding_affinities = list()
            fnames = list(map(str, range(len(smiles_set))))
            for i in range(self.config['n_conf']):
                child_env = os.environ.copy()
                child_env['OB_RANDOM_SEED'] = str(self.seed + i)
                with Pool(processes=self.config['num_sub_proc'], initializer=init, initargs=(child_env,)) as pool:
                    binding_affinities.append(pool.starmap(self.docking, zip(smiles_set, fnames)))

            binding_affinities = dict(zip(smiles_set, np.minimum.reduce(binding_affinities)))
            self.results = {**self.results, **binding_affinities}

        files = glob.glob(f"{self.temp_dir}/*")
        for file in files:
            os.remove(file)

        rewards = self._postprocess([self.results[smile] for smile in smiles_list])
        for smi in list(self.results.keys())[:max(0, len(self.results) - self.max_cache_size)]:
            del self.results[smi]
        return rewards

    def _postprocess(self, affinities):
        return self.config['alpha'] * -np.minimum(affinities, 0.0).astype(np.float32)

    def docking(self, smi, fname):
        return DockingRewardMolecule._docking(smi, fname, **self.config)

    @staticmethod
    def _docking(smi, fname, *, vina_program, receptor_file, temp_dir, box_center,
            box_size, error_val, seed, num_modes, exhaustiveness,
            timeout_dock, timeout_gen3d, **kwargs):

        ligand_file = os.path.join(temp_dir, "ligand_{}.pdbqt".format(fname))
        docking_file = os.path.join(temp_dir, "dock_{}.pdbqt".format(fname))

        run_line = "obabel -:{} --gen3D -h -O {}".format(smi, ligand_file)
        try:
            result = run(run_line.split(), capture_output=True, text=True, timeout=timeout_gen3d, env=os.environ)
        except:
            return error_val

        if "Open Babel Error" in result.stdout or "3D coordinate generation failed" in result.stdout:
            return error_val

        run_line = vina_program
        run_line += " --receptor {} --ligand {} --out {}".format(receptor_file, ligand_file, docking_file)
        run_line += " --center_x {} --center_y {} --center_z {}".format(*box_center)
        run_line += " --size_x {} --size_y {} --size_z {}".format(*box_size)
        run_line += " --num_modes {}".format(num_modes)
        run_line += " --exhaustiveness {}".format(exhaustiveness)
        run_line += " --seed {}".format(seed)
        try:
            result = run(run_line.split(), capture_output=True, text=True, timeout=timeout_dock)
        except:
            return error_val

        return DockingRewardMolecule.parse_output(result.stdout, error_val)

    @staticmethod
    def parse_output(result, error_val):
        result_lines = result.split('\n')
        check_result = False
        affinity = error_val

        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            break
        return affinity


class Agent(object):
    def __init__(self, input_length, output_length, device, replay_buffer_size, optimizer, learning_rate):
        self.device = device
        self.dqn, self.target_dqn = (
            MolDQN(input_length, output_length).to(self.device),
            MolDQN(input_length, output_length).to(self.device),
        )
        for p in self.target_dqn.parameters():
            p.requires_grad = False
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.optimizer = getattr(opt, optimizer)(
            self.dqn.parameters(), lr=learning_rate
        )

    def get_action(self, observations, epsilon_threshold):

        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn.forward(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).numpy()

        return action

    def update_params(self, batch_size, gamma, polyak):
        # update target network

        # sample batch of transitions
        states, _, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        q_t = torch.zeros(batch_size, 1, requires_grad=False)
        v_tp1 = torch.zeros(batch_size, 1, requires_grad=False)
        for i in range(batch_size):
            state = (
                torch.FloatTensor(states[i])
                .unsqueeze(0)
                .to(self.device)
            )
            q_t[i] = self.dqn(state)

            next_state = (
                torch.FloatTensor(next_states[i])
                .to(self.device)
            )
            v_tp1[i] = torch.max(self.target_dqn(next_state))

        rewards = torch.FloatTensor(rewards).reshape(q_t.shape).to(self.device)
        q_t = q_t.to(self.device)
        v_tp1 = v_tp1.to(self.device)
        dones = torch.FloatTensor(dones).reshape(q_t.shape).to(self.device)

        # # get q values
        q_tp1_masked = (1 - dones) * v_tp1
        q_t_target = rewards + gamma * q_tp1_masked
        td_error = q_t - q_t_target

        q_loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error * td_error,
            1.0 * (torch.abs(td_error) - 0.5),
        )
        q_loss = q_loss.mean()

        # backpropagate
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return q_loss

#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
import time
import pickle
import re
import threading
import os
from itertools import repeat
from multiprocessing import Pool
from subprocess import run, PIPE
import glob
from functools import partial
rdBase.DisableLog('rdApp.error')

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a single SMILES of
   argument and returns a float. A multiprocessing class will then spawn workers and divide the
   list of SMILES given between them.

   Passing *args and **kwargs through a subprocess call is slightly tricky because we need to know
   their types - everything will be a string once we have passed it. Therefor, we instead use class
   attributes which we can modify in place before any subprocess is created. Any **kwarg left over in
   the call to get_scoring_function will be checked against a list of (allowed) kwargs for the class
   and if a match is found the value of the item will be the new value for the class.

   If num_processes == 0, the scoring function will be run in the main process. Depending on how
   demanding the scoring function is and how well the OS handles the multiprocessing, this might
   be faster than multiprocessing in some cases."""

class no_sulphur():
    """Scores structures based on not containing sulphur."""

    kwargs = []

    def __init__(self):
        pass
    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            has_sulphur = any(atom.GetAtomicNum() == 16 for atom in mol.GetAtoms())
            return float(not has_sulphur)
        return 0.0

class tanimoto():
    """Scores structures based on Tanimoto similarity to a query structure.
       Scores are only scaled up to k=(0,1), after which no more reward is given."""

    kwargs = ["k", "query_structure"]
    k = 0.7
    query_structure = "Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)C(F)(F)F"

    def __init__(self):
        query_mol = Chem.MolFromSmiles(self.query_structure)
        self.query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=True, useFeatures=True)

    def __call__(self, smiles):
        smiles = []
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        for mol in mols:
            fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
            score = DataStructs.TanimotoSimilarity(self.query_fp, fp)
            score = min(score, self.k) / self.k
            return float(score)
        return 0.0

class activity_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'data/clf.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = activity_model.fingerprints_from_mol(mol)
            score = self.clf.predict_proba(fp)[:, 1]
            return float(score)
        return 0.0

    @classmethod
    def fingerprints_from_mol(cls, mol):
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        size = 2048
        nfp = np.zeros((1, size), np.int32)
        for idx,v in fp.GetNonzeroElements().items():
            nidx = idx%size
            nfp[0, nidx] += int(v)
        return nfp

def init(env):
    os.environ = env

def valid(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return 1

class docking_score():
    def __init__(self, **config):
        self.config = config
        self.temp_dir = config['temp_dir']
        self.seed = config['seed']
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        print(config)
        self.results = dict()

    def __call__(self, smiles_list):
        smiles_set = list(set(smiles_list).difference(self.results))

        if smiles_set:
            binding_affinities = list()
            fnames = list(map(str, range(len(smiles_set))))
            for i in range(self.config['n_conf']):
                child_env = os.environ.copy()
                child_env['OB_RANDOM_SEED'] = str(self.seed + i)
                with Pool(processes=self.config['num_sub_proc'], initializer=init, initargs=(child_env,)) as pool:
                    binding_affinities.append(pool.starmap(self.docking, zip(smiles_set, fnames)))

            binding_affinities = dict(zip(smiles_set, np.minimum.reduce(binding_affinities)))
            #print(binding_affinities)
            self.results = {**self.results, **binding_affinities}

        files = glob.glob(f"{self.temp_dir}/*")
        for file in files:
            os.remove(file)

        return self._postprocess([self.results[smile] for smile in smiles_list])

    def _postprocess(self, affinities):
        return self.config['alpha'] * -np.minimum(affinities, 0.0).astype(np.float32)

    def docking(self, smi, fname):
        return docking_score._docking(smi, fname, **self.config)

    @staticmethod
    def _docking(smi, fname, *, vina_program, receptor_file, temp_dir, box_center,
            box_size, error_val, seed, num_modes, exhaustiveness,
            timeout_dock, timeout_gen3d, **kwargs):

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            #print(error_val)
            return error_val

        ligand_file = os.path.join(temp_dir, "ligand_{}.pdbqt".format(fname))
        docking_file = os.path.join(temp_dir, "dock_{}.pdbqt".format(fname))

        run_line = "obabel -:{} --gen3D -h -O {}".format(smi, ligand_file)
        try:
            result = run(run_line.split(), capture_output=True, text=True, timeout=timeout_gen3d, env=os.environ)
            #print(result)
        except:
            #print(error_val)
            return error_val

        if "Open Babel Error" in result.stdout or "3D coordinate generation failed" in result.stdout:
            #print(error_val)
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
            #print(result)
        except:
            #print(error_val)
            return error_val

        return docking_score.parse_output(result.stdout, error_val)

    @staticmethod
    def parse_output(result, error_val):
        result_lines = result.split('\n')
        check_result = False
        affinity = error_val

        for result_line in result_lines:
            print(result_line)
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
        print(affinity)
        return affinity

class Worker():
    """A worker class for the Multiprocessing functionality. Spawns a subprocess
       that is listening for input SMILES and inserts the score into the given
       index in the given list."""
    def __init__(self, scoring_function=None):
        """The score_re is a regular expression that extracts the score from the
           stdout of the subprocess. This means only scoring functions with range
           0.0-1.0 will work, for other ranges this re has to be modified."""

        self.proc = pexpect.spawn('./multiprocess.py ' + scoring_function,
                                  encoding='utf-8')

        print(self.is_alive())

    def __call__(self, smile, index, result_list):
        self.proc.sendline(smile)
        output = self.proc.expect([re.escape(smile) + " 1\.0+|[0]\.[0-9]+", 'None', pexpect.TIMEOUT])
        if output is 0:
            score = float(self.proc.after.lstrip(smile + " "))
        elif output in [1, 2]:
            score = 0.0
        result_list[index] = score

    def is_alive(self):
        return self.proc.isalive()

class Multiprocessing():
    """Class for handling multiprocessing of scoring functions. OEtoolkits cant be used with
       native multiprocessing (cant be pickled), so instead we spawn threads that create
       subprocesses."""
    def __init__(self, num_processes=None, scoring_function=None):
        self.n = num_processes
        self.workers = [Worker(scoring_function=scoring_function) for _ in range(num_processes)]
        self.results = dict()

    def alive_workers(self):
        return [i for i, worker in enumerate(self.workers) if worker.is_alive()]

    def __call__(self, smiles):
        original_list = smiles
        smiles = list(set(smiles).difference(self.results.keys()))
        scores = [0 for _ in range(len(smiles))]
        smiles_copy = [smile for smile in smiles]
        while smiles_copy:
            alive_procs = self.alive_workers()
            if not alive_procs:
               raise RuntimeError("All subprocesses are dead, exiting.")
            # As long as we still have SMILES to score
            used_threads = []
            # Threads name corresponds to the index of the worker, so here
            # we are actually checking which workers are busy
            for t in threading.enumerate():
                # Workers have numbers as names, while the main thread cant
                # be converted to an integer
                try:
                    n = int(t.name)
                    used_threads.append(n)
                except ValueError:
                    continue
            free_threads = [i for i in alive_procs if i not in used_threads]
            for n in free_threads:
                if smiles_copy:
                    # Send SMILES and what index in the result list the score should be inserted at
                    smile = smiles_copy.pop()
                    idx = len(smiles_copy)
                    t = threading.Thread(target=self.workers[n], name=str(n), args=(smile, idx, scores))
                    t.start()
            time.sleep(0.01)
        for t in threading.enumerate():
            try:
                n = int(t.name)
                t.join()
            except ValueError:
                continue
        self.results = {**self.results, **dict(zip(smiles, scores))}
        return np.array([self.results[smile] for smile in original_list], dtype=np.float32)

class Singleprocessing():
    """Adds an option to not spawn new processes for the scoring functions, but rather
       run them in the main process."""
    def __init__(self, scoring_function=None):
        self.scoring_function = scoring_function()
    def __call__(self, smiles):
        scores = [self.scoring_function(smile) for smile in smiles]
        return np.array(scores, dtype=np.float32)

def get_scoring_function(scoring_function, decorate=True, num_processes=None, **kwargs):
    """Function that initializes and returns a scoring function by name"""
    scoring_function_classes = [no_sulphur, tanimoto, activity_model, docking_score]
    scoring_functions = [f.__name__ for f in scoring_function_classes]
    scoring_function_class = [f for f in scoring_function_classes if f.__name__ == scoring_function][0]

    if scoring_function not in scoring_functions:
        raise ValueError("Scoring function must be one of {}".format([f for f in scoring_functions]))

    if decorate:
        for k, v in kwargs.items():
            if k in scoring_function_class.kwargs:
                setattr(scoring_function_class, k, v)
        if num_processes == 0:
            return Singleprocessing(scoring_function=scoring_function_class)
        return Multiprocessing(scoring_function=scoring_function, num_processes=num_processes)
    return scoring_function_class(**kwargs)
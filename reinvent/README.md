## REINVENT
Environment: freedpp
```
python main.py \
    --batch-size 32 \
    --num-steps 3000 \
    --save-dir 'experiments/reinvent/' \
    --seed 150
    --sigma' 60
    --scoring-function docking_score \
    --scoring-function-kwargs --scoring-function-kwargs "exhaustiveness 1 n_conf 1 num_modes 10 error_val 99.9 alpha 0.1 timeout_dock 90 timeout_gen3d 30 receptor_file '/mnt/tank/scratch/avepreva/molecule_generation/code_submission/reinvent/COX-2.pdbqt' box_center (27.116,24.090,14.936) box_size (9.427,10.664,10.533) vina_program '/mnt/tank/scratch/avepreva/molecule_generation/code_submission/reinvent/qvina02' num_sub_proc 12 seed 150 temp_dir 'experiments/reinevent/tmp'"
```

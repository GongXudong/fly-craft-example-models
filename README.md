# Models trained by [various RL algorithms](https://github.com/GongXudong/fly-craft-examples) on [FlyCraft](https://github.com/GongXudong/fly-craft)

Algorithms listed in [fly-craft-examples](https://github.com/GongXudong/fly-craft-examples) are utilized to train these models. The directory includes:

* configs: the configurations utilized to train to get these models.
* checkpoints: direction of the trained models.
* checkpoints/bc: direction of the models trained by Behaviral Cloning.
* checkpoints/rl_single: direction of the models trained by off-policy RL (SAC, HER), on-policy RL (PPO), and NMR. In addition, this directory also includes all the models trained in the ablation studies in our FlyCraft paper.
* checkpoints/rl: direction of the models that fine-tuning a BC pre-trained model with PPO ([IRPO](https://github.com/GongXudong/IRPO)).

## BC

Train models with:

```bash
python train_scripts/train_with_bc_ppo.py --config-file-name configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

This willl produced a model named "10hz_128_128_300epochs_easy_1/bc_checkpoint.zip" in the folder "checkpoints/bc", the model name and save path are defined in the configuration file "configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json".

## PPO

Train models with:

```bash
python train_scripts/train_with_rl_ppo.py --config-file-name configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

This willl produced a model named "10hz_128_128_2e8steps_easy_1_singleRL/best_model.zip" in the folder "checkpoints/rl_single", the model name and save path are defined in the configuration file "configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json".

## SAC

Train models with:

```bash
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/sac_without_her/sac_config_10hz_128_128_1.json
```

This willl produced a model named "sac_without_her_10hz_128_128_1e6steps_loss_1_singleRL/best_model.zip" in the folder "checkpoints/rl_single", the model name and save path are defined in the configuration file "configs/train/sac/sac_without_her/sac_config_10hz_128_128_1.json".

## HER

Train models with:

```bash
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/sac_her/sac_config_10hz_128_128_1.json
```

This willl produced a model named "sac_her_10hz_128_128_1e6steps_loss_1_singleRL/best_model.zip" in the folder "checkpoints/rl_single", the model name and save path are defined in the configuration file "configs/train/sac/sac_her/sac_config_10hz_128_128_1.json".

### NMR

Train models with:

```bash
# test SAC on NMR(last 10 observations)
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/easy_her_sparse_negative_non_markov_reward_persist_1_sec/sac_config_10hz_128_128_1.json
```

This willl produced a model named "sac_her_sparse_negative_non_markov_reward_persist_1_sec_10hz_128_128_1e6steps_loss_1_singleRL/best_model.zip" in the folder "checkpoints/rl_single", the model name and save path are defined in the configuration file "configs/train/sac/easy_her_sparse_negative_non_markov_reward_persist_1_sec/sac_config_10hz_128_128_1.json".

The other settings of NMR models can be trained with the following:

```bash
# test SAC on NMR(last 20 observations)
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/easy_her_sparse_negative_non_markov_reward_persist_2_sec/sac_config_10hz_128_128_1.json

# test SAC on NMR(last 30 observations)
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/easy_her_sparse_negative_non_markov_reward_persist_3_sec/sac_config_10hz_128_128_1.json

# try solve NMR with framestack
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/hard_her_framestack_sparse_negative_non_markov_reward_persist_1_sec/sac_config_10hz_128_128_1.json
```

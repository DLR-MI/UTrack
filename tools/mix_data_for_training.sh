# training on CrowdHuman and MOT17 half train, evaluate on MOT17 half val.
python tools/mix_data_ablation.py

# training on CrowdHuman and MOT20 half train, evaluate on MOT20 half val.
python tools/mix_data_ablation_20.py

# training on MOT17, CrowdHuman, ETHZ, Citypersons, evaluate on MOT17 train.
python tools/mix_data_test_mot17.py

# training on MOT20 and CrowdHuman, evaluate on MOT20 train.
python tools/mix_data_test_mot20.py

# training on dancetrack
python tools/mix_data_dancetrack.py

# reset ultralytics default datasets path
python tools/reset_datasets_path.py
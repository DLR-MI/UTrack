
# UTrack

[![arXiv](https://img.shields.io/badge/arXiv-2408.17098-<COLOR>.svg)](https://arxiv.org/abs/2408.17098) [![License: MIT](https://img.shields.io/badge/License-AGPLv3-yellow.svg)](https://www.gnu.org/licenses/agpl-3.0) ![test](https://img.shields.io/static/v1?label=By&message=Pytorch&color=red)

> [**UTrack: Multi-Object Tracking with Uncertain Detections**](https://arxiv.org/abs/2408.17098)
> 
> Edgardo Solano-Carrillo, Felix Sattler, Antje Alex, Alexander Klein, Bruno Pereira Costa, Angel Bueno Rodriguez, Jannis Stoppe


## Abstract 

The tracking-by-detection paradigm is the mainstream in
multi-object tracking, associating tracks to the predictions of an object
detector. Although exhibiting uncertainty through a confidence score,
these predictions do not capture the entire variability of the inference
process. For safety and security critical applications like autonomous
driving, surveillance, etc., knowing this predictive uncertainty is essential though.
Therefore, we introduce, for the first time, a fast way to
obtain the empirical predictive distribution during object detection and
incorporate that knowledge in multi-object tracking. Our mechanism can
easily be integrated into state-of-the-art trackers, enabling them to fully
exploit the uncertainty in the detections. Additionally, novel association
methods are introduced that leverage the proposed mechanism. We
demonstrate the effectiveness of our contribution on a variety of benchmarks,
such as MOT17, MOT20, DanceTrack, and KITTI.


## Highlights 🚀

- YOLOv8 support using a fork of [Ultralytics](https://github.com/DLR-MI/ultralytics/tree/nms-var)
- Non maximum suppresion (NMS) with box variances using [NMS-var](https://github.com/DLR-MI/nms_var)
- Fast camera motion compensation (affine & homography) using [FastGMC](https://github.com/DLR-MI/fast_gmc)
- Estimates of errors in IoU using [fuzzy cython-bbox](https://github.com/DLR-MI/fuzzy_cython_bbox)
- Visualize metrics using [Dash](https://dash.plotly.com/)
- Several integrated trackers
  - [x] ByteTrack
  - [x] BoT-SORT
  - [x] SparseTrack
  - [x] UTrack 
  - [ ] UCMCTrack (experimental)
  - [ ] KCM-Track (experimental)

## Installation
First clone this repository:
```shell
git clone https://github.com/DLR-MI/UTrack
cd UTrack
```
and install suitable dependencies. This was tested on conda environment with Python 3.8.
```shell
mamba create -n track python=3.8
mamba activate track
mamba install pytorch torchvision
pip install cython
pip install -r requirements.txt
```


## Datasets
Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) and put them in a `/data` folder. Prepare the datasets for training MOT17 and MOT20 for ablation and testing:

```shell
bash tools/convert_datasets_to_coco.sh
bash tools/mix_data_for_training.sh
```

If you use another path for the data, make sure to change the `DATA_PATH` in the scripts invoked above. To clip the annotated bounding boxes to the image, run (e.g. for MOT17)

```shell
python tools/fix_yolo_annotations.py --folder /data/MOT17
```

For [DanceTrack](https://dancetrack.github.io/) and [KITTI](https://www.cvlibs.net/datasets/kitti/eval_tracking.php), you can also find scripts in `./tools` to convert to the COCO format. No mixing is necessary here. 

> **Note:** make sure that the folder structure of KITTI, after converting to COCO, mimics the one for MOTXX.

## Training

As an example, training `YOLOv8-l` for the ablation experiments of MOT17 is done by running

```shell
$ python train.py --model yolov8l --exp ablation_17 --gpu_id 0
```
Take a look at `train.py` for the available `mot_choices` for the `--exp` argument.

The model weights used for the experiments in the paper can be downloaded below.

| Dataset | Weights|
|-----| ------|
| Ablation 17 |  [[download]](https://zenodo.org/records/13604403/files/ablation_17_best.pt?download=1) |
| Ablation 20 |  [[download]](https://zenodo.org/records/13604403/files/ablation_20_best.pt?download=1) |
| Mix 17 |  [[download]](https://zenodo.org/records/13604403/files/mix_17_best.pt?download=1) |
| Mix 20 |  [[download]](https://zenodo.org/records/13604403/files/mix_20_best.pt?download=1) |
| DanceTrack |  [[download]](https://zenodo.org/records/13604403/files/dancetrack_best.pt?download=1) |
| KITTI |  [[download]](https://zenodo.org/records/13604403/files/kitti_best.pt?download=1) |

After downloading, rename to `best.pt` and place it within the corresponding folder in `./yolov8l-mix/EXP/weights`, where `EXP` is the experiment name referred to during training (i.e. `mix_17`, `ablation_17`, etc.) Note that KITTI was only trained for the pedestrian class.

## Evaluation
For tracking, you just need to modify `./tracker/config/track_EXP.yaml` if considering different tracker parameters, where `EXP` is the experiment name. 

```shell
$ python track.py --project yolov8l-mix --exp ablation_17 --data_root /data/MOT17 --association uk_botsort
```

This produces an entry in the folder `./track_results` with relevant metrics. The available trackers can be explored
by referring to the `./tracker/associations/collections.py` module.

You can do hyperparameter search of multiple trackers in parallel, or run multiple seeds for a single tracker. The former is done by executing `./hp_search.py` on a `.yaml` template in `./tracker/eval/hp_search`. The latter, by using
the `--association` argument.


## Results on DanceTrack test set

Just by adding (on top of BoT-SORT) the right observation noise in the Kalman filter:


|  Method  | HOTA | DetA | AssA | MOTA | IDF1 |
|------------|-------|-------|------|------|-------|
| BoT-SORT | 53.8 | 77.8 | 37.3 | 89.7 | 56.1 |
| UTrack | 55.8 | 79.4 | 39.3 | 89.7 | 56.4 |

The evaluation runs at **27 FPS** on a NVIDIA A100 GPU, compared with the 32 FPS of BoT-SORT in the same machine.

### Demo

A demo can be run on a sample video from [Youtube](https://www.youtube.com/watch?v=qv6gl4h0dvg) by executing

```shell
python ./tools/track_demo.py --exp dancetrack --association uk_botsort --video_path /path/to/video --output_path /path/to/output/video
```

<img src="assets/dance_sample_track.gif" width="400"/>


## Citation
If you find this work useful, it would be cool to know it by giving us a star 🌟. Also, consider citing it as


```bibtex
@misc{UTrack,
      title={UTrack: Multi-Object Tracking with Uncertain Detections}, 
      author={Edgardo Solano-Carrillo and Felix Sattler and Antje Alex and Alexander Klein and Bruno Pereira Costa and Angel Bueno Rodriguez and Jannis Stoppe},
      year={2024},
      eprint={2408.17098},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.17098}, 
}
```

## Acknowledgements
A portion of the code is borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BOT-SORT), [SparseTrack](https://github.com/hustvl/SparseTrack/tree/main), [UCMCTrack](https://github.com/corfyi/UCMCTrack), and [TrackEval](https://github.com/JonathonLuiten/TrackEval). 
Many thanks for their contributions.

Also thanks to [Ultralytics](https://github.com/ultralytics/ultralytics) for making object detection more user friendly. 






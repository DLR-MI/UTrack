import argparse
from ultralytics import YOLO

def make_parser():
    parser = argparse.ArgumentParser("Tracker")
    mot_choices = ['mix_17', 'mix_20', 'ablation_17', 'ablation_20', 'dancetrack', 'kitti']
    yolov8_choices = [f'yolov8{x}' for x in ('n', 's', 'm', 'l', 'x')] 
    parser.add_argument("--model", type=str, default='yolov8l', choices=yolov8_choices, help="yolov8 pre-trained model") 
    parser.add_argument("--exp", type=str, default='ablation_17', choices=mot_choices, help="experiment name")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu device index")
    return parser.parse_args()

if __name__ == '__main__':
    args = make_parser()
    # Load a model
    model = YOLO(f'{args.model}.pt') 
    # Train
    results = model.train(
        project=f'{args.model}-mix',
        name=args.exp,
        data=f'datasets/mix/{args.exp}/train_config.yaml', 
        imgsz=(800, 1440),
        rect=True,
        batch=24,
        epochs=100, 
        # device=[0,1,2,3], ## rect=True incompatible with multi-GPU?
        device=[args.gpu_id], 
        verbose=True
    )
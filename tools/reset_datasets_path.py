import os
from pathlib import Path
from omegaconf import OmegaConf

config_path = f'/home/{os.getlogin()}/.config/Ultralytics/settings.yaml'
settings = OmegaConf.load(config_path)
datasets_dir = Path(settings.datasets_dir)
if datasets_dir.name == 'datasets':
    settings.datasets_dir = str(datasets_dir.parent)
OmegaConf.save(config=settings, f=config_path)
"""Predict direction from 20 chronological closes and a trained checkpoint."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torchvision.transforms import Normalize
from src.charts import render_chart
from src.model import ResNet18Classifier
from scripts.train_spy_resnet import MEAN, STD


def read_closes(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['Close']:
            raise ValueError('CSV must have exactly one column named Close')
        try:
            values = [float(row['Close']) for row in reader]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError('Close values must be numeric') from exc
    values = np.asarray(values, dtype=float)
    if values.shape != (20,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('CSV must contain exactly 20 finite positive Close values')
    return values


def predict(closes, checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError('Trained checkpoint not found. Provide --checkpoint PATH to a trained project checkpoint; see README for training instructions. Pretrained ImageNet weights alone are not a trained checkpoint.')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = saved['model_config']
    if config.get('architecture') != 'resnet18' or config.get('num_classes') != 2 or config.get('training_mode') not in ('classifier_only', 'full_finetune'):
        raise ValueError('Checkpoint does not describe a compatible ResNet-18 classifier')
    model = ResNet18Classifier(pretrained=False, training_mode=config['training_mode'])
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval()
    pixels = np.array(render_chart(closes), dtype=np.uint8, copy=True)
    image = torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32).div_(255.0)
    image = Normalize(MEAN, STD)(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(image), dim=1)[0]
    non_up, up = probabilities.tolist()
    return {'prediction': 'Up' if up > non_up else 'Non-up', 'probability_up': up, 'probability_non_up': non_up, 'training_mode': config['training_mode']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='CSV with one Close column and 20 values, oldest first')
    parser.add_argument('--checkpoint', required=True, type=Path, help='trained project checkpoint')
    args = parser.parse_args()
    try:
        print(json.dumps(predict(read_closes(args.input), args.checkpoint), indent=2))
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')

if __name__ == '__main__':
    main()

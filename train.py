import torch
import argparse, os, pickle
from torch.utils.tensorboard import SummaryWriter
from data import ImageDetectionsField, TextField

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Partial Non-Autoregressive Image Captioning')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='PNAIC')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--feature_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    args = parser.parse_args()
    print(args)

    print('Partial Non-Autoregressive Image Captioning training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Read image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Read text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
























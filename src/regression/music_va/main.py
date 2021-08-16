import os
import torch
import argparse
from solver import Solver
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def main(config):
    solver = Solver(config)
    checkpoint_callback = ModelCheckpoint(filepath=config.model_save_path,
                                          save_top_k=1,
                                          verbose=True,
                                          monitor="monitor",
                                          mode="max",
                                          prefix="")
    trainer = Trainer(default_root_dir=config.model_save_path,
                      gpus=config.gpu_id,
                      checkpoint_callback=checkpoint_callback,
                      max_epochs=config.n_epochs)
    if config.mode == 'TRAIN':
        solver.load_pretrained('./data/mtat.ckpt')
        trainer.fit(solver)
        trainer.save_checkpoint(os.path.join(config.model_save_path, 'last.ckpt'))
    elif config.mode == 'TEST':
        solver.load_pretrained(config.model_load_path)
        trainer.test(solver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='TRAIN', choices=['TRAIN', 'TEST'])

    # model parameters
    parser.add_argument('--ndim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_chunk', type=int, default=16)
    parser.add_argument('--input_length', type=int, default=80000)

    # stft parameters
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--win_length', type=int, default=512)
    parser.add_argument('--n_bins', type=int, default=128)
    parser.add_argument('--output_type', type=str, default='spec', choices=['raw', 'spec', 'cqt'])

    # augmentation parameters
    parser.add_argument('--is_gain', type=bool, default=True)
    parser.add_argument('--gain_db_min', type=float, default=-20.0)
    parser.add_argument('--gain_db_max', type=float, default=0.0)

    parser.add_argument('--is_noise', type=bool, default=True)
    parser.add_argument('--noise_snr_min', type=float, default=40.0)
    parser.add_argument('--noise_snr_max', type=float, default=80.0)

    parser.add_argument('--is_pitch_shift', type=bool, default=False)
    parser.add_argument('--pitch_shift_margin', type=int, default=10)

    parser.add_argument('--is_noise2d', type=bool, default=False)
    parser.add_argument('--noise2d_ratio', type=float, default=0.0)

    parser.add_argument('--is_time_mask', type=bool, default=True)
    parser.add_argument('--time_mask_ratio', type=float, default=0.4)

    parser.add_argument('--is_freq_mask', type=bool, default=True)
    parser.add_argument('--freq_mask_ratio', type=int, default=0.4)

    # training parameters
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default='.')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints')
    parser.add_argument('--model_load_path', type=str, default='.')

    config = parser.parse_args()

    print(config)
    main(config)

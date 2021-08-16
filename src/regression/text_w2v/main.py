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
                                          mode="min",
                                          prefix="")
    trainer = Trainer(default_root_dir=config.model_save_path,
                      gpus=config.gpu_id,
                      checkpoint_callback=checkpoint_callback,
                      max_epochs=config.n_epochs)
    if config.mode == 'TRAIN':
        trainer.fit(solver)
        trainer.save_checkpoint(os.path.join(config.model_save_path, 'last.ckpt'))
    elif config.mode == 'TEST':
        solver.load_pretrained(config.model_load_path)
        trainer.test(solver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='TRAIN', choices=['TRAIN', 'TEST'])
    parser.add_argument('--dataset', type=str, default='alm', choices=['isear', 'alm'])

    # model parameters
    parser.add_argument('--ndim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)

    # training parameters
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_save_path', type=str, default='./checkpoints')
    parser.add_argument('--model_load_path', type=str, default='.')

    config = parser.parse_args()

    print(config)
    main(config)

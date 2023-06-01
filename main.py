# This is the implementation of Yicun Duan (20216268)'s Final Year Project

import argparse

import discriminator
import generator
import train
import os
import torch

parser = argparse.ArgumentParser("train complex-valued Restormer on VoiceBank+DEMAND dataset")
parser.add_argument('--json_dir', type=str, default=os.path.join(".", "dataset", "json"),
                    help='The directory of the dataset feat,json format')
parser.add_argument('--batch_size', type=int, default=4,
                    help='The number of the batch size')
parser.add_argument('--test_batch_size', type=int, default=2,
                    help='The number of the batch size')
parser.add_argument('--epochs', type=int, default=80,
                    help='The number of the training epoch')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Learning rate of the network')
parser.add_argument('--l2', type=float, default=1e-7,
                    help='weight decay (L2 penalty)')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of workers to generate batch')
parser.add_argument('--print_freq', type=int, default=10,
                    help='The frequency of printing loss information')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--half_lr', type=int, default=1,
                    help='Whether to decay learning rate to half scale')

if __name__ == "__main__":
    args = parser.parse_args()

    model = generator.ComplexRestormer()  # type: torch.nn.Module

    discriminator_mpd = discriminator.MultiPeriodDiscriminator()  # type: torch.nn.Module
    discriminator_msd = discriminator.MultiScaleDiscriminator()  # type: torch.nn.Module
    discriminator_pesq = discriminator.MetricDiscriminator()  # type: torch.nn.Module

    print("+" * 150)
    print(args)
    print("+" * 150)

    train.train(args, model,
                discriminator_mpd=discriminator_mpd, discriminator_msd=discriminator_msd,
                discriminator_pesq=discriminator_pesq)

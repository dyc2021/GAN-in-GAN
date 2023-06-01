import torch
import dataset
import train_solver
import itertools


def train(args, model: torch.nn.Module,
          discriminator_mpd: torch.nn.Module, discriminator_msd: torch.nn.Module,
          discriminator_pesq: torch.nn.Module):
    train_dataset = dataset.TrainDataset(json_dir=args.json_dir,
                                         batch_size=args.batch_size)
    test_dataset = dataset.TestDataset(json_dir=args.json_dir,
                                       batch_size=args.test_batch_size)
    train_loader = dataset.TrainDataLoader(data_set=train_dataset,
                                           batch_size=1,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
    test_loader = dataset.TestDataLoader(data_set=test_dataset,
                                         batch_size=1,
                                         num_workers=args.num_workers,
                                         pin_memory=True)
    data = {'train_loader': train_loader, 'test_loader': test_loader}

    generator_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                           lr=args.lr,
                                           weight_decay=args.l2)
    discriminator_mpd_msd_optimizer = torch.optim.AdamW(itertools.chain(discriminator_mpd.parameters(),
                                                                        discriminator_msd.parameters()),
                                                        lr=0.0002,
                                                        betas=(0.8, 0.99))
    discriminator_mpd_msd_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_mpd_msd_optimizer,
                                                                             gamma=0.999)
    discriminator_pesq_optimizer = torch.optim.AdamW(discriminator_pesq.parameters(),
                                                     lr=0.0002,
                                                     betas=(0.8, 0.99))
    discriminator_pesq_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_pesq_optimizer,
                                                                          gamma=0.999)

    solver = train_solver.TrainSolver(data, model, generator_optimizer,
                                      discriminator_mpd=discriminator_mpd,
                                      discriminator_msd=discriminator_msd,
                                      discriminator_mpd_msd_optimizer=discriminator_mpd_msd_optimizer,
                                      discriminator_mpd_msd_scheduler=discriminator_mpd_msd_scheduler,
                                      discriminator_pesq=discriminator_pesq,
                                      discriminator_pesq_optimizer=discriminator_pesq_optimizer,
                                      discriminator_pesq_scheduler=discriminator_pesq_scheduler,
                                      args=args)
    solver.run()

import torch
import dataset
import train_solver


def train(args, model):
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

    optimizer = torch.optim.Adam(model.parameters(),
                                 args.lr,
                                 weight_decay=args.l2)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=[int(args.epochs / 4), 2 * int(args.epochs / 4),
    #                                                              3 * int(args.epochs / 4)],
    #                                                  gamma=0.1)

    solver = train_solver.Solver(data, model, optimizer, args)
    solver.run()

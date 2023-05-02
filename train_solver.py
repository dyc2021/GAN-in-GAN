import torch
import time
import gc
from contextlib import contextmanager


def com_mag_mse_loss(esti_complex, label_complex, frame_list):
    esti = torch.stack((esti_complex.real, esti_complex.imag), dim=1).to(esti_complex.device)
    label = torch.stack((label_complex.real, label_complex.imag), dim=1).to(label_complex.device)

    # print("esti: {}".format(esti.shape))
    # print("label: {}".format(label.shape))

    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = torch.nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    esti_mag, label_mag = torch.norm(esti, dim=1), torch.norm(label, dim=1)
    loss1 = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    loss2 = (((esti_mag - label_mag) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()
    return 0.5 * (loss1 + loss2)


class Solver(object):
    def __init__(self, data, model, optimizer, args):
        # load args parameters
        self.tr_loader = data['train_loader']
        self.cv_loader = data['test_loader']
        self.model = model.cuda()
        self.optimizer = optimizer
        self.epochs = args.epochs
        self.print_freq = args.print_freq

        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.prev_cv_loss = float("inf")
        self.cv_no_impv = 0
        self.having = False

    def run(self):
        for epoch in range(self.epochs):
            print("=" * 75)
            print("Epoch {} begins".format(epoch))

            # train
            print("Training:")
            self.model.train()
            start = time.time()
            tr_avg_loss = self.run_one_epoch(epoch)
            print("End of Training, Time: {:.4f} s, Train Loss: {:.5f}".format(time.time() - start, tr_avg_loss))

            # test
            print("Testing:")
            self.model.eval()
            cv_avg_loss = self.run_one_epoch(epoch, cross_valid=True)

            # adjust learning rate and early stop
            if self.half_lr:
                if cv_avg_loss >= self.prev_cv_loss:
                    self.cv_no_impv += 1
                    if self.cv_no_impv == 3:
                        self.having = True
                    if self.cv_no_impv >= 5 and self.early_stop == True:
                        print("No improvement and apply early stop")
                        break
                else:
                    self.cv_no_impv = 0

            if self.having:
                optim_state = self.optimizer.state_dict()
                for i in range(len(optim_state['param_groups'])):
                    optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print("Learning rate adjusted to {:.5f}".format((optim_state['param_groups'][0]['lr'])))
                self.having = False

            self.prev_cv_loss = cv_avg_loss

            print("End of Testing, Time: {:.4f} s, Test Loss: {:.5f}".format(time.time() - start, cv_avg_loss))
            print("=" * 75)

    def run_one_epoch(self, epoch, cross_valid=False):
        def _batch(_, batch_info):
            batch_feat = batch_info.feats.cuda()
            batch_label = batch_info.labels.cuda()
            noisy_phase = torch.atan2(batch_feat.imag, batch_feat.real)
            clean_phase = torch.atan2(batch_label.imag, batch_label.real)
            batch_frame_mask_list = batch_info.frame_mask_list

            # feature compression:
            batch_feat_mag_compressed, batch_label_mag_compressed = \
                torch.sqrt(batch_feat.imag ** 2 + batch_feat.real ** 2) ** 0.5, \
                torch.sqrt(batch_label.imag ** 2 + batch_label.real ** 2) ** 0.5

            batch_feat_compressed = torch.complex(batch_feat_mag_compressed * torch.cos(noisy_phase),
                                                  batch_feat_mag_compressed * torch.sin(noisy_phase))
            batch_label_compressed = torch.complex(batch_label_mag_compressed * torch.cos(clean_phase),
                                                   batch_label_mag_compressed * torch.sin(clean_phase))

            esti_compressed = self.model(batch_feat_compressed)

            # print("batch_feat: {}".format(batch_feat_compressed.shape))
            # print("batch_label: {}".format(batch_label_compressed.shape))
            # print("esti_compressed: {}".format(esti_compressed.shape))

            if not cross_valid:
                batch_loss = com_mag_mse_loss(esti_compressed,
                                              batch_label_compressed, batch_frame_mask_list)
                batch_loss_res = batch_loss.item()

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            else:
                batch_loss = com_mag_mse_loss(esti_compressed,
                                              batch_label_compressed, batch_frame_mask_list)
                batch_loss_res = batch_loss.item()

            return batch_loss_res

        start1 = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        total_iter = len(data_loader.get_data_loader())
        batch_id = 0
        for batch_id, batch_info in enumerate(data_loader.get_data_loader()):
            batch_loss_res = _batch(batch_id, batch_info)
            total_loss += batch_loss_res
            gc.collect()
            if batch_id % self.print_freq == 0:
                print("Epoch: {}, Iteration: [{}/{}], average loss: {:.5f}, "
                      "current loss: {:.5f}, {:.4f} ms/batch".format(
                    epoch, batch_id, total_iter, total_loss / (batch_id + 1), batch_loss_res,
                                     1000 * (time.time() - start1) / (batch_id + 1)))
        return total_loss / (batch_id + 1)


@contextmanager
def set_default_tensor_type(tensor_type):
    if torch.tensor(0).is_cuda:
        old_tensor_type = torch.cuda.FloatTensor
    else:
        old_tensor_type = torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(old_tensor_type)

import torch
import time
import gc
from contextlib import contextmanager
import torch.nn.functional as F
import pesq as pesq_lib
from pesq.cypesq import PesqError
from joblib import Parallel, delayed


# def com_mag_mse_loss(esti_complex_spectro, label_complex_spectro, frame_list):
#     esti_complex_spectro = esti_complex_spectro.squeeze(1)
#     label_complex_spectro = label_complex_spectro.squeeze(1)
#     esti = torch.stack((esti_complex_spectro.real, esti_complex_spectro.imag), dim=1).to(esti_complex_spectro.device)
#     label = torch.stack((label_complex_spectro.real, label_complex_spectro.imag), dim=1).to(
#         label_complex_spectro.device)
#
#     mask_for_loss = []
#     utt_num = esti.size()[0]
#     with torch.no_grad():
#         for i in range(utt_num):
#             tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
#             mask_for_loss.append(tmp_mask)
#         mask_for_loss = torch.nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
#         com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
#     esti_mag, label_mag = torch.norm(esti, dim=1), torch.norm(label, dim=1)
#     loss1 = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
#     loss2 = (((esti_mag - label_mag) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()
#     return 0.5 * (loss1 + loss2)


def multi_resolution_discriminator_loss(real_outputs, generated_outputs):
    loss = 0
    for real_output, generated_output in zip(real_outputs, generated_outputs):
        r_loss = torch.mean((1 - real_output) ** 2)
        g_loss = torch.mean(generated_output ** 2)
        loss += (r_loss + g_loss)

    return loss


def multi_resolution_generator_loss(discriminator_outputs):
    loss = 0
    for discriminator_output in discriminator_outputs:
        loss += torch.mean((1 - discriminator_output) ** 2)

    return loss


def multi_resolution_feature_maps_loss(real_outputs, generated_outputs):
    loss = 0
    for real_output, generated_output in zip(real_outputs, generated_outputs):
        for real_slice, generated_slice in zip(real_output, generated_output):
            loss += torch.mean(torch.abs(real_slice - generated_slice))

    return loss


@torch.no_grad()
def calculate_pesq(input, target):
    pesq_value = pesq_lib.pesq(16000, target, input, "wb", PesqError.RETURN_VALUES)
    if pesq_value != -1:
        return pesq_value
    else:
        return 0.0


class TrainSolver(object):
    def __init__(self,
                 data,
                 model,
                 generator_optimizer,
                 discriminator_mpd,
                 discriminator_msd,
                 discriminator_mpd_msd_optimizer,
                 discriminator_mpd_msd_scheduler,
                 discriminator_pesq,
                 discriminator_pesq_optimizer,
                 discriminator_pesq_scheduler,
                 args):
        # load args parameters
        self.train_loader = data['train_loader']
        self.test_loader = data['test_loader']
        self.epochs = args.epochs
        self.print_freq = args.print_freq
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop

        self.prev_cv_loss = float("inf")
        self.test_no_improve = 0
        self.having = False

        self.model = model.cuda()
        self.generator_optimizer = generator_optimizer
        self.mpd = discriminator_mpd.cuda()
        self.msd = discriminator_msd.cuda()
        self.mpd_msd_optimizer = discriminator_mpd_msd_optimizer
        self.mpd_msd_scheduler = discriminator_mpd_msd_scheduler
        self.pesq = discriminator_pesq.cuda()
        self.pesq_optimizer = discriminator_pesq_optimizer
        self.pesq_scheduler = discriminator_pesq_scheduler

        self.pesq_estimator = calculate_pesq

    def run(self):
        for epoch in range(self.epochs):
            print("=" * 75)
            print("Epoch {} begins".format(epoch))

            epoch_start_time = time.time()

            # train
            print("Training:")
            self.model.train()
            self.mpd.train()
            self.msd.train()
            self.pesq.train()
            train_avg_loss = self.run_one_epoch(epoch, test=False)
            print("End of Training, Time: {:.4f} s, Train Loss: {:.5f}".format(time.time() - epoch_start_time,
                                                                               train_avg_loss))

            # test
            print("Testing:")
            self.model.eval()
            self.mpd.eval()
            self.msd.eval()
            self.pesq.eval()
            test_avg_loss = self.run_one_epoch(epoch, test=True)

            # adjust learning rate and early stop
            self.mpd_msd_scheduler.step()
            self.pesq_scheduler.step()

            if self.half_lr:
                if test_avg_loss >= self.prev_cv_loss:
                    self.test_no_improve += 1
                    if self.test_no_improve == 3:
                        self.having = True
                    if self.test_no_improve >= 5 and self.early_stop:
                        print("No improvement and apply early stop")
                        break
                else:
                    self.test_no_improve = 0

            if self.having:
                optim_state = self.generator_optimizer.state_dict()
                for i in range(len(optim_state['param_groups'])):
                    optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                self.generator_optimizer.load_state_dict(optim_state)
                print("Learning rate adjusted to {:.5f}".format((optim_state['param_groups'][0]['lr'])))
                self.having = False

            self.prev_cv_loss = test_avg_loss

            print("End of Testing, Time: {:.4f} s, Test Loss: {:.5f}".format(time.time() - epoch_start_time,
                                                                             test_avg_loss))
            print("=" * 75)

    def run_one_epoch(self, epoch, test=False):
        def _batch(batch_info):
            batch_feat = batch_info.feats.cuda()
            batch_label = batch_info.labels.cuda()
            batch_frame_mask_list = batch_info.frame_mask_list

            # feature compression
            # noisy_phase = torch.atan2(batch_feat.imag, batch_feat.real)
            # clean_phase = torch.atan2(batch_label.imag, batch_label.real)
            # batch_feat_mag_compressed, batch_label_mag_compressed = \
            #     torch.sqrt(batch_feat.imag ** 2 + batch_feat.real ** 2) ** 0.5, \
            #     torch.sqrt(batch_label.imag ** 2 + batch_label.real ** 2) ** 0.5
            #
            # batch_feat_compressed = torch.complex(batch_feat_mag_compressed * torch.cos(noisy_phase),
            #                                       batch_feat_mag_compressed * torch.sin(noisy_phase))
            # batch_label_compressed = torch.complex(batch_label_mag_compressed * torch.cos(clean_phase),
            #                                        batch_label_mag_compressed * torch.sin(clean_phase))

            if not test:
                batch_feat_compressed_spectro, esti_compressed_spectro, esti = self.model(batch_feat)
                _, _, batch_label_compressed_spectro = self.model.compress(batch_label)
                batch_label_compressed_spectro_mag = torch.sqrt(batch_label_compressed_spectro.real ** 2 +
                                                                batch_label_compressed_spectro.imag ** 2)
                esti_compressed_spectro_mag = torch.sqrt(esti_compressed_spectro.real ** 2 +
                                                         esti_compressed_spectro.imag ** 2)
                batch_feat_compressed_spectro_mag = torch.sqrt(batch_feat_compressed_spectro.real ** 2 +
                                                               batch_feat_compressed_spectro.imag ** 2)

                # train discriminator pesq
                self.pesq_optimizer.zero_grad()
                # clean-clean
                clean_clean_pair = torch.cat((batch_label_compressed_spectro_mag, batch_label_compressed_spectro_mag),
                                             dim=1)
                clean_clean_score_in_discriminator = self.pesq(clean_clean_pair)
                clean_clean_loss = F.mse_loss(clean_clean_score_in_discriminator,
                                              torch.ones(clean_clean_score_in_discriminator.shape).cuda())
                # esti-clean
                esti_clean_pair = torch.cat((esti_compressed_spectro_mag.detach(), batch_label_compressed_spectro_mag),
                                            dim=1)
                esti_clean_score_in_discriminator = self.pesq(esti_clean_pair)
                esti_clean_score_in_estimator = self.compute_metric(inputs=esti.detach(), targets=batch_label,
                                                                    metric="pesq")
                esti_clean_loss = F.mse_loss(esti_clean_score_in_discriminator, esti_clean_score_in_estimator)
                # noisy-clean
                noisy_clean_pair = torch.cat((batch_feat_compressed_spectro_mag, batch_label_compressed_spectro_mag),
                                             dim=1)
                noisy_clean_score_in_discriminator = self.pesq(noisy_clean_pair)
                noisy_clean_score_in_estimator = self.compute_metric(inputs=batch_feat, targets=batch_label,
                                                                     metric="pesq")
                noisy_clean_loss = F.mse_loss(noisy_clean_score_in_discriminator, noisy_clean_score_in_estimator)

                pesq_loss = (clean_clean_loss + esti_clean_loss + noisy_clean_loss) / 3.0
                pesq_loss_item = pesq_loss.item()
                pesq_loss.backward()
                self.pesq_optimizer.step()

                # train discriminator mpd and msd
                self.mpd_msd_optimizer.zero_grad()

                mpd_real, _ = self.mpd(batch_label.unsqueeze(1))
                mpd_fake, _ = self.mpd(esti.detach().unsqueeze(1))
                msd_real, _ = self.msd(batch_label.unsqueeze(1))
                msd_fake, _ = self.msd(esti.detach().unsqueeze(1))

                mpd_msd_loss = multi_resolution_discriminator_loss(real_outputs=mpd_real, generated_outputs=mpd_fake)
                mpd_msd_loss += multi_resolution_discriminator_loss(real_outputs=msd_real, generated_outputs=msd_fake)

                mpd_msd_loss.backward()
                mpd_msd_loss_item = mpd_msd_loss.item()
                self.mpd_msd_optimizer.step()

                # train generator
                self.generator_optimizer.zero_grad()

                esti_clean_pair = torch.cat((esti_compressed_spectro_mag, batch_label_compressed_spectro_mag), dim=1)
                esti_clean_score_in_generator = self.pesq(esti_clean_pair)
                mpd_real_value, mpd_real_feature_maps = self.mpd(batch_label.unsqueeze(1))
                mpd_fake_value, mpd_fake_feature_maps = self.mpd(esti.unsqueeze(1))
                msd_real_value, msd_real_feature_maps = self.msd(batch_label.unsqueeze(1))
                msd_fake_value, msd_fake_feature_maps = self.msd(esti.unsqueeze(1))

                generator_pesq_loss = F.mse_loss(esti_clean_score_in_generator,
                                                 torch.ones(esti_clean_score_in_generator.shape).cuda())
                generator_feature_map_loss = multi_resolution_feature_maps_loss(real_outputs=mpd_real_feature_maps,
                                                                                generated_outputs=mpd_fake_feature_maps)
                generator_feature_map_loss += multi_resolution_feature_maps_loss(real_outputs=msd_real_feature_maps,
                                                                                 generated_outputs=msd_fake_feature_maps)

                generator_multi_resolution_loss = multi_resolution_generator_loss(mpd_fake_value)
                generator_multi_resolution_loss += multi_resolution_generator_loss(msd_fake_value)

                generator_abs_loss = F.l1_loss(esti, batch_label)

                generator_loss = generator_feature_map_loss * 2 + \
                                 generator_multi_resolution_loss + \
                                 generator_abs_loss * 45 + \
                                 generator_pesq_loss * 9
                generator_loss_item = generator_loss.item()

                generator_loss.backward()
                self.generator_optimizer.step()
            else:
                # test generator
                batch_feat_compressed_spectro, esti_compressed_spectro, esti = self.model(batch_feat)
                _, _, batch_label_compressed_spectro = self.model.compress(batch_label)
                # you could self-define the testing here;
                # we don't provide detailed testing, since it's easy to implement
                generator_loss = F.l1_loss(esti, batch_label)
                generator_loss_item = generator_loss.item()

                mpd_msd_loss_item = 0
                pesq_loss_item = 0

            return generator_loss_item, mpd_msd_loss_item, pesq_loss_item

        batch_start_time = time.time()
        total_generator_loss = 0
        total_mpd_msd_loss = 0
        total_pesq_loss = 0
        data_loader = self.train_loader if not test else self.test_loader
        total_iter = len(data_loader.get_data_loader())
        batch_id = 0
        for batch_id, batch_info in enumerate(data_loader.get_data_loader()):
            batch_loss_item, mpd_msd_loss_item, pesq_loss_item = _batch(batch_info)
            total_generator_loss += batch_loss_item
            total_mpd_msd_loss += mpd_msd_loss_item
            total_pesq_loss += pesq_loss_item
            gc.collect()
            if batch_id % self.print_freq == 0:
                print("Epoch: {}, Iteration: [{}/{}], "
                      "average loss: {:.5f}, "
                      "current loss: {:.5f}, "
                      "mpd_msd average loss: {:.5f}, "
                      "mpd_msd current loss: {:.5f}, "
                      "pesq average loss: {:.5f}, "
                      "pesq current loss: {:.5f}, "
                      "{:.4f} ms/batch".format(
                    epoch, batch_id, total_iter,
                    total_generator_loss / (batch_id + 1),
                    batch_loss_item,
                    total_mpd_msd_loss / (batch_id + 1),
                    mpd_msd_loss_item,
                    total_pesq_loss / (batch_id + 1),
                    pesq_loss_item,
                    1000 * (time.time() - batch_start_time) / (batch_id + 1)))
        return total_generator_loss / (batch_id + 1)

    def compute_metric(self, inputs, targets, metric="pesq"):
        inputs_numpy = inputs.detach().cpu().numpy()
        targets_numpy = targets.detach().cpu().numpy()
        ori_device = inputs.device

        if metric == "pesq":
            metric_function = self.pesq_estimator
        else:
            raise NotImplemented("Metric {} is not supported".format(metric))

        if metric == "pesq":
            outputs_list = Parallel(n_jobs=4)(delayed(metric_function)(i, t)
                                              for i, t in zip(inputs_numpy, targets_numpy))
            outputs = torch.tensor(outputs_list).reshape((-1, 1))
            outputs = (outputs + 0.5) / 5.0

        outputs = outputs.to(device=ori_device)

        return outputs


@contextmanager
def set_default_tensor_type(tensor_type):
    if torch.tensor(0).is_cuda:
        old_tensor_type = torch.cuda.FloatTensor
    else:
        old_tensor_type = torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(old_tensor_type)

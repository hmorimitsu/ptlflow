from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tf


def occ_to_mask(occ, return_np=False):
    tens_out = nn.Sigmoid()(occ)
    if return_np:
        return np.round(tens_out.expand(-1, 3, -1, -1).data.cpu().numpy().transpose(
            [0, 2, 3, 1])) * 255
    return tens_out.round()


def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)


def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)


def _downsample2d_as(inputs, target_as):
    _, _, h, w = target_as.size()
    return tf.adaptive_avg_pool2d(inputs, [h, w])


def _upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)


def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 1)


def fbeta_score(y_true, y_pred, beta, eps=1e-8):
    beta2 = beta ** 2

    y_pred = y_pred.float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=2).sum(dim=2)
    precision = true_positive / (y_pred.sum(dim=2).sum(dim=2) + eps)
    recall = true_positive / (y_true.sum(dim=2).sum(dim=2) + eps)

    return torch.mean(precision * recall / (precision * beta2 + recall + eps) * (1 + beta2))


def f1_score_bal_loss(y_pred, y_true):
    eps = 1e-8

    tp = -(y_true * torch.log(y_pred + eps)).sum(dim=2).sum(dim=2).sum(dim=1)
    fn = -((1 - y_true) * torch.log((1 - y_pred) + eps)).sum(dim=2).sum(dim=2).sum(dim=1)

    denom_tp = y_true.sum(dim=2).sum(dim=2).sum(dim=1) + y_pred.sum(dim=2).sum(dim=2).sum(dim=1) + eps
    denom_fn = (1 - y_true).sum(dim=2).sum(dim=2).sum(dim=1) + (1 - y_pred).sum(dim=2).sum(dim=2).sum(dim=1) + eps

    return ((tp / denom_tp).sum() + (fn / denom_fn).sum()) * y_pred.size(2) * y_pred.size(3) * 0.5


class MultiScaleEPE_PWC_Bi_Occ_upsample(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Bi_Occ_upsample, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            output_flo = output_dict['flow_preds']
            output_occ = output_dict['occ_preds']

            # div_flow trick
            target_flo_f = self._args.div_flow * target_dict["flows"][:, 0]
            target_flo_b = self._args.div_flow * target_dict["flows_b"][:, 0]
            target_occ_f = target_dict["occs"][:, 0]
            target_occ_b = target_dict["occs_b"][:, 0]

            # bchw
            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    loss_ii = loss_ii + _elementwise_epe(output_ii[2 * jj],
                                                         _downsample2d_as(target_flo_f, output_ii[2 * jj])).sum()
                    loss_ii = loss_ii + _elementwise_epe(output_ii[2 * jj + 1],
                                                         _downsample2d_as(target_flo_b, output_ii[2 * jj + 1])).sum()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii)

            for ii, output_ii in enumerate(output_occ):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    output_occ_f = self.occ_activ(output_ii[2 * jj])
                    output_occ_b = self.occ_activ(output_ii[2 * jj + 1])
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_f,
                                                               _downsample2d_as(target_occ_f, output_occ_f))
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_b,
                                                               _downsample2d_as(target_occ_b, output_occ_b))
                occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii)

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["flows"][:, 0]).mean()
            loss_dict["F1"] = f1_score(target_dict["occs"][:, 0], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict


class MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.occ_loss_bce = nn.BCELoss(reduction='sum')
        self.seploss = hasattr(args, 'seploss') and self._args.seploss or False
        if self.seploss:
            print("Starting MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel loss with seploss!")
        self.loss_perc = hasattr(args, 'loss_perc') and args.loss_perc
        if self.loss_perc:
            from torchpercentile import Percentile
            # from tensorboard import summary
            # self.writer = summary(args.save)
            from matplotlib.pyplot import hist
            self.perc = Percentile()
            print("Starting MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel loss percentile!")
            self.min_p = nn.Parameter(torch.Tensor([30]), requires_grad=False)
            self.max_p = nn.Parameter(torch.Tensor([97]), requires_grad=False)

    def _get_flow_loss(self, output_flo, target_flo_f):
        flow_loss = 0
        for ii, output_ii in enumerate(output_flo):
            loss_ii = 0
            for jj in range(0, len(output_ii) // 2):
                cur_epe = _elementwise_robust_epe_char(output_ii[2 * jj],
                                                       _downsample2d_as(target_flo_f,
                                                                        output_ii[2 * jj]))
                if self.loss_perc:
                    tmin = torch.Tensor([self.perc(ten.flatten(), [self.min_p])
                                         for ten in cur_epe])
                    # tmin.requires_grad = False
                    tmax = torch.Tensor([self.perc(ten.flatten(), [self.max_p])
                                         for ten in cur_epe])
                    # tmax.requires_grad = False
                    cur_epe[cur_epe > tmax.view(-1, 1, 1, 1).cuda()] = 0.
                    cur_epe[cur_epe < tmin.view(-1, 1, 1, 1).cuda()] = 0.
                    tmin.detach()
                    tmax.detach()
                # self.writer.add_histogram('flow_epe', cur_epe.grad)
                # self.writer.add_scalar('test', 1)
                loss_ii = loss_ii + cur_epe.sum()
                output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
            flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii) * 2
        return flow_loss

    def _get_occ_loss(self, output_occ, target_occ_f):
        occ_loss = 0
        for ii, output_ii in enumerate(output_occ):
            loss_ii = 0
            for jj in range(0, len(output_ii) // 2):
                output_occ_f = self.occ_activ(output_ii[2 * jj])
                output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                loss_ii = loss_ii + self.occ_loss_bce(output_occ_f,
                                                      _downsample2d_as(target_occ_f,
                                                                       output_occ_f))
            occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii) * 2
        return occ_loss

    @staticmethod
    def mask_flow(output_flo, target_flo_f, output_occ, use_target=False, min_layer=0):
        for lix, layer_flow in enumerate(output_flo):
            if lix < min_layer:
                continue

            for fcix, flow_category in enumerate(layer_flow):
                if use_target:
                    occ_mask = _downsample2d_as(target_flo_f, output_flo[lix][fcix])
                else:
                    occ_mask = occ_to_mask(output_occ[lix][fcix])
                output_flo[lix][fcix] = (1 - occ_mask) * output_flo[lix][
                    fcix]

        if use_target:
            target_flo_f = (1 - target_flo_f) * target_flo_f
        else:
            target_flo_f = (1 - occ_to_mask(output_occ[6][0])) * target_flo_f

        return output_flo, target_flo_f

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            # Extract output
            output_flo = output_dict['flow_preds']
            output_occ = output_dict['occ_preds']

            # Extract and prepare target
            target_flo_f = self._args.div_flow * target_dict["flows"][:, 0]
            target_occ_f = target_dict["occs"][:, 0]

            # Mask with occ
            if self.seploss:
                output_flo, target_flo_f = self.mask_flow(output_flo,
                                                          target_flo_f,
                                                          output_occ)

            # Calc multi level losses
            flow_loss = self._get_flow_loss(output_flo, target_flo_f)
            occ_loss = self._get_occ_loss(output_occ, target_occ_f)
            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size

            # Calc total loss
            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1
            losses_mix = (flow_loss * f_l_w + occ_loss * o_l_w)
            loss_dict["loss"] = losses_mix / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"],
                                                target_dict["flows"][:, 0]).mean()
            loss_dict["F1"] = f1_score(target_dict["occs"][:, 0],
                                       torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict


class MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.001, 0.001, 0.001, 0.002, 0.004, 0.004, 0.004]

        self.occ_activ = nn.Sigmoid()

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        valid_mask = target_dict["valids"][:, 0]
        b, _, h, w = target_dict["flows"][:, 0].size()

        if self.training:
            output_flo = output_dict['flow_preds']
            output_occ = output_dict['occ_preds']

            # div_flow trick
            target_flo_f = self._args.div_flow * target_dict["flows"][:, 0]

            # bchw
            flow_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    valid_epe = _elementwise_robust_epe_char(_upsample2d_as(output_ii[2 * jj], target_flo_f),
                                                             target_flo_f) * valid_mask

                    for bb in range(0, b):
                        valid_epe[bb, ...][valid_mask[bb, ...] == 0] = valid_epe[bb, ...][
                            valid_mask[bb, ...] == 0].detach()
                        norm_const = h * w / (valid_mask[bb, ...].sum())
                        loss_ii = loss_ii + valid_epe[bb, ...][valid_mask[bb, ...] != 0].sum() * norm_const

                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii) * 2

            for ii, output_ii in enumerate(output_occ):
                for jj in range(0, len(output_ii) // 2):
                    output_ii[2 * jj] = output_ii[2 * jj].detach()
                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["loss"] = flow_loss / self._batch_size

        else:
            flow_gt_mag = torch.norm(target_dict["flows"][:, 0], p=2, dim=1, keepdim=True) + 1e-8
            flow_epe = _elementwise_epe(output_dict["flow"], target_dict["flows"][:, 0]) * valid_mask

            epe_per_image = (flow_epe.view(b, -1).sum(1)) / (valid_mask.view(b, -1).sum(1))
            loss_dict["epe"] = epe_per_image.mean()

            outlier_epe = (flow_epe > 3).float() * ((flow_epe / flow_gt_mag) > 0.05).float() * valid_mask
            outlier_per_image = (outlier_epe.view(b, -1).sum(1)) / (valid_mask.view(b, -1).sum(1))
            loss_dict["outlier"] = outlier_per_image.mean()

        return loss_dict


class NO_OP(nn.Module):
    def __init__(self, args=None):
        super(NO_OP, self).__init__()

    def forward(self, output_dict, target_dict):
        return {'flow_loss': -1, 'epe': -1,
                'total_loss': torch.Tensor([0])}
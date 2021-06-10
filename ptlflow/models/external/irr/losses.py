from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf


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


class MultiScaleEPE_FlowNet(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet, self).__init__()
        self._args = args        
        self._batch_size = args.train_batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target = self._args.div_flow * target_dict["flows"][:, 0]

            total_loss = 0
            for i, output_i in enumerate(outputs):
                target_i = _downsample2d_as(target, output_i)
                epe_i = _elementwise_epe(output_i, target_i)
                total_loss = total_loss + self._weights[i] * epe_i.sum() / self._batch_size
                loss_dict["epe%i" % (i + 2)] = epe_i.mean()
            loss_dict["loss"] = total_loss
        else:
            output = output_dict["flow1"]
            target = target_dict["flows"][:, 0]
            epe = _elementwise_epe(output, target)
            loss_dict["epe"] = epe.mean()

        return loss_dict

class MultiScaleEPE_FlowNet_IRR(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet_IRR, self).__init__()
        self._args = args        
        self._batch_size = args.train_batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target_f = self._args.div_flow * target_dict["flows"][:, 0]

            total_loss = 0
            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj, target_f_ii)
                    total_loss = total_loss + self._weights[ii] * epe_f_ii.sum()
                    loss_dict["epe%i" % (ii + 2)] = epe_f_ii.mean()
            loss_dict["loss"] = total_loss / self._batch_size / self._num_iters

        else:
            output = output_dict["flow1"]
            target_f = target_dict["flows"][:, 0]
            epe_f = _elementwise_epe(target_f, output)
            loss_dict["epe"] = epe_f.mean()

        return loss_dict

class MultiScaleEPE_FlowNet_IRR_Bi(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet_IRR_Bi, self).__init__()
        self._args = args        
        self._batch_size = args.train_batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        if self.training:
            outputs_flo = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target_f = self._args.div_flow * target_dict["flows"][:, 0]
            target_b = self._args.div_flow * target_dict["flows_b"][:, 0]

            total_loss = 0
            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0][0])
                target_b_ii = _downsample2d_as(target_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj[0], target_f_ii)
                    epe_b_ii = _elementwise_epe(output_ii_jj[1], target_b_ii)
                    total_loss = total_loss + self._weights[ii] * (epe_f_ii.sum() + epe_b_ii.sum())
                    loss_dict["epe%i" % (ii + 2)] = (epe_f_ii.mean() + epe_b_ii.mean()) / 2
            loss_dict["loss"] = total_loss / self._batch_size / self._num_iters / 2
        else:
            epe_f = _elementwise_epe(output_dict["flow1"], target_dict["flows"][:, 0])
            loss_dict["epe"] = epe_f.mean()

        return loss_dict

class MultiScaleEPE_FlowNet_IRR_Occ(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet_IRR_Occ, self).__init__()
        self._args = args        
        self._batch_size = args.train_batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters

        self.f1_score_bal_loss = f1_score_bal_loss
        self.occ_activ = nn.Sigmoid()
        
    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]
            outputs_occ = [output_dict[key] for key in ["occ2", "occ3", "occ4", "occ5", "occ6"]]

            # div_flow trick
            target = self._args.div_flow * target_dict["flows"][:, 0]
            target_occ = target_dict["occs"][:, 0]

            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                target_ii = _downsample2d_as(target, output_ii[0])
                for jj, output_ii_jj in enumerate(output_ii):
                    flow_loss = flow_loss + self._weights[ii] * _elementwise_epe(output_ii_jj, target_ii).sum()

            for ii, output_ii in enumerate(outputs_occ):
                target_occ_f = _downsample2d_as(target_occ, output_ii[0])
                for jj, output_ii_jj in enumerate(output_ii):
                    occ_loss = occ_loss + self._weights[ii] * self.f1_score_bal_loss(self.occ_activ(output_ii_jj), target_occ_f)

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size / self._num_iters
            loss_dict["occ_loss"] = occ_loss / self._batch_size / self._num_iters
            loss_dict["loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size / self._num_iters

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow1"], target_dict["flows"][:, 0]).mean()
            loss_dict["F1"] = f1_score(target_dict["occs"][:, 0], torch.round(self.occ_activ(output_dict["occ1"])))

        return loss_dict

class MultiScaleEPE_FlowNet_IRR_Bi_Occ(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet_IRR_Bi_Occ, self).__init__()
        self._args = args        
        self._batch_size = args.train_batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters

        self.f1_score_bal_loss = f1_score_bal_loss
        self.occ_activ = nn.Sigmoid()

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]
            outputs_occ = [output_dict[key] for key in ["occ2", "occ3", "occ4", "occ5", "occ6"]]

            # div_flow trick
            target_f = self._args.div_flow * target_dict["flows"][:, 0]
            target_b = self._args.div_flow * target_dict["flows_b"][:, 0]
            target_occ_f = target_dict["occs"][:, 0]
            target_occ_b = target_dict["occs_b"][:, 0]

            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0][0])
                target_b_ii = _downsample2d_as(target_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj[0], target_f_ii)
                    epe_b_ii = _elementwise_epe(output_ii_jj[1], target_b_ii)
                    flow_loss = flow_loss + self._weights[ii] * (epe_f_ii.sum() + epe_b_ii.sum()) * 0.5

            for ii, output_ii in enumerate(outputs_occ):
                target_occ_f = _downsample2d_as(target_occ_f, output_ii[0][0])
                target_occ_b = _downsample2d_as(target_occ_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    output_occ_f = self.occ_activ(output_ii_jj[0])
                    output_occ_b = self.occ_activ(output_ii_jj[1])
                    bce_f_ii = self.f1_score_bal_loss(output_occ_f, target_occ_f)
                    bce_b_ii = self.f1_score_bal_loss(output_occ_b, target_occ_b)
                    occ_loss = occ_loss + self._weights[ii] * (bce_f_ii + bce_b_ii) * 0.5

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size / self._num_iters
            loss_dict["occ_loss"] = occ_loss / self._batch_size / self._num_iters
            loss_dict["loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size / self._num_iters
        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow1"], target_dict["flows"][:, 0]).mean()
            loss_dict["F1"] = f1_score(target_dict["occs"][:, 0], torch.round(self.occ_activ(output_dict["occ1"])))

        return loss_dict

class MultiScaleEPE_FlowNet_IRR_Bi_Occ_upsample(nn.Module):
    def __init__(self,
                 args):
        super(MultiScaleEPE_FlowNet_IRR_Bi_Occ_upsample, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size        
        self._weights = [0.0003125, 0.00125, 0.005, 0.01, 0.02, 0.08, 0.32]
        
        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = [output_dict[key] for key in ["flow", "flow1", "flow2", "flow3", "flow4", "flow5", "flow6"]]
            outputs_occ = [output_dict[key] for key in ["occ", "occ1", "occ2", "occ3", "occ4", "occ5", "occ6"]]

            # div_flow trick
            target_f = self._args.div_flow * target_dict["flows"][:, 0]
            target_b = self._args.div_flow * target_dict["flows_b"][:, 0]
            target_occ_f = target_dict["occs"][:, 0]
            target_occ_b = target_dict["occs_b"][:, 0]

            num_iters = len(outputs_flo[0])
            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0][0])
                target_b_ii = _downsample2d_as(target_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj[0], target_f_ii)
                    epe_b_ii = _elementwise_epe(output_ii_jj[1], target_b_ii)
                    flow_loss = flow_loss + self._weights[ii] * (epe_f_ii.sum() + epe_b_ii.sum()) * 0.5

            for ii, output_ii in enumerate(outputs_occ):
                target_occ_f = _downsample2d_as(target_occ_f, output_ii[0][0])
                target_occ_b = _downsample2d_as(target_occ_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    output_occ_f = self.occ_activ(output_ii_jj[0])
                    output_occ_b = self.occ_activ(output_ii_jj[1])
                    bce_f_ii = self.f1_score_bal_loss(output_occ_f, target_occ_f)
                    bce_b_ii = self.f1_score_bal_loss(output_occ_b, target_occ_b)
                    occ_loss = occ_loss + self._weights[ii] * (bce_f_ii + bce_b_ii) * 0.5

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size / num_iters
            loss_dict["occ_loss"] = occ_loss / self._batch_size / num_iters
            loss_dict["loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size / num_iters
        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flows"][:, 0], target_dict["flows"][:, 0]).mean()
            loss_dict["F1"] = f1_score(target_dict["occs"][:, 0], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict



class MultiScaleEPE_PWC(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow_preds']

            # div_flow trick
            target = self._args.div_flow * target_dict["flows"][:, 0]

            total_loss = 0
            for ii, output_ii in enumerate(outputs):
                loss_ii = _elementwise_epe(output_ii, _downsample2d_as(target, output_ii)).sum()
                total_loss = total_loss + self._weights[ii] * loss_ii
            loss_dict["loss"] = total_loss / self._batch_size

        else:
            epe = _elementwise_epe(output_dict["flows"][:, 0], target_dict["flows"][:, 0])
            loss_dict["epe"] = epe.mean()

        return loss_dict

class MultiScaleEPE_PWC_Bi(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Bi, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow_preds']

            # div_flow trick
            target_f = self._args.div_flow * target_dict["flows"][:, 0]
            target_b = self._args.div_flow * target_dict["flows_b"][:, 0]

            total_loss = 0
            for i, output_i in enumerate(outputs):
                epe_i_f = _elementwise_epe(output_i[0], _downsample2d_as(target_f, output_i[0]))
                epe_i_b = _elementwise_epe(output_i[1], _downsample2d_as(target_b, output_i[1]))
                total_loss = total_loss + self._weights[i] * (epe_i_f.sum() + epe_i_b.sum())
            loss_dict["loss"] = total_loss / (2 * self._batch_size)
        else:
            epe = _elementwise_epe(output_dict["flows"][:, 0], target_dict["flows"][:, 0])
            loss_dict["epe"] = epe.mean()

        return loss_dict

class MultiScaleEPE_PWC_Occ(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Occ, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            output_flo = output_dict['flow_preds']
            output_occ = output_dict['occ_preds']

            # div_flow trick
            target_flo = self._args.div_flow * target_dict["flows"][:, 0]
            target_occ = target_dict["occs"][:, 0]

            flow_loss = 0
            occ_loss = 0

            for i, output_i in enumerate(output_flo):
                flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i, _downsample2d_as(target_flo, output_i)).sum()

            for i, output_i in enumerate(output_occ):
                output_occ = self.occ_activ(output_i)
                occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ, _downsample2d_as(target_occ, output_occ))

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flows"][:, 0], target_dict["flows"][:, 0]).mean()
            loss_dict["F1"] = f1_score(target_dict["occs"][:, 0], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict

class MultiScaleEPE_PWC_Bi_Occ(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Bi_Occ, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size        
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

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

            for i, output_i in enumerate(output_flo):
                flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i[0], _downsample2d_as(target_flo_f, output_i[0])).sum()
                flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i[1], _downsample2d_as(target_flo_b, output_i[1])).sum()

            for i, output_i in enumerate(output_occ):
                output_occ_f = self.occ_activ(output_i[0])
                output_occ_b = self.occ_activ(output_i[1])
                occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ_b, _downsample2d_as(target_occ_b, output_occ_b))

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / (2 * self._batch_size)
            loss_dict["occ_loss"] = occ_loss / (2 * self._batch_size) 
            loss_dict["loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / (2 * self._batch_size)

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flows"][:, 0], target_dict["flows"][:, 0]).mean()
            loss_dict["F1"] = f1_score(target_dict["occs"][:, 0], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict

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
                    loss_ii = loss_ii + _elementwise_epe(output_ii[2 * jj], _downsample2d_as(target_flo_f, output_ii[2 * jj])).sum()
                    loss_ii = loss_ii + _elementwise_epe(output_ii[2 * jj + 1], _downsample2d_as(target_flo_b, output_ii[2 * jj + 1])).sum()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii)

            for ii, output_ii in enumerate(output_occ):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    output_occ_f = self.occ_activ(output_ii[2 * jj])
                    output_occ_b = self.occ_activ(output_ii[2 * jj + 1])
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_b, _downsample2d_as(target_occ_b, output_occ_b))
                occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii)

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flows"][:, 0], target_dict["flows"][:, 0]).mean()
            loss_dict["F1"] = f1_score(target_dict["occs"][:, 0], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict

class MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel, self).__init__()
        self._args = args
        self._batch_size = args.train_batch_size        
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.occ_loss_bce = nn.BCELoss(reduction='sum')

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            output_flo = output_dict['flow_preds']
            output_occ = output_dict['occ_preds']

            # div_flow trick
            target_flo_f = self._args.div_flow * target_dict["flows"][:, 0]
            target_occ_f = target_dict["occs"][:, 0]

            # bchw
            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    loss_ii = loss_ii + _elementwise_robust_epe_char(output_ii[2 * jj], _downsample2d_as(target_flo_f, output_ii[2 * jj])).sum()
                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii) * 2

            for ii, output_ii in enumerate(output_occ):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    output_occ_f = self.occ_activ(output_ii[2 * jj])
                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                    loss_ii = loss_ii + self.occ_loss_bce(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii) * 2

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if f_loss > o_loss:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flows"][:, 0], target_dict["flows"][:, 0]).mean()
            loss_dict["F1"] = f1_score(target_dict["occs"][:, 0], torch.round(self.occ_activ(output_dict["occ"])))

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

        valid_mask = target_dict["input_valid"]
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
                    valid_epe = _elementwise_robust_epe_char(_upsample2d_as(output_ii[2 * jj], target_flo_f), target_flo_f) * valid_mask

                    for bb in range(0, b):
                        valid_epe[bb, ...][valid_mask[bb, ...] == 0] = valid_epe[bb, ...][valid_mask[bb, ...] == 0].detach()
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
            flow_epe = _elementwise_epe(output_dict["flows"][:, 0], target_dict["flows"][:, 0]) * valid_mask

            epe_per_image = (flow_epe.view(b, -1).sum(1)) / (valid_mask.view(b, -1).sum(1))
            loss_dict["epe"] = epe_per_image.mean()

            outlier_epe = (flow_epe > 3).float() * ((flow_epe / flow_gt_mag) > 0.05).float() * valid_mask
            outlier_per_image = (outlier_epe.view(b, -1).sum(1)) / (valid_mask.view(b, -1).sum(1))
            loss_dict["outlier"] = outlier_per_image.mean()

        return loss_dict
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

kMODE_DISC= 0
kMODE_GEN= 1
kMODE_MT = 2
kMODE_DISC_X = 3
kMODE_DISC_Y = 4

@dataclass
class DisentanglementSharedCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    disc_chance: float = field(
        default=0.5,
        metadata={"help": "percentage of minibatches used for training discriminators"},
    )
    lmbda: float = field(
        default=1.0,
        metadata={"help": "lmbda"},
    )
    lmbda_decay: float = field(
        default=0.0,
        metadata={"help": "how much to decay lmbda per update"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("label_smoothed_shared_disentanglement_loss", dataclass=DisentanglementSharedCriterionConfig)
class DisentanglementSharedCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, disc_chance, lmbda, lmbda_decay):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.lmbda = lmbda
        self.lmbda_init = lmbda
        self.lmbda_decay = lmbda_decay
        self.disc_chance = disc_chance

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
            
        chance = torch.rand(1)[0]

        mode = kMODE_GEN
        mode_model = kMODE_GEN
        if (chance<self.disc_chance):
            mode = kMODE_DISC
            mode_model = kMODE_DISC_X
        if (not model.training):
            mode = kMODE_GEN #Always gen mode in eval so we get all losses
            mode_model = kMODE_GEN #Always gen mode in eval so we get all losses
        net_output = model(**sample["net_input"],mode=mode_model,return_all_hiddens=False )
        l_components, nll_components = self.compute_loss(model, net_output, sample, reduce=reduce, mode = mode)

        assert(self.sentence_avg == False)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        x_ntokens  = sum(sample["net_input"]["src_lengths"])
        x_sample_size = x_ntokens
        ntokens = sample["ntokens"]
        nsentences = sample["target"].size(0)
        update = 0
        m = metrics.get_meter("train","num_updates")
        if m is not None:
            update = m.avg
        try:
            self.lmbda = max(0.05, self.lmbda_init  - self.lmbda_decay*update)
        except Exception:
            self.lmbda = 1
        metrics.log_scalar("lmbda", self.lmbda,0)
        metrics.log_scalar("disc_chance", self.disc_chance,0)

        if (mode == kMODE_GEN):
            losses = [self.lmbda*l_components[0], -(1-self.lmbda)*l_components[1]]
            loss = sum(losses)
            logging_output = {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
                "g_ntokens": sample["ntokens"],
                "g_loss_x": l_components[1].data,
                "g_loss_y": l_components[0].data,
                "g_nll_x": nll_components[1].data,
                "g_nll_y": nll_components[0].data,
                "g_x_ntokens": x_ntokens,
                "d_loss": 0,
                "d_loss_x": 0,
                "d_nll_y": 0,
                "d_ntokens": 0,
                "d_x_ntokens": 0,
                "d_nsentences": 0,
            }
        elif (mode == kMODE_DISC):
            losses = [l_components[0]]
            loss = losses[0]#*10
            logging_output = {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
                "d_loss": loss.data,
                "d_loss_x": l_components[0].data,
                "d_nll_x": nll_components[0].data,
                "d_ntokens": sample["ntokens"],
                "d_x_ntokens": x_ntokens,
                "d_nsentences": sample["target"].size(0),
                "g_ntokens": 0,
                "g_loss_x": 0,
                "g_loss_y": 0,
                "g_nll_x": 0,
                "g_nll_y": 0,
                "g_x_ntokens": 0,
            }

        else:
            print("WARNING | Unrecognized loss type from compute_loss")
        return losses, sample_size, logging_output, mode

    def compute_loss(self, model, net_output, sample, reduce=True, mode=kMODE_GEN):
        assert(reduce==True) #For now, maybe support not reducing later
        target_x, target_y = [target.view(-1) for target in  model.get_targets_list(sample, net_output)]

        loss = torch.tensor([0])
        if mode == kMODE_DISC:
            lp_x = model.get_normalized_probs_list(net_output, log_probs=True)[0]
            lp_x = lp_x.view(-1, lp_x.size(-1))
            
            l_x, nll_x = label_smoothed_nll_loss(
                lp_x,
                target_x,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            ) 
            return [l_x], [nll_x]
        elif mode == kMODE_GEN:
            lp_x, lp_y = model.get_normalized_probs_list(net_output, log_probs=True)
            lp_x = lp_x.view(-1, lp_x.size(-1))
            lp_y = lp_y.view(-1, lp_y.size(-1))
            l_x,nll_x = label_smoothed_nll_loss(lp_x, target_x, self.eps, ignore_index=self.padding_idx, reduce=reduce)
            l_y, nll_y = label_smoothed_nll_loss(lp_y, target_y, self.eps, ignore_index=self.padding_idx, reduce=reduce)
            return [l_y, l_x], [nll_y, nll_x]
        else:
            assert(False)





    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        assert(sample_size>0)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )


        g_loss_x_sum = sum(log.get("g_loss_x", 0) for log in logging_outputs)
        g_loss_y_sum = sum(log.get("g_loss_y", 0) for log in logging_outputs)

        g_nll_x_sum = sum(log.get("g_nll_x", 0) for log in logging_outputs)
        g_nll_y_sum = sum(log.get("g_nll_y", 0) for log in logging_outputs)

        g_ntokens = sum(log.get("g_ntokens", 0) for log in logging_outputs)
        g_x_ntokens = sum(log.get("g_x_ntokens", 0) for log in logging_outputs)
        g_sample_size = g_ntokens
        g_x_sample_size = g_x_ntokens

        d_loss_x_sum = sum(log.get("d_loss_x", 0) for log in logging_outputs)

        d_nll_x_sum = sum(log.get("d_nll_x", 0) for log in logging_outputs)

        d_ntokens = sum(log.get("d_ntokens", 0) for log in logging_outputs)
        d_sample_size = d_ntokens
        d_x_ntokens = sum(log.get("d_x_ntokens", 0) for log in logging_outputs)
        d_x_sample_size = d_x_ntokens

        # we divide by log(2) to convert the loss from base e to base 2
        if (g_sample_size!=0):#At least one gen batch
            if (g_x_sample_size ==0):
                g_x_sample_size = 1#To prevent dividing by zero in mt only mode
            metrics.log_scalar(
                "g_loss_x", g_loss_x_sum / g_x_sample_size / math.log(2), g_x_sample_size, round=3
            )
            metrics.log_scalar(
                "g_loss_y", g_loss_y_sum / g_sample_size / math.log(2), g_sample_size, round=3
            )
            metrics.log_scalar(
                "g_nll_x", g_nll_x_sum / g_x_sample_size / math.log(2), g_x_sample_size, round=3
            )
            metrics.log_scalar(
                "g_nll_y", g_nll_y_sum / g_sample_size / math.log(2), g_sample_size, round=3
            )
            if sample_size != ntokens:
                assert(False)#Should never get here, not supporting sentence_avg for now
            else:
                metrics.log_derived(
                    "g_ppl_x", lambda meters: utils.get_perplexity(meters["g_nll_x"].avg)
                )
                metrics.log_derived(
                    "g_ppl_y", lambda meters: utils.get_perplexity(meters["g_nll_y"].avg)
                )
        else:
            pass
        if (d_sample_size!=0):#At least one disc bat
            metrics.log_scalar(
                "d_loss_x", d_loss_x_sum / d_x_sample_size / math.log(2), d_x_sample_size, round=3
            )

            metrics.log_scalar(
                "d_nll_x", d_nll_x_sum / d_x_sample_size / math.log(2), d_x_sample_size, round=3
            )
            if sample_size != ntokens:
                assert(False)#Should never get here, not supporting sentence_avg for now
            else:
                metrics.log_derived(
                    "d_ppl_x", lambda meters: utils.get_perplexity(meters["d_nll_x"].avg)
                )
        else:
            pass


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


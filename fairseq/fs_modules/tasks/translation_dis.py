# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from argparse import Namespace
import copy
import torch

import numpy as np


from dataclasses import dataclass, field
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    TransformEosLangPairDataset,
    TransformEosDataset,
)
from fairseq.tasks.translation import TranslationTask, TranslationConfig, load_langpair_dataset
from fairseq.tasks import register_task


kMODE_DISC= 0
kMODE_GEN= 1
kMODE_MT = 2


@dataclass 
class DisentangledTranslationConfig(TranslationConfig):
    inference_mode : int = field(
        default = 0,
        metadata = {
        "help": "inference mode, shared->Y (0) or shared->X (1)"
        }
    )

@register_task("translation-dis", dataclass=DisentangledTranslationConfig)
class DisentangledTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """
    def __init__(self, cfg: DisentangledTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.inference_mode = self.cfg.inference_mode
        if (self.inference_mode == 1):
            self.src_langtok = 40000#Hacky, assumes we are using dict size of 40K (mBART), so next two are src and tgt langtoks
            self.tgt_langtok = 40000
        elif (self.inference_mode == 0):
            self.src_langtok = 40000
            self.tgt_langtok = 40001
        else:
            print('Not doing inference: Unknown inference mode')
        print('Inference mode is', self.inference_mode, 'src tgt langtoks are', self.src_langtok, self.tgt_langtok)

    def alter_dataset_langtok(
        self,
        lang_pair_dataset,
        src_eos=None,
        src_lang=None,
        tgt_eos=None,
        tgt_lang=None,
    ):
        if self.args.encoder_langtok is None and not self.args.decoder_langtok:
            return lang_pair_dataset

        new_src_eos = None
        if (
            self.args.encoder_langtok is not None
            and src_eos is not None
            and src_lang is not None
            and tgt_lang is not None
        ):
            new_src_eos = self.get_encoder_langtok(src_lang, tgt_lang)
        else:
            src_eos = None

        new_tgt_bos = None
        if self.args.decoder_langtok and tgt_eos is not None and tgt_lang is not None:
            new_tgt_bos = self.get_decoder_langtok(tgt_lang)
        else:
            tgt_eos = None

        return TransformEosLangPairDataset(
            lang_pair_dataset,
            src_eos=src_eos,
            new_src_eos=new_src_eos,
            tgt_bos=tgt_eos,
            new_tgt_bos=new_tgt_bos,
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if (self.inference_mode==1 and "target" in sample):#Hack for proper scoring of paraphrasing
            sample["target"][sample["target"]==2] = 40000
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=self.tgt_langtok,
            )
    
    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        dataset = super().build_dataset_for_inference(src_tokens, src_lengths)
        assert(self.inference_mode != -1)

        return TransformEosLangPairDataset(
            dataset,
            src_eos=self.src_dict.eos(),
            new_src_eos=self.src_langtok,
            tgt_bos=self.tgt_dict.eos(),
            new_tgt_bos=self.tgt_langtok,
        )
    def build_generator(self, models, args, **unused):
        kwargs = {'eos':self.tgt_langtok} if self.inference_mode==1 else {}
        return super().build_generator(models,args, extra_gen_cls_kwargs=kwargs)
    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output, _ = criterion(model, sample)
        return loss, sample_size, logging_output
        
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            losses, sample_size, logging_output, mode = criterion(model, sample)


        assert(len(losses)==1 or len(losses)==2)
        if ignore_grad:
            for loss in losses:
                loss *= 0

        with torch.autograd.profiler.record_function("backward"):
            if (mode == kMODE_DISC):#Updating adversarial decoders
                loss = losses[0]
                model.set_freeze_encoders(True)
                optimizer.backward(loss)
                #loss.backward()
                model.set_freeze_encoders(False)
            elif (mode == kMODE_GEN):#Updating everything else
                l1 = losses[0]
                l2 = losses[1]
                #Freeze decoder and backprop adversarial part of loss to encoder
                model.set_freeze_decoder(True)
                optimizer.backward(l2, retain_graph=True)
                #l2.backward(retain_graph=True)
                #Unfreeze decoder and backprop generative part of loss to encoder + decoder


                model.set_freeze_decoder(False)
                optimizer.backward(l1)
                #l1.backward()
            else:
                assert(mode == kMODE_MT)
                loss = losses[0]
                optimizer.backward(loss)
        loss = sum(losses)
        return loss, sample_size, logging_output

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        dataset = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )
        print('Loaded langpair dataset,size ', dataset.__len__())
        if (self.inference_mode!=-1):
            dataset = TransformEosLangPairDataset(
                dataset,
                src_eos=self.src_dict.eos(),
                new_src_eos=self.src_langtok,
                tgt_bos=self.tgt_dict.eos(),
                new_tgt_bos=self.tgt_langtok,
            )
            dataset = TransformEosDataset(dataset, self.tgt_langtok, append_eos_to_tgt=True)
            dataset = TransformEosDataset(dataset, 2, remove_eos_from_tgt=True)
        
        self.datasets[split] = dataset

# Principled Paraphrasing from Parallel Corpora

This repository contains the implementation of our paraphrasing model, as described in [our paper](https://aclanthology.org/2022.acl-long.114.pdf).

## Training

It is important that you use the version of fairseq included in this repo, as it contains minor changes to the optimizers that allow to pass the `retain_graph` flag to the `backward` function, which is necessary for the proper propagation of gradients as described in the paper.

To train a model, you first need to binarize your bilingual training data, using the mBART sentencepiece tokenizer as described in the [mBART page](https://github.com/facebookresearch/fairseq/blob/main/examples/mbart/README.md). Instead of using the full pre-trained dictionary of 250K words for `fairseq-preprocess`, we create a new one of the most common 40K words. That is, your `fairseq-preprocess` call should look like this:

```
fairseq-preprocess \
  --source-lang en_XX \
  --target-lang fr_XX \
  --trainpref train.spm \
  --validpref valid.spm \
  --testpref test.spm \
  --destdir ${data_bin_dir} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --nwordssrc 40000 \
  --nwordstgt 40000 \
  --joined-dictionary \
  --workers 70 
  ```
  
  Once you have binarized your data, you can call fairseq-trained, passing the `fs_modules` subdirectory as the `user-dir`. Here is how a call might look like:
  
  ```
  user_dir=/path/to/fairseq/fs_modules/
  mbart_dir=/path/to/mBART/mbart.cc25.v2/
  fairseq-train  ${data_bin_dir} \
  --arch transformer-dis-single \
  --criterion label_smoothed_shared_disentanglement_loss \
  --task translation-dis \
  --load-mbart ${mbart_dir} \
  --optimizer adam \
  --user-dir ${user_dir} \
  --encoder-embed-dim 1024 \
  --encoder-ffn-embed-dim 4096 \
  --decoder-embed-dim 1024 \
  --decoder-ffn-embed-dim 4096 \
  --encoder-embed-dim 1024 \
  --encoder-ffn-embed-dim 4096 \
  --ddp-backend=no_c10d \
  --max-tokens 2048 \
  --adam-betas '(0.9, 0.98)' \
  --update-freq 8 \
  --log-format json \
  --log-interval 1 \
  --lr 3e-5 \
  --lr-scheduler polynomial_decay \
  --total-num-update ${total_num_update} \
  --save-interval-updates 10000 \
  --warmup-updates 2500  \
  --share-all-embeddings \
  --fp16 \
  --label-smoothing 0.2 \
  --dropout 0.3 \
  --attention-dropout 0.1 \
  --skip-invalid-size-inputs-valid-test  \
  --src-langtok '[en_XX]' \
  --tgt-langtok '[fr_XX]' \
  -s en_XX -t fr_XX \
  --disc-chance ${disc_chance} \
  --lmbda ${lambda} \
  --save-dir ${save_dir} \
  --max-update 100005
  ```
  
  Where `${lambda}` and `${disc_chance}` correspond to $\lambda$ and $K$ in the paper, respectively.
  
  ## Pretrained model
  Alternatively, you can download our pretrained model for $\lambda=0.73$ and $K=0.7$ [here](ttps://ixa2.si.ehu.eus/principled-paraphrasing/model_and_dicts.tar.gz).
  
  ## Evaluation
  
  Once the model is trained, you can run inference on it using the same task as for training, with an additional `--inference-mode 1` parameter. An inference command might look like this:
  
  ```
  fairseq-interactive ${data_bin_dir} \
  --path $checkpoint \
  --task translation-dis \
  --source-lang en_XX \
  --target-lang fr_XX \
  --bpe sentencepiece \
  --sentencepiece-model ${mbart_dir}/sentence.bpe.model \
  --user-dir ${user_dir} \
  --nbest 1 \
  --beam 5 \
  --inference-mode 1 \
  ```
  
  ## Citing
  
  If you use the code in this repository for your research, please cite our paper:
  ```
  @inproceedings{artetxe2018acl,
  author    = {Ormazabal, Aitor and Artetxe, Mikel and Soroa, Aitor and Labaka, Gorka and Agirre, Eneko},
  title     = {Principled Paraphrase Generation with Parallel Corpora},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2022},
  pages     = {1621–1638}
  }
  ```

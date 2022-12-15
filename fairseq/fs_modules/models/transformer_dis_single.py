# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import sys
import torch
import torch.nn as nn
import copy
from fairseq import utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.bart import BARTModel
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerEncoder, TransformerModel

from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

kMODE_DISC= 0
kMODE_GEN= 1
kMODE_MT = 2
kMODE_DISC_X = 3
kMODE_DISC_Y = 4




class TransformerBilingualDecoder(FairseqDecoder):
    def __init__(self, base_decoder, bos): 
        super().__init__(base_decoder.dictionary)
        self.base_decoder = base_decoder
        self.bos = bos
    def forward(
        self,
        prev_output_tokens,
        **kwargs
        ):
        #assert(self.bos is not None)
        #prev_output_tokens[:, 0] = self.bos This is done in task now
        return  self.base_decoder.forward(prev_output_tokens, **kwargs)




@register_model("transformer-dis-single")
class TransformerDisentangledSingleModel(BaseFairseqModel):

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
        }
        # fmt: on

    def __init__(self, args, encoder, universal_decoder, decoder_src_bos, decoder_tgt_bos, inference_mode):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.universal_decoder = universal_decoder

        self.decoder_src_bos = decoder_src_bos
        self.decoder_tgt_bos = decoder_tgt_bos
        self.inference_mode = inference_mode

        print('Model inference mode:', inference_mode)
        print("Decoder src bos, decoder tgt bos:", decoder_src_bos, decoder_tgt_bos)


        self.decoder = TransformerBilingualDecoder(universal_decoder, decoder_tgt_bos)
        self.supports_align_args = True


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--generation-mode',default=0)
        parser.add_argument('--score_paraphrases',default=False)
        parser.add_argument('--src-langtok',default=None)
        parser.add_argument('--tgt-langtok',default=None)
        parser.add_argument('--load-pretrained-model', default=None,
                            help = 'file from which to load pretrained encoder/decoder')
        parser.add_argument('--load-mbart', default=None,
                            help = 'file from which to load pretrained mBART encoder/decoder')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num shared encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)


        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        
       
        eos = src_dict.eos()
        assert args.load_mbart is not None, "You must provide mbart location to load-mbart"
        assert(args.src_langtok is not None and args.tgt_langtok is not None)
        src_dict.add_symbol(args.src_langtok)
        src_dict.add_symbol(args.tgt_langtok)
        src_bos =  tgt_dict.add_symbol(args.src_langtok)
        tgt_bos =  tgt_dict.add_symbol(args.tgt_langtok)
        assert(src_bos != tgt_dict.unk() and tgt_bos != tgt_dict.unk())
        print("Mbart src and tgt bos idx:",src_bos,tgt_bos)

        print("Generation mode is ", args.generation_mode)
        print('unk eos bos:',src_dict.unk(),src_dict.eos(),src_dict.bos())
        trained_model  = None
        trained_model = BARTModel.from_pretrained(args.load_mbart).models[0]
        assert(args.share_all_embeddings)
        trained_encoder = trained_model.encoder
        trained_decoder = trained_model.decoder

        encoder_dict = trained_encoder.dictionary
        decoder_dict = trained_decoder.dictionary

        encoder_embs = trained_encoder.embed_tokens
        decoder_embs = trained_decoder.embed_tokens

        assert(encoder_embs.embedding_dim == args.encoder_embed_dim)
        assert(decoder_embs.embedding_dim == args.decoder_embed_dim)

        if src_dict != tgt_dict:
            raise ValueError("--share-all-embeddings requires a joined dictionary")
        if args.encoder_embed_dim != args.decoder_embed_dim:
            raise ValueError(
                "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
            )
        encoder_embed_tokens = cls.build_embedding_from_pretrained(
            args, src_dict, args.encoder_embed_dim, encoder_embs, encoder_dict
        )
        decoder_embed_tokens = encoder_embed_tokens
        args.share_decoder_input_output_embed = True


        trained_encoder.embed_tokens = encoder_embed_tokens 
        trained_decoder.embed_tokens = decoder_embed_tokens
        trained_encoder.dictionary = src_dict
        trained_decoder.dictionary = tgt_dict
        trained_decoder.output_projection = nn.Linear(
                decoder_embed_tokens.weight.shape[1],
                decoder_embed_tokens.weight.shape[0],
                bias=False,
        )
        trained_decoder.output_projection.weight = decoder_embed_tokens.weight
        

        encoder = trained_encoder
        universal_decoder = trained_decoder

        print("Resetting dropout", args.dropout, args.attention_dropout, args.activation_dropout)

        encoder.dropout_module.p = args.dropout
        universal_decoder.dropout_module.p = args.dropout

        for l in encoder.layers:
            l.dropout_module.p = args.dropout
            l.self_attn.dropout_module.p = args.attention_dropout
            l.activation_dropout_module.p = args.activation_dropout
        for l in universal_decoder.layers:
            l.dropout_module.p = args.dropout
            l.self_attn.dropout_module.p = args.attention_dropout
            l.encoder_attn.dropout_module.p = args.attention_dropout
            l.activation_dropout_module.p = args.activation_dropout
        print("Resulting embedding sizes mBart:")
        print("Encoder mbart", encoder.embed_tokens.weight.size(), encoder.dictionary.__len__())
        print("Universal decoder mbart", universal_decoder.embed_tokens.weight.size(), universal_decoder.dictionary.__len__())


        
        return cls(args, encoder, universal_decoder, src_bos, tgt_bos, task.inference_mode)
    @classmethod
    def build_encoder_layer(cls, args):
        layer = TransformerEncoderLayer(args)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_embedding_from_pretrained(cls, args, dictionary, embed_dim, pre_embedding, pre_dictionary):#Builds embedding initializing vectors for tokens present in a preexisting embedding
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        iv = 0
        oov = 0
        for idx in range(len(dictionary)):
            token = dictionary[idx]
            if pre_dictionary.__contains__(token):
                pre_idx = pre_dictionary.index(token)
                emb.weight.data[idx] = pre_embedding.weight.data[pre_idx]
                iv+=1
            else:
                text = ("WARNING: Word{} with id {} is OOV when building from pretrained\n".format(token,idx)).encode('utf-8')
                sys.stdout.buffer.write(text)
                oov+=1

        print("Percentage of words found in pre-trained model vocab:", 100*iv/(iv+oov))
        return emb
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        **kwargs
    ):
        val =  super().load_state_dict(state_dict, strict=True, **kwargs)
        return val

    def get_targets_list(self, sample, net_output):
        #print(' TARGET LIST:',sample["net_input"]["src_tokens"], sample["target"])
        return sample["net_input"]["src_tokens"], sample["target"]

    def get_targets(self, sample, net_output):
        #print('TARGET:', sample["target"])
        return sample["target"]

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    def set_freeze_encoders(self, freeze):
        val = True 
        if (freeze):
            val = False
        for param in self.encoder.parameters():
            param.requires_grad = val
    def set_freeze_decoder(self, freeze):
        val = True 
        if (freeze):
            val = False
        for param in self.universal_decoder.parameters():
            param.requires_grad = val
          
    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        mode: int = 1, #Default mode generative
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        src_tokens = copy.deepcopy(src_tokens)

        if (src_tokens[0,-1] != self.decoder_src_bos):
            print("WARNING: src tokens bos not what it should be, it is {}, should be {}".format(src_tokens[0,-1], self.decoder_src_bos))

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)



        prev_output_tokens = copy.deepcopy(prev_output_tokens)

        prev_output_tokens_source = src_tokens[:,:-1]
        src_bos_tokens = prev_output_tokens_source.new_full((prev_output_tokens_source.size()[0],1), self.decoder_src_bos)
        prev_output_tokens_source = torch.cat((src_bos_tokens, prev_output_tokens_source), dim=1)


        #print('Mode',mode)
        #print('src_tokens',src_tokens)
        #print('prev_output_tokens',prev_output_tokens)
        #print('prev_output_tokens_source',prev_output_tokens_source)

        if (mode == kMODE_DISC_Y):
            ancillary_target_out = self.universal_decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )

            return [ancillary_target_out]

        elif (mode == kMODE_DISC_X):
            sufficient_source_out = self.universal_decoder(
                prev_output_tokens_source,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )
            return [sufficient_source_out]


        elif (mode == kMODE_GEN):
            source_out = self.universal_decoder(
                prev_output_tokens_source,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )
            target_out = self.universal_decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )
            if (self.inference_mode==1):#Inference mode 1 means target is en_XX, return only target for scoring
                return target_out
            else:
                return [source_out, target_out]
        
        #Should never get here
        raise NotImplementedError


    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_outputs: List[Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        if isinstance(net_outputs,list):#for when we are training with a criterion other than disentanglement, selects the l_shared_y part of loss to train 
            net_output = net_outputs[0]
        else:#For generation when encoder/decoder forward has been done separately, loss is already l_shared_y part
            net_output = net_outputs#[3] #Assuming this is called only when generating, where encoder_forward has been done separately and we ony have one output, change later?
        logits = net_output[0]
        if log_probs:
            return ( utils.log_softmax(logits, dim=-1, onnx_trace=False))
        else:
            return ( utils.softmax(logits, dim=-1, onnx_trace=False) )

    @torch.jit.export
    def get_normalized_probs_list(
        self,
        net_outputs: List[Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        outputs = []

        for output in net_outputs:
            logits = output[0]
            if log_probs:
                outputs.append( utils.log_softmax(logits, dim=-1, onnx_trace=False))
            else:
                outputs.append( utils.softmax(logits, dim=-1, onnx_trace=False) )

        """Get normalized probabilities (or log probs) from a net's output."""
        return outputs


@register_model_architecture("transformer-dis-single", "transformer-dis-single")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)

    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


import torch
import torch.nn as nn 
from .transformer import Transformer
from .seq2seq import EncoderDecoder
class Seq2seq(nn.Module):
    def __init__(self, opt):
        super(Seq2seq, self).__init__()
        self.model = Transformer(
            opt.src_vocab_size,
            opt.tgt_vocab_size,
            opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=opt.proj_share_weight,
            emb_src_tgt_weight_sharing=opt.embs_share_weight,
            d_k=opt.d_k,
            d_v=opt.d_v,
            d_model=opt.d_model,
            d_word_vec=opt.d_word_vec,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            dropout=opt.dropout
        )
        self.model = EncoderDecoder(src_vocab=opt.src_vocab_size, tgt_vocab=opt.tgt_vocab_size, embedding_size=opt.embedding_size, hidden_size=opt.hidden_size, num_layer=opt.num_layer, dropout=opt.dropout)
    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        return self.model(src_seq, src_pos, tgt_seq, tgt_pos)
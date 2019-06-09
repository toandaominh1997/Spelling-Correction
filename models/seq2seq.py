import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, src_vocab, embedding_size, hidden_size, num_layer=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layer
        self.embedding = nn.Embedding(src_vocab, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layer, batch_first=True, bidirectional=True, dropout=dropout)
        
    def forward(self, input):
        input = self.embedding(input) 
        output, (hidden, cell) = self.rnn(input)
        fwd_hidden = hidden[0:hidden.size(0):2]
        bwd_hidden = hidden[1:hidden.size(0):2]
        fwd_cell = cell[0:cell.size(0):2]
        bwd_cell = cell[1:cell.size(0):2]
        hidden = torch.cat([fwd_hidden, bwd_hidden], dim=2)  # [num_layers, batch, 2*dim]
        cell = torch.cat([fwd_cell, bwd_cell], dim=2)
        return output, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        '''
        query: (B x 1 x dim)
        proj_key: (B x seq_len x dim)
        value: (B x seq_len x dim*2)
        '''
        assert mask is not None, "mask is required"

        # query: (B x 1 x feature)
        query = self.query_layer(query) # query: (B x 1 x hidden size)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key)) # scores: (B x seq_len x 1)
        scores = scores.squeeze(2).unsqueeze(1) # scores: (B x 1 x seq_len)
        
        mask = mask.unsqueeze(-2)
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1) # alphas: (B x 1 x seq_len)
        self.alphas = alphas
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value) # context: (B x 1 x dim*2)
        
        # context: (B, 1, 2*dim), alphas shape: (B, 1, seq_len)
        return context, alphas    
    

class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, tgt_vocab, embedding_size, hidden_size, attention, num_layers=1, dropout=0.5, bridge=True):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.embedding = nn.Embedding(tgt_vocab, embedding_size)
        self.rnn = nn.LSTM(embedding_size + 2*hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
                
        # to initialize from the final encoder state
        self.bridge_hidden = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None
        self.bridge_cell = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + embedding_size, hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_pos, proj_key, hidden, cell):
        """Perform a single decoder step (1 word)"""
        
        # compute context vector using attention mechanism

        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(query=query, proj_key=proj_key, value=encoder_hidden, mask=src_pos)
        # update rnn hidden state
        
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        output = torch.cat([prev_embed, output, context], dim=2)
        output = self.dropout_layer(output)
        output = self.pre_output_layer(output)

        return output
    
    def forward(self, tgt_seq, encoder_output, encoder_hidden, encoder_cell,
                src_pos, tgt_pos, hidden_cell=None, max_len=None):
        """Unroll the decoder one step at a time."""
        outputs = []
        tgt_embedding = self.embedding(tgt_seq)
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = tgt_seq.size(-1)

        # initialize decoder hidden state, hidden: (num_layers, batch_size, hidden_size) 
        if hidden_cell is None:
            hidden, cell = self.init_hidden(encoder_hidden, encoder_cell)

        
        proj_key = self.attention.key_layer(encoder_output)

        for i in range(max_len):
            prev_embedding = tgt_embedding[:, i].unsqueeze(1)

            output = self.forward_step(prev_embedding, encoder_output, src_pos, proj_key, hidden, cell)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def init_hidden(self, encoder_hidden, encoder_cell):
        if encoder_hidden is None or encoder_cell is None:
            return None, None  # start with zeros
        
        return torch.tanh(self.bridge_hidden(encoder_hidden)), torch.tanh(self.bridge_cell(encoder_cell))

class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embedding_size=256, hidden_size=512, num_layer=1, dropout=0.1):
        super(EncoderDecoder, self).__init__()
        self.attention = BahdanauAttention(hidden_size)
        self.encoder = Encoder(src_vocab=src_vocab, embedding_size=embedding_size, hidden_size=hidden_size, num_layer=num_layer, dropout=dropout)
        self.decoder = Decoder(tgt_vocab=tgt_vocab, embedding_size=embedding_size, hidden_size=hidden_size, attention=self.attention, num_layers=num_layer, dropout=dropout, bridge=True)
        self.fc = nn.Linear(hidden_size, tgt_vocab, bias=False)

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        encoder_output, encoder_hidden, encoder_cell = self.encoder(src_seq)
        outputs = self.decoder(tgt_seq=tgt_seq, encoder_output=encoder_output, encoder_hidden=encoder_hidden, encoder_cell=encoder_cell, src_pos=src_pos, tgt_pos=tgt_pos, hidden_cell=None, max_len=None)
        out = F.log_softmax(self.fc(outputs), dim=-1)
        return out.view(-1, out.size(-1))

'''
This script handling the training process.
'''

import argparse
import math
import time
import os

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import modules.Constants as Constants
from modules.Models import Transformer
from modules.Optim import ScheduledOptim
import modules.Constants as Constants
from glob import glob
from datasets import dataset
import time 
import datetime
from models.model import Seq2seq

from util.util import get_gpu

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss

def train_epoch(model, train_dataloader, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    for batch in tqdm(train_dataloader, mininterval=2, desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)

        gold = tgt_seq[:, 1:]
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
        optimizer.zero_grad()
        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, valid_dataloader, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, mininterval=2, desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, train_dataloader, valid_dataloader, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, train_dataloader, optimizer, device, smoothing=opt.label_smoothing)
        
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        valid_loss, valid_accu = eval_epoch(model, valid_dataloader, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = os.path.join(opt.save_model, 'model_{accu:3.3f}.pt'.format(accu=100*valid_accu))
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':

                model_name = os.path.join(opt.save_model, 'model_best.pth')
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    ''' FOR DATASET '''
    parser.add_argument('--train_src', required=True)
    parser.add_argument('--valid_src', required=True)
    parser.add_argument('--max_word_seq_len', type=int, default=100)
    parser.add_argument('--min_word_count', type=int, default=5)
    parser.add_argument('--keep_case', action='store_true')

    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_worker', type=int, default=8)
    
    ''' FOR model '''
    parser.add_argument('--net', type=str, default='transformer', help='transformer, seq2seq')


    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_inner_hid', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)

    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_warmup_steps', type=int, default=4000)

    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layer', type=int, default=1)
    
    

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embs_share_weight', action='store_true')
    parser.add_argument('--proj_share_weight', action='store_true')

    parser.add_argument('--log', default=None)
    parser.add_argument('--save_model', default='saved')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('--cuda', action='store_true', default=True)

    ''' FOR optimizer'''
    parser.add_argument('--label_smoothing', action='store_true')
    opt = parser.parse_args()
    start_time = datetime.datetime.now().strftime('%m-%d_%H%M')
    if not os.path.exists(os.path.join(opt.save_model, start_time)):
        os.makedirs(os.path.join(opt.save_model, start_time))
        opt.save_model = os.path.join(opt.save_model, start_time)

    if(torch.cuda.is_available() and opt.cuda):
        torch.cuda.set_device(get_gpu())
    opt.d_word_vec = opt.d_model

    opt.max_token_seq_len = opt.max_word_seq_len + 2
    #========= Loading Dataset =========#
    train_dataset = dataset.TranslationDataset(dir_name=opt.train_src, max_word_seq_len= opt.max_word_seq_len, min_word_count=opt.min_word_count, keep_case=opt.keep_case, training=1, src_word2idx=None, tgt_word2idx=None)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.num_worker, batch_size=opt.batch_size, collate_fn=dataset.alignCollate(opt), shuffle=True)

    valid_dataset = dataset.TranslationDataset(dir_name=opt.valid_src, max_word_seq_len= opt.max_word_seq_len, min_word_count=opt.min_word_count, keep_case=opt.keep_case, training=0, src_word2idx=train_dataloader.dataset.src_word2idx, tgt_word2idx=train_dataloader.dataset.tgt_word2idx)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, num_workers=opt.num_worker, batch_size=opt.batch_size, collate_fn=dataset.alignCollate(opt), shuffle=False)
    
    data = {
        'dict': {
            'src': train_dataloader.dataset.src_word2idx,
            'tgt': train_dataloader.dataset.tgt_word2idx}
        }
    print('[Info] Dumping the processed data to pickle file', opt.save_model)
    torch.save(data, os.path.join(opt.save_model, 'dict.pth'))
    print('[Info] Finish.')
    del data
    opt.src_vocab_size = train_dataloader.dataset.src_vocab_size
    opt.tgt_vocab_size = train_dataloader.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert train_dataloader.dataset.src_word2idx == train_dataloader.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')

    model = Seq2seq(opt).to(device)
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train(model, train_dataloader, valid_dataloader, optimizer, device ,opt)




if __name__ == '__main__':
    main()

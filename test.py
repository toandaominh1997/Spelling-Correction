import torch 
import torch.utils.data 
import argparse
from tqdm import tqdm 

from datasets.dataset import collate_fn, TranslationDataset
from transformer.Translator import Translator

def main():
    parser = argparse.ArgumentParser(description='test.py')
    parser.add_argument('--model', required=True, help='Path to model .pth file')
    parser.add_argument('--src', required=True, help='Path to source file of test file')
    parser.add_argument('--vocab', required=True, help='Path to vocab of test file')
    parser.add_argument('--max_word_seq_len', type=int, default=100)
    parser.add_argument('--min_word_count', type=int, default=5)
    parser.add_argument('--keep_case', action='store_true')

    parser.add_argument('--output', default='pred.txt', help='Path to output the predictions (each line will be the decoded sequence')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--num_worker', type=int, default=8)

    parser.add_argument('--n_best', type=int, default=1, help='If verbose is set, will output the n_best decoded sentences')
    parser.add_argument('no_cuda', action='store_true')
    parser.add_argument('cuda', action='store_true')
    opt = parser.parse_args()

    
    if opt.no_cuda==False:
        opt.cuda=True
    
    preprocess_data = torch.load(opt.vocab)['dict']
    src_word2idx = preprocess_data['src']
    tgt_word2idx = preprocess_data['tgt']
    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            dir_name=opt.src,
            max_word_seq_len=opt.max_word_seq_len,
            min_word_count = opt.min_word_count,
            keep_case=opt.keep_case,
            training=False,
            src_word2idx=src_word2idx,
            tgt_word2idx=tgt_word2idx
        ),
        num_workers=opt.num_worker,
        batch_size=opt.batch_size,
        collate_fn=collate_fn,
        shuffle=False)
    translator = Translator(opt)

    with open(opt.output, 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores = translator.translate_batch(*batch)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = ''.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                    f.write(pred_line + '\n')
    print('[Info] Finished.')


if __name__ =="__main__":
    main()
''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants
from util import generate_mistake
from util import getvocab
import glob

from sklearn.model_selection import train_test_split
def read_instances_from_file(inst_file, max_sent_len, keep_case, vocab=None):
    ''' Convert file into word seq lists and vocab '''

    # word_insts = []
    train_word_insts = []
    target_word_insts = []
    trimmed_sent_count = 0
    with open(inst_file, encoding='latin-1') as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            # words = sent.split()
            sent = sent.replace('\n', '').replace('\\', '')
            sent = sent.replace('\t', '')
            words = list(sent)
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]
            if word_inst:
                # word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                train_word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                target_word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                if(vocab!=None):
                    aug_word = generate_mistake.generate_mistakes(words, vocab)
                    aug_word_inst = aug_word[:max_sent_len] 
                    train_word_insts += [[Constants.BOS_WORD] + aug_word_inst + [Constants.EOS_WORD]]
                    target_word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            # else:
            #     word_insts += [None]
            #     # train_word_insts += [None]
            #     # target_word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(train_word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return train_word_insts, target_word_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    # parser.add_argument('-train_tgt', required=True)
    # parser.add_argument('-valid_src', required=True)
    # parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=100)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')

    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # full_vocab = getvocab.getvocab(opt.train_src)
    train_src_word_insts = []
    train_tgt_word_insts = []
    full_vocab = ''
    for filename in glob.glob('{}/*.*'.format(opt.train_src)):
        print(filename)
        full_vocab +=  getvocab.getvocab(filename).replace('\n', '').replace('\t', '').replace('\ufeff', '')
    full_vocab = ''.join(sorted(set(full_vocab)))
    # Training set
    train_src_word_insts = []
    train_tgt_word_insts = []
    for filename in glob.glob('{}/*.*'.format(opt.train_src)):
        src_word_insts, tgt_word_insts = read_instances_from_file(filename, opt.max_word_seq_len, opt.keep_case, full_vocab)
        train_src_word_insts +=src_word_insts
        train_tgt_word_insts +=tgt_word_insts
    
    # train_tgt_word_insts = read_instances_from_file(
    #     opt.train_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # # Validation set
    # valid_src_word_insts = read_instances_from_file(
    #     opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    # valid_tgt_word_insts = read_instances_from_file(
    #     opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)

    # if len(valid_src_word_insts) != len(valid_tgt_word_insts):
    #     print('[Warning] The validation instance count is not equal.')
    #     min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
    #     valid_src_word_insts = valid_src_word_insts[:min_inst_count]
    #     valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    # #- Remove empty instances
    # valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
    #     (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))
    
    train_src_word_insts, valid_src_word_insts, train_tgt_word_insts, valid_tgt_word_insts = train_test_split(train_src_word_insts, train_tgt_word_insts, test_size=0.2, random_state=42)
    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()

import os
from glob import glob 
# import transformer.Constants as Constants

def getvocab(dir_name):
    vocab = list()
    if(os.path.isdir(dir_name)):
        for filename in glob('{}/*.*'.format(dir_name)):
            vocab += list(open(filename, encoding='utf-8'))
    else:
        vocab += list(open(dir_name, encoding='utf-8'))
    
    alphabets = ''.join(sorted(set(''.join(vocab))))
    # word2idx = {
    #     Constants.BOS_WORD: Constants.BOS,
    #     Constants.EOS_WORD: Constants.EOS,
    #     Constants.PAD_WORD: Constants.PAD,
    #     Constants.UNK_WORD: Constants.UNK}
    # word_count = {w:0 for w in alphabets}
    
    return alphabets.replace('\n', '').replace('\t', '')

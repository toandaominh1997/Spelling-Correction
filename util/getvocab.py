import transformer.Constants as Constants
def getvocab(filename):
    vocab = list(open(filename))
    alphabets = ''.join(sorted(set(''.join(vocab))))
    return alphabets
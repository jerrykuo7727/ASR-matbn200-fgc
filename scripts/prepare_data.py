import os
import sys


if __name__ == '__main__':

    # Language data
    phone_set = set()
    with open('data/local/dict/lexicon.txt', encoding='utf-8') as f:
        f.readline()
        f.readline()
        for line in f:
            for phone in line.split()[1:]:
                phone_set.add(phone)
    
    with open('data/local/dict/nonsilence_phones.txt', mode='w') as f:
        for phone in phone_set:
            f.write('%s\n' % phone)
        
    with open('data/local/dict/silence_phones.txt', mode='w') as f:
        f.write('sil\n')
        f.write('spn\n')
            
    with open('data/local/dict/optional_silence.txt', mode='w') as f:
        f.write('sil\n')

    print('`scripts/prepare_data.py` ends successfully.')
    exit(0)

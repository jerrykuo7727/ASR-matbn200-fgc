import sys
import os

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python $0 <text-in> <text-out>')
        exit(1)

    text_in = sys.argv[1]
    text_out = sys.argv[2]
    if not os.path.exists(text_in):
        print("Error: '%s' does not exist!" % text_in)
        exit(1)

    with open(text_in, 'r', encoding='utf-8') as fin:
        with open(text_out, 'w', encoding='utf-8') as fout:
            for line in fin:
                sentence = ' '.join(line.split()[1:])
                fout.write('%s\n' % sentence)

    print("`%s` ends successfully." % sys.argv[0])
    exit(0)

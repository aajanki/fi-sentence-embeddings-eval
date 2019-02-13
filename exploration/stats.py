import os.path
import re
import pandas as pd


def main():
    datadir = '../data/UD_Finnish-TDT'
    train_filename = os.path.join(datadir, 'fi_tdt-ud-train.conllu')
    dev_filename = os.path.join(datadir, 'fi_tdt-ud-dev.conllu')
    test_filename = os.path.join(datadir, 'fi_tdt-ud-test.conllu')

    df_train = parse_conllu(open(train_filename))
    df_dev = parse_conllu(open(dev_filename))
    df_test = parse_conllu(open(test_filename))

    print(sentence_counts(df_train, df_dev, df_test).to_string())


def sentence_counts(df_train, df_dev, df_test):
    return (
        pd.concat((
            df_train.groupby('source_type').size(),
            df_dev.groupby('source_type').size(),
            df_test.groupby('source_type').size()
        ), axis=1, sort=True)
        .fillna(0)
        .astype(int)
        .rename(columns={0: 'train', 1: 'dev', 2: 'test'})
    )


def parse_conllu(f):
    text_re = re.compile(r'^# text += +(.*)')
    source_re = re.compile(r'^# sent_id += +([a-zA-Z]+)([0-9]+)\.([0-9]+)')
    
    sentence_data = []
    text = ''
    for line in f.readlines():
        if line.startswith('#'):
            text_match = text_re.match(line)
            source_match = source_re.match(line)

            if text_match:
                text = text_match.group(1)
            elif source_match:
                source_type = source_match.group(1)
                source = source_type + source_match.group(2)
                ordinal = int(source_match.group(3))
                sentence_data.append((text, source_type, source, ordinal))
                text = ''

    return pd.DataFrame(sentence_data,
                        columns=['sentence', 'source_type',
                                 'source', 'ordinal'])


if __name__ == '__main__':
    main()

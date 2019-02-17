import os.path
import re
import pandas as pd


def print_source_percentages():
    df_train, df_test = load_UD('../data/UD_Finnish-TDT')

    print('Class proportions')
    print(source_type_percentages(df_train, df_test))


def load_UD(datadir, return_dev_set=False):
    train_filename = os.path.join(datadir, 'fi_tdt-ud-train.conllu')
    if return_dev_set:
        test_filename = os.path.join(datadir, 'fi_tdt-ud-dev.conllu')
    else:
        test_filename = os.path.join(datadir, 'fi_tdt-ud-test.conllu')

    selected_sources = ('b', 'e', 'j', 's', 't', 'u', 'w', 'wn')

    df_train = parse_conllu(open(train_filename))
    df_train = df_train[df_train['source_type'].isin(selected_sources)]

    df_test = parse_conllu(open(test_filename))
    df_test = df_test[df_test['source_type'].isin(selected_sources)]

    return (df_train, df_test)


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


def source_type_percentages(df_train, df_test):
    return pd.DataFrame({
        'train': df_train.groupby('source_type').size() / df_train.shape[0],
        'test': df_test.groupby('source_type').size() / df_test.shape[0]
    }) * 100


if __name__ == '__main__':
    print_source_percentages()

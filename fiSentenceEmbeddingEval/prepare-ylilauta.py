import re
import os.path
import numpy as np
import pandas as pd
import xml.dom.minidom
from xml.parsers.expat import ExpatError


def main():
    print('Preprocessing ylilauta dataset, this will take while...')
    
    data_dir = 'data/ylilauta'
    n_train = 5000
    n_dev = 500
    n_test = 2000

    np.random.seed(9999)
    
    df = load(os.path.join(data_dir, 'ylilauta_20150304.vrt'))
    df.to_csv(os.path.join(data_dir, 'sentences.tab'), sep='\t', index=False)

    print(df[['section', 'sentence']].groupby('section').count())

    ignore_sections = ['joulukalenteri', 'palaute', 'int', 'rule34',
                       'hiekkalaatikko']
    df = df[~df['section'].isin(ignore_sections) & (df['num_tokens'] > 4)]

    train, dev, test = generate_samples(df, n_train, n_dev, n_test)
    train.to_csv(os.path.join(data_dir, 'train.tab'), sep='\t', index=False)
    dev.to_csv(os.path.join(data_dir, 'dev.tab'), sep='\t', index=False)
    test.to_csv(os.path.join(data_dir, 'test.tab'), sep='\t', index=False)


def load(filename):
    sentences = []
    raw = open(filename).read()
    for m in re.finditer(r'<text .*?</text>', raw, re.DOTALL):
        document = parse_vrt_object(m.group())
        if document:
            sentences.extend(document)

    column_names = ['document_id', 'paragraph_id', 'sentence_id',
                    'section', 'sentence', 'num_tokens']
    return pd.DataFrame(sentences, columns=column_names)


def parse_vrt_object(mtext):
    mtext = re.sub(r'&(?![a-z]{2,4};)', '&amp;', mtext)
        
    try:
        doc = xml.dom.minidom.parseString(mtext)
    except ExpatError as ex:
        print('Warning: Failed to parse')
        print(mtext)
        print(ex)
        return None

    sentences = []
    text_elem = doc.firstChild
    doc_id = text_elem.attributes['id'].value
    section = text_elem.attributes['sec'].value
    for i, para in enumerate(text_elem.getElementsByTagName('paragraph')):
        for j, s in enumerate(para.getElementsByTagName('sentence')):
            tokens = parse_vrt_sentence(s.firstChild.nodeValue)
            tokens = remove_link_token(tokens)
            if tokens:
                text = ' '.join(tokens)
                sentences.append((doc_id, i, j, section, text, len(tokens)))

    return sentences


def parse_vrt_sentence(data):
    return [line.split('\t', 1)[0]
            for line in data.split('\n') if line.strip()]


def generate_samples(df, n_train, n_dev, n_test, pos_frac=0.5):
    n_train_pos = int(n_train * pos_frac)
    n_train_neg = n_train - n_train_pos
    n_dev_pos = int(n_dev * pos_frac)
    n_dev_neg = n_dev - n_dev_pos
    n_test_pos = int(n_test * pos_frac)
    n_test_neg = n_test - n_test_pos
    n_pos = n_train_pos + n_dev_pos + n_test_pos
    n_neg = n_train_neg + n_dev_neg + n_test_neg

    positives = generate_positive_samples(df, n_pos)
    positives['label'] = 1

    negatives = generate_negative_samples(df, n_neg)
    negatives['label'] = -1

    pos_train, pos_dev, pos_test = np.array_split(
        positives, [n_train_pos, n_train_pos + n_dev_pos])

    neg_train, neg_dev, neg_test = np.array_split(
        negatives, [n_train_neg, n_train_neg + n_dev_neg])

    train = (pd.concat([pos_train, neg_train], axis=0)
             .sample(frac=1)
             .reset_index(drop=True))

    dev = (pd.concat([pos_dev, neg_dev], axis=0)
           .sample(frac=1)
           .reset_index(drop=True))

    test = (pd.concat([pos_test, neg_test], axis=0)
            .sample(frac=1)
            .reset_index(drop=True))

    return train, dev, test


def generate_positive_samples(df, n):
    consecutive = []
    for (_, a), (_, b) in zip(df.iterrows(), df[1:].iterrows()):
        if (a['document_id'] == b['document_id'] and
            a['paragraph_id'] == b['paragraph_id'] and
            a['sentence_id'] == b['sentence_id'] - 1):
            consecutive.append({
                'sentence1': a['sentence'],
                'sentence2': b['sentence']
            })

    return pd.DataFrame(consecutive).sample(n).reset_index(drop=True)


def generate_negative_samples(df, n):
    samples = []
    for (_, row) in df.sample(n).iterrows():
        doc_id = row['document_id']
        neg = df[df['document_id'] != doc_id].sample(1)
        samples.append({
            'sentence1': row['sentence'],
            'sentence2': neg['sentence'].iloc[0]
        })

    return pd.DataFrame(samples).sample(frac=1)


def remove_link_token(tokens):
    if tokens and tokens[0].startswith('>>'):
        return tokens[1:]
    else:
        return tokens


if __name__ == '__main__':
    main()

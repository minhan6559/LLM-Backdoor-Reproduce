#!/usr/bin/env python
import os
from datasets import load_dataset
import csv


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'SentimentAnalysis', 'SST-2')
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset('glue', 'sst2')

    def write_tsv(split_name, out_name):
        split = ds[split_name]
        path = os.path.join(out_dir, out_name)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['sentence', 'label'], delimiter='\t')
            writer.writeheader()
            for ex in split:
                sent = ex['sentence']
                label = int(ex['label']) if 'label' in ex and ex['label'] is not None else 0
                writer.writerow({'sentence': sent, 'label': label})

    # GLUE SST-2 uses 'train' and 'validation' (no labeled public test)
    write_tsv('train', 'train.tsv')
    write_tsv('validation', 'dev.tsv')
    # Create a labeled test.tsv by duplicating validation to satisfy the loader
    write_tsv('validation', 'test.tsv')

    print(f"Wrote SST-2 train/dev/test to: {out_dir}")


if __name__ == '__main__':
    main()

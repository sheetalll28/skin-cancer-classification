#!/usr/bin/env python3
import argparse
import os
import json
from datasets import load_dataset
from tqdm import tqdm

def save_split(split_name, out_dir, limit=None):
    ds = load_dataset('cnn_dailymail', '3.0.0', split=split_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{split_name}.jsonl')
    written = 0
    with open(out_path, 'w', encoding='utf-8') as fout:
        for i, item in enumerate(tqdm(ds, desc=f'Writing {split_name}')):
            if limit and written >= limit:
                break
            article = item.get('article', '').strip()
            highlights = item.get('highlights', '').strip()
            record = {'id': f'{split_name}-{i}', 'article': article, 'highlights': highlights}
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            written += 1
    print(f'Wrote {written} examples to {out_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='data/cnn_dm', help='output directory')
    parser.add_argument('--limit', type=int, default=None, help='max examples per split (optional)')
    args = parser.parse_args()
    # Download and save each split
    for split in ['train', 'validation', 'test']:
        save_split(split, args.out_dir, limit=args.limit)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Read data/cnn_dm/{train,validation,test}.jsonl (each line has 'article' and 'highlights'),
split articles into sentences, produce greedy oracle extractive labels, and write
data/cnn_dm/{train_labeled.jsonl,validation_labeled.jsonl,test_labeled.jsonl}.

Usage:
  python src/prep_and_label.py --limit 2000

Options:
  --limit   : Optional int, limit number of examples processed per split (useful for quick iteration).
  --max_sent_per_article : truncate very long articles to this many sentences (default 50).
  --k       : number of sentences to select for oracle (default 3).
"""
import argparse
import json
import os
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer

# ensure punkt tokenizer is available
nltk.download('punkt', quiet=True)

def sentence_split(text):
    return nltk.tokenize.sent_tokenize(text)

def greedy_oracle_labels(sentences, reference_summary, max_sentences=3):
    """Greedy select up to max_sentences sentences that maximize ROUGE-L with reference."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    chosen = []
    remaining = list(range(len(sentences)))
    best_score = 0.0
    labels = [0]*len(sentences)
    for _ in range(min(max_sentences, len(sentences))):
        best_i=None
        best_i_score = best_score
        for i in remaining:
            cand = ' '.join([sentences[j] for j in chosen + [i]])
            score = scorer.score(reference_summary, cand)['rougeL'].fmeasure
            if score > best_i_score:
                best_i_score = score
                best_i = i
        if best_i is None:
            break
        chosen.append(best_i)
        remaining.remove(best_i)
        best_score = best_i_score
    for i in chosen:
        labels[i] = 1
    return labels

def process_split(in_path, out_path, limit=None, max_sent_per_article=50, k=3):
    print(f'Processing {in_path} -> {out_path}')
    written = 0
    with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(tqdm(fin)):
            if limit and written >= limit:
                break
            try:
                item = json.loads(line)
            except Exception:
                continue
            article = item.get('article', '').strip()
            summary = item.get('highlights', '').strip()
            if not article:
                continue
            sents = sentence_split(article)
            if len(sents) == 0:
                continue
            # optionally truncate to a manageable number of sentences
            if len(sents) > max_sent_per_article:
                sents = sents[:max_sent_per_article]
            labels = greedy_oracle_labels(sents, summary, max_sentences=k)
            out = {
                'id': item.get('id', f'{os.path.basename(in_path)}-{i}'),
                'sentences': sents,
                'labels': labels,
                'summary': summary
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
            written += 1
    print(f'Wrote {written} examples to {out_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/cnn_dm', help='where input jsonl files are')
    parser.add_argument('--limit', type=int, default=None, help='max examples per split (optional)')
    parser.add_argument('--max_sent_per_article', type=int, default=50)
    parser.add_argument('--k', type=int, default=3, help='number of oracle sentences to select')
    args = parser.parse_args()

    splits = [('train.jsonl', 'train_labeled.jsonl'),
              ('validation.jsonl', 'validation_labeled.jsonl'),
              ('test.jsonl', 'test_labeled.jsonl')]

    for inp, outp in splits:
        in_path = os.path.join(args.data_dir, inp)
        out_path = os.path.join(args.data_dir, outp)
        if not os.path.exists(in_path):
            print('Skipping (missing):', in_path)
            continue
        process_split(in_path, out_path, limit=args.limit,
                      max_sent_per_article=args.max_sent_per_article, k=args.k)

if __name__ == '__main__':
    main()
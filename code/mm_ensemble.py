
from pathlib import Path
import os


data_path = Path('../data/')

if not os.path.exists('../data'):
    data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'
evaluation_path = data_path / 'validation'
test_path = data_path / 'test'

SAMPLE_SUBMISSION = data_path / 'sample_submission.csv'


SUBMISSION_FILENAME = "submission_ensemble.csv"

import pandas as pd
import math
import random
import numpy as np
from random import seed

seed(2023)
ss_path = str(SAMPLE_SUBMISSION)
idx = pd.read_csv(ss_path).output_id.values

#####

def remove_within_dups_from_row(row):
    preds = row.split(' ')
    return ' '.join([
        pred for i, pred in enumerate(preds) if preds.index(pred) == i
    ])

def remove_within_dups(sub):
    sub_new = sub.copy()
    sub_new.output = sub.output.apply(remove_within_dups_from_row)
    return sub_new


def get_sub(nick):

    csv_filename = f'{nick}'
    if ".csv" not in csv_filename:
        csv_filename = csv_filename + ".csv"

    sub = pd.read_csv(f'{nick}')
    if 'output_id' in sub.columns:
        sub = sub.set_index('output_id')
    sub.index.name = 'output_id'
    if not isinstance(sub.index[0], str):
        sub.index = pd.read_csv(ss_path).output_id.values
    sub = sub.fillna('')
    for c in ['|0|', '|0| |0| |0|', '|123|456|789|']:
        sub = sub.replace(c, '')
    sub['output'] = sub['output'].str.strip()
    sub = remove_within_dups(sub)
    return sub

def combine(subs, first_only, order_by_confidence):
    if order_by_confidence:
        f = lambda df: df[df['output'] != ''].shape[0]
        subs = sorted(subs, key=f)
    submission = pd.DataFrame(columns=['output'])
    submission.index.name = 'output_id'
    for i, df in enumerate(subs):
        output_ids = set(df[df['output'] != ''].index)
        for output_id in output_ids:
            current = df.loc[output_id, 'output']
            if first_only and i != len(subs) - 1:
                current = current.split(' ')[0]
            if output_id in submission.index:
                submission.loc[output_id, 'output'] = submission.loc[output_id, 'output'] + ' ' + current
            else:
                submission.loc[output_id, 'output'] = current 
    return submission

def top_three(preds_str):
    preds, counts = [], []
    for pred in preds_str.split(' '):
        if pred in preds:
            counts[preds.index(pred)] += 1
        else:
            preds.append(pred)
            counts.append(1)
    preds_capped = []
    p, c = [p for p in preds], [c for c in counts]
    while len(preds_capped) < 3 and len(p) > 0:
        mxcount = max(c)
        idx = c.index(mxcount)
        preds_capped.append(p[idx])
        p = [x for i, x in enumerate(p) if i != idx]
        c = [x for i, x in enumerate(c) if i != idx]
    return ' '.join(preds_capped)

def main(nicks, first_only, majority_voting, order_by_confidence):
    subs = [get_sub(nick) for nick in nicks]
    sub = combine(subs, first_only=first_only, order_by_confidence=order_by_confidence)
    submission = sub.copy()
    if majority_voting:
        submission.output = submission.output.apply(top_three)
    else:
        submission = remove_within_dups(submission)
        f = lambda s: ' '.join(s.split(' ')[:3])
        submission.output = submission.output.apply(f)
    
    return submission


lines =  [ 'submission_top3_dsl8.csv', 
            'submission_top3_dsl5_r1.csv',  'submission_top3_dsl5_r2.csv', 
            'submission_top3_zoltan.csv']

lines +=  [ 'submission_crop5.csv',  'submission_mosai5.csv', 
                'submission_top1.csv', 'submission_top10.csv',
                'submission_top8_tree.csv']    

nicks = lines[:]

submission = main(
    nicks=nicks,
    first_only=True,
    majority_voting=False,
    order_by_confidence=True
)

#####
sub = submission.copy()

final_submission = sub.copy()
final_submission.to_csv(SUBMISSION_FILENAME)

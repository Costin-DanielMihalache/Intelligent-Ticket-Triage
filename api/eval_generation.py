"""
Generation eval: samples real tickets, runs the full pipeline, and writes a
CSV for manual grading. There is no ground truth for "is this a good reply",
so this produces the material for a human pass, not a pass/fail check.

Usage:
    python eval_generation.py --n 25 --seed 42 --out eval_results.csv
"""
import argparse
import csv
import sys
import pandas as pd
from models import route_ticket, BASE_DIR

def sample_tickets(n,seed):
    df=pd.read_csv(BASE_DIR/"data"/"df_resolved.csv")
    # Stratified by queue so the sample isn't dominated by the two biggest
    # categories (Technical Support, Product Support cover >45% of the data)
    per_queue=max(1,n//df['queue'].nunique())
    parts=[g.sample(min(len(g),per_queue),random_state=seed) for _,g in df.groupby('queue')]
    sample=pd.concat(parts)
    return sample.sample(frac=1,random_state=seed).head(n).reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=25)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="eval_results.csv")
    args=ap.parse_args()

    sample=sample_tickets(args.n,args.seed)
    rows=[]
    for i,row in sample.iterrows():
        ticket_text=str(row['body'])
        print(f"[{i+1}/{len(sample)}] routing ticket (true queue: {row['queue']})...",file=sys.stderr)
        result=route_ticket(ticket_text)
        rows.append({
            'id':i,
            'true_queue':row['queue'],
            'ticket_text':ticket_text,
            'predicted_category':result['category'],
            'confidence':round(result['confidence'],3),
            'decision':result['decision'],
            'generated_response':result['generated_response'] or '',
            # left blank for manual grading:
            'relevance_1to5':'',
            'hallucination_yn':'',
            'notes':''
        })

    with open(args.out,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}",file=sys.stderr)

if __name__=="__main__":
    main()
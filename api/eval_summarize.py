"""
Summarizes a manually-graded eval CSV produced by eval_generation.py.
Usage: python eval_summarize.py eval_results.csv
"""
import sys
import pandas as pd

def main():
    path=sys.argv[1] if len(sys.argv)>1 else "eval_results.csv"
    df=pd.read_csv(path)

    print(f"Total tickets: {len(df)}")
    print(f"Decisions: {df['decision'].value_counts().to_dict()}")

    graded=df[df['relevance_1to5'].notna() & (df['relevance_1to5']!='')]
    if graded.empty:
        print("\nNo graded rows yet - fill in relevance_1to5 (1-5) and hallucination_yn (y/n) first.")
        return

    print(f"\nGraded rows: {len(graded)}/{len(df)}")
    print(f"Mean relevance: {graded['relevance_1to5'].astype(float).mean():.2f} / 5")
    halluc=graded['hallucination_yn'].astype(str).str.lower().eq('y')
    print(f"Hallucination rate: {halluc.mean()*100:.0f}% ({halluc.sum()}/{len(graded)})")

    resolved=graded[graded['decision']=='AUTO_RESOLVE']
    if not resolved.empty:
        print(f"\nAUTO_RESOLVE only - mean relevance: {resolved['relevance_1to5'].astype(float).mean():.2f} / 5")

if __name__=="__main__":
    main()
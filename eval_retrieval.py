"""
Retrieval evaluation for the CBU knowledge base.

Each case is a question and the document that should answer it. Reports how
often the right document is the top hit and how often it appears at all, so
changes to chunking, embeddings or reranking can be compared instead of
judged by feel.

    python eval_retrieval.py
"""

import sys

sys.path.insert(0, ".")

CASES: list[tuple[str, str]] = [
    ("What is a variance and who approves it?", "Authorizations and Variances"),
    ("What classes do I take in year one fall of BASAI?", "CBU_BASAI_26-27"),
    ("Can I transfer credits from a community college?", "Transfer"),
    ("What general education requirements do I need?", "GE (General Education)"),
    ("How many internship hours do I need to graduate?", "Non-Course Requirements"),
    ("What math class do I start with as a freshman?", "Math Placement"),
    ("Does AP Calculus credit count?", "AP-Calculus-Policy"),
    ("What is in the cybersecurity concentration?", "Cybersecurity_CONC"),
    ("How long is the MSAI program?", "MSAI_CBU Program"),
    ("What is the cross cultural requirement?", "Cross Cultural"),
    ("What changed in the program catalog?", "Program & Catalog Changes"),
    ("What is a personal advising sheet?", "Personal Advising"),
    ("Tell me about the machine learning concentration", "ML_CONC"),
    ("What is the BASAI program?", "BASAI Program Overview"),
    ("Who do I contact in the CSDS department?", "CSDS"),
]


def main() -> int:
    import server

    server.build_lexical_index()
    top1 = hit = 0
    for question, expected in CASES:
        _ctx, sources, relevant, score = server.retrieve(question)
        names = [s.split("/")[-1] for s in sources]
        first_ok = bool(names) and expected.lower() in names[0].lower()
        any_ok = any(expected.lower() in n.lower() for n in names)
        top1 += first_ok
        hit += any_ok
        mark = "top1" if first_ok else ("hit " if any_ok else "MISS")
        print(f"  {mark}  score={score:6.2f}  {question[:44]:<44} -> {names[0][:34] if names else '-'}")
    total = len(CASES)
    print(f"\n  top-1 {top1}/{total} ({100*top1/total:.0f}%)   "
          f"in-context {hit}/{total} ({100*hit/total:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

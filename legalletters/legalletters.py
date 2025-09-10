"""
End-to-end pipeline for the legal letters case study using Judgeval.

Features:
- Load CSV of (facts, rough, final)
- Build Judgeval Examples pairing facts + rough + final
- Optionally generate NEW candidate drafts with an LLM
- Run batch evals (blocking) and/or CI fail-fast assertions
- Optional: online evals hooked to the generation step

Assumptions:
  CSV columns: case_id, visa_type, beneficiary_data, recommender_data, rough_draft, final_draft
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Optional

import pandas as pd
from openai import OpenAI

from judgeval.tracer import Tracer, wrap
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer
# (Add more scorers here if you want)

# -------------------- CONFIG --------------------
PROJECT_NAME = "LegalLetters"
MODEL_GENERATOR = "gpt-4.1"   # generation model (candidate drafts)
MODEL_JUDGE     = "gpt-4.1"   # judge model (for scorers)
CSV_PATH        = "/Users/joshualum/Desktop/judgement/legalletters/legal_letters.csv"  

# Score thresholds (tune once you see baseline results)
THRESH_RELEVANCY   = 0.70
THRESH_FAITHFUL    = 0.70

# If you don’t want to generate candidates, set to False
GENERATE_CANDIDATES = True

# -------------------- DATA TYPES --------------------
@dataclass
class LetterRecord:
    case_id: str
    visa_type: Optional[str]
    beneficiary_data: str
    recommender_data: str
    rough_draft: str
    final_draft: str

# -------------------- IO / WRANGLING --------------------
def load_records(csv_path: str) -> List[LetterRecord]:
    df = pd.read_csv(csv_path)
    # Basic sanity checks
    required = ["beneficiary_data", "recommender_data", "rough_draft", "final_draft"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Fill optionals safely
    if "case_id" not in df.columns: df["case_id"] = [f"case_{i}" for i in range(len(df))]
    if "visa_type" not in df.columns: df["visa_type"] = None

    # Minimal cleaning 
    for c in ["beneficiary_data","recommender_data","rough_draft","final_draft"]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    records: List[LetterRecord] = []
    for _, r in df.iterrows():
        if not r["rough_draft"] or not r["final_draft"]:
            continue  # drop malformed rows
        records.append(LetterRecord(
            case_id=str(r["case_id"]),
            visa_type=(None if pd.isna(r["visa_type"]) else str(r["visa_type"])),
            beneficiary_data=r["beneficiary_data"],
            recommender_data=r["recommender_data"],
            rough_draft=r["rough_draft"],
            final_draft=r["final_draft"]
        ))
    return records

def build_input_from_facts(beneficiary: str, recommender: str, visa_type: Optional[str]) -> str:
    # This “input” is shown to scorers like AnswerRelevancy (acts as the task context)
    header = f"Visa Type: {visa_type}\n" if visa_type else ""
    return (
        f"{header}"
        f"Beneficiary Facts:\n{beneficiary}\n\n"
        f"Recommender Facts:\n{recommender}\n"
    )

def build_retrieval_context(beneficiary: str, recommender: str) -> List[str]:
    # FaithfulnessScorer will check claims against these
    return [beneficiary, recommender]

# -------------------- (OPTIONAL) GENERATION + ONLINE EVALS --------------------
# If you want to instrument generation with tracing + async evals:
tracer = Tracer(project_name=PROJECT_NAME)
client_gen = wrap(OpenAI())  # wraps LLM for telemetry

@tracer.observe(span_type="tool")
def format_prompt(visa_type: Optional[str], beneficiary: str, recommender: str) -> str:
    target = f" for {visa_type}" if visa_type else ""
    return (
        "Write a professional immigration recommendation letter"
        f"{target}, using only the facts below.\n\n"
        f"{build_input_from_facts(beneficiary, recommender, visa_type)}\n"
        "The letter should be factual, persuasive, and legally appropriate."
    )

class DraftAgent:
    def __init__(self, name: str = "DraftAgent"):
        self.name = name

    @tracer.agent(identifier="name")
    @tracer.observe(span_type="function")
    def generate(self, visa_type: Optional[str], beneficiary: str, recommender: str) -> str:
        prompt = format_prompt(visa_type, beneficiary, recommender)
        resp = client_gen.chat.completions.create(
            model=MODEL_GENERATOR,
            messages=[{"role": "user", "content": prompt}],
            timeout=60
        )
        draft = (resp.choices[0].message.content or "").strip()

        # Attach ONLINE evals (non-blocking) to the trace so you see it in the UI
        tracer.async_evaluate(
            scorer=AnswerRelevancyScorer(threshold=THRESH_RELEVANCY),
            example=Example(
                input=prompt,
                actual_output=draft
            ),
            model=MODEL_JUDGE
        )
        # Faithfulness only runs if we pass context:
        tracer.async_evaluate(
            scorer=FaithfulnessScorer(threshold=THRESH_FAITHFUL),
            example=Example(
                input=prompt,
                actual_output=draft,
                retrieval_context=build_retrieval_context(beneficiary, recommender)
            ),
            model=MODEL_JUDGE
        )
        return draft

# -------------------- BUILD EXAMPLES --------------------
def examples_from_records_for_candidate(records: Iterable[LetterRecord], candidate_texts: Iterable[str]) -> List[Example]:
    """
    Pair facts + candidate (actual_output) + lawyer final (expected_output).
    """
    examples: List[Example] = []
    for rec, cand in zip(records, candidate_texts):
        ex = Example(
            input=build_input_from_facts(rec.beneficiary_data, rec.recommender_data, rec.visa_type),
            actual_output=cand,
            expected_output=rec.final_draft,
            retrieval_context=build_retrieval_context(rec.beneficiary_data, rec.recommender_data),
            additional_metadata={"case_id": rec.case_id, "visa_type": rec.visa_type or "NA"}
        )
        examples.append(ex)
    return examples

def examples_from_records_for_rough_baseline(records: Iterable[LetterRecord]) -> List[Example]:
    """
    Baseline: evaluate the original rough drafts against the same gold/context.
    """
    examples: List[Example] = []
    for rec in records:
        ex = Example(
            input=build_input_from_facts(rec.beneficiary_data, rec.recommender_data, rec.visa_type),
            actual_output=rec.rough_draft,
            expected_output=rec.final_draft,
            retrieval_context=build_retrieval_context(rec.beneficiary_data, rec.recommender_data),
            additional_metadata={"case_id": rec.case_id, "visa_type": rec.visa_type or "NA", "baseline": True}
        )
        examples.append(ex)
    return examples

# -------------------- EVALUATION (BATCH / CI) --------------------
def run_batch_eval(examples: List[Example], fail_fast: bool = False):
    client = JudgmentClient()
    scorers = [
        AnswerRelevancyScorer(threshold=THRESH_RELEVANCY),
        FaithfulnessScorer(threshold=THRESH_FAITHFUL),
        # Add more scorers here (e.g., a custom CompletenessScorer) if available
    ]

    if fail_fast:
        # CI-style: raise AssertionError if any score < threshold
        client.assert_test(
            examples=examples,
            scorers=scorers,
            model=MODEL_JUDGE,
            project_name=PROJECT_NAME
        )
        print("CI gate passed ✅")
    else:
        results = client.run_evaluation(
            examples=examples,
            scorers=scorers,
            model=MODEL_JUDGE
        )
        print("Batch results:")
        print(results)

# -------------------- MAIN --------------------
def main():
    # 1) Load data
    records = load_records(CSV_PATH)
    if not records:
        raise RuntimeError("No valid records loaded from CSV.")

    # 2) Option A: Evaluate the existing ROUGH drafts as a baseline
    rough_examples = examples_from_records_for_rough_baseline(records)
    print(f"Evaluating ROUGH baseline on {len(rough_examples)} examples…")
    run_batch_eval(rough_examples, fail_fast=False)

    # 3) Option B: Generate NEW candidate drafts (instrumented) and evaluate
    if GENERATE_CANDIDATES:
        agent = DraftAgent("GenV1")
        candidates: List[str] = []
        for rec in records:
            cand = agent.generate(rec.visa_type, rec.beneficiary_data, rec.recommender_data)
            candidates.append(cand)

        candidate_examples = examples_from_records_for_candidate(records, candidates)
        print(f"Evaluating NEW candidates on {len(candidate_examples)} examples…")
        run_batch_eval(candidate_examples, fail_fast=False)

        # Optional CI gate:
        # run_batch_eval(candidate_examples, fail_fast=True)

if __name__ == "__main__":
    main()

# 1. Similarity-to-Final Scorer

# Compute semantic similarity or edit distance between the candidate draft and the final draft.

# Higher = closer to lawyer edits.

# 2. Anchored Pairwise Scorer

# In pairwise mode, give the judge both candidates + the final draft and ask:
# “Which candidate is closer to the final draft, and why?”

# 3. Fine-tuned Evaluator

# Actually fine-tune a model on (rough, final) pairs so it learns lawyer preferences directly.

# That’s what the report hints at with “distill the legal expertise into GPT-4.1.”
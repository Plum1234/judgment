import csv, random, textwrap, os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# ------------- CONFIG (tweak freely) -------------
N_ROWS = 200
OUT_PATH = "legal_letters.csv"
RANDOM_SEED = 13

VISA_TYPES = ["EB-1A", "O-1A", "NIW"]  # common categories
FIELDS = ["computer vision", "computational biology", "robotics", "NLP", "theoretical CS", "HCI", "cybersecurity"]

# Noise knobs for rough drafts (probabilities)
P_OMIT_KEY_FACT = 0.30      # omit one key fact
P_WRONG_COUNT = 0.20        # e.g., say 12 pubs instead of 22
P_HALLUCINATION = 0.12      # invent award/venue that’s not in facts
P_WEAK_VISA_MENTION = 0.50  # never name the visa or name it vaguely
P_STYLE_WEAK = 0.60         # generic, repetitive tone
P_SECTION_MISS = 0.25       # miss a required section (e.g., closing or recommender credentials)

# Count ranges
PUB_RANGE = (8, 80)                   # publications
CITATION_RANGE = (200, 5000)          # citations
AWARD_POOL = [
    "ACM Best Paper Award", "IEEE Fellow", "Sloan Fellowship",
    "NSF CAREER Award", "Turing Award nomination", "AAAI Fellow",
    "ACL Best Paper Award", "NeurIPS Outstanding Paper", "ICLR Spotlight"
]
VENUE_POOL = ["NeurIPS", "ICML", "CVPR", "ACL", "EMNLP", "KDD", "AAAI", "ICLR"]
ORG_POOL = ["MIT", "Stanford", "CMU", "Berkeley", "Google DeepMind", "Microsoft Research", "OpenAI", "Caltech"]
TITLE_POOL = ["Professor", "Associate Professor", "Research Scientist", "Principal Scientist", "Chair", "Director"]

# ------------- DATA MODEL -------------
@dataclass
class Person:
    full_name: str
    affiliation: str
    title: str

@dataclass
class BeneficiaryFacts:
    name: str
    field: str
    pubs: int
    citations: int
    key_awards: List[str]
    key_venues: List[str]

@dataclass
class Record:
    case_id: str
    visa_type: str
    beneficiary_data: str
    recommender_data: str
    rough_draft: str
    final_draft: str

# ------------- HELPERS -------------
def choice_no_replacement(pool: List[str], k: int) -> List[str]:
    return random.sample(pool, k=min(k, len(pool)))

def mk_beneficiary(i: int) -> BeneficiaryFacts:
    field = random.choice(FIELDS)
    pubs = random.randint(*PUB_RANGE)
    citations = random.randint(*CITATION_RANGE)
    awards = choice_no_replacement(AWARD_POOL, random.randint(0, 2))
    venues = choice_no_replacement(VENUE_POOL, random.randint(1, 3))
    return BeneficiaryFacts(
        name=f"Dr. Alex {i}",
        field=field,
        pubs=pubs,
        citations=citations,
        key_awards=awards,
        key_venues=venues
    )

def mk_recommender(i: int) -> Person:
    aff = random.choice(ORG_POOL)
    title = random.choice(TITLE_POOL)
    return Person(
        full_name=f"Dr. Jordan {i}",
        affiliation=aff,
        title=title
    )

def facts_to_string(b: BeneficiaryFacts) -> str:
    awards = (", ".join(b.key_awards)) if b.key_awards else "—"
    venues = ", ".join(b.key_venues)
    return textwrap.fill(
        f"{b.name} works in {b.field}. Publications: {b.pubs}. "
        f"Citations: {b.citations}. Awards: {awards}. "
        f"Key venues: {venues}.", width=120
    )

def recommender_to_string(r: Person) -> str:
    return f"{r.full_name}, {r.title}, {r.affiliation}"

def legal_intro(visa_type: str, r: Person, b: BeneficiaryFacts) -> str:
    return (f"I am honored to recommend {b.name} for the {visa_type} visa category. "
            f"As {r.title} at {r.affiliation}, I have closely followed {b.name}'s work in {b.field}.")

def legal_evidence(b: BeneficiaryFacts) -> str:
    parts = [f"{b.name} has authored {b.pubs} peer-reviewed publications with approximately {b.citations} citations."]
    if b.key_awards:
        parts.append(f"Notably, {b.name} received {', '.join(b.key_awards)}.")
    parts.append(f"{b.name}'s research appears at {', '.join(b.key_venues)} and is widely recognized in {b.field}.")
    return " ".join(parts)

def legal_closing(b: BeneficiaryFacts) -> str:
    return (f"In summary, {b.name} demonstrates sustained national and international acclaim. "
            f"I strongly endorse this petition and am available for any additional information.")

def mk_final_draft(visa_type: str, r: Person, b: BeneficiaryFacts) -> str:
    # A crisp, persuasive, factual final draft
    paragraphs = [
        legal_intro(visa_type, r, b),
        legal_evidence(b),
        legal_closing(b)
    ]
    return "\n\n".join(textwrap.fill(p, width=110) for p in paragraphs)

def maybe_wrong_count(true_val: int) -> int:
    # Off-by-5..20-ish error, sometimes down, sometimes up
    delta = random.randint(5, 20)
    return max(1, true_val + random.choice([-delta, delta]))

def mk_rough_draft(visa_type: str, r: Person, b: BeneficiaryFacts) -> str:
    # Start from a weaker template, then inject noise
    intro = f"I am writing to recommend {b.name}. {b.name} is a talented researcher in {b.field}."
    evidence = f"They have many publications and their work is known at venues like {', '.join(b.key_venues)}."
    closing = "I believe they are qualified for the visa."

    chunks = [intro, evidence, closing]

    # Noise injections
    if random.random() < P_WEAK_VISA_MENTION:
        # remove explicit visa mention
        chunks[-1] = "I believe they are highly qualified."  # weaker, vague
    else:
        chunks[-1] = f"I believe they are qualified for the {visa_type} visa."

    if random.random() < P_OMIT_KEY_FACT:
        # drop a key fact (e.g., citations or awards)
        if b.key_awards and random.random() < 0.7:
            # replace evidence mentioning awards with generic
            evidence = f"They have many publications and are recognized in their field."
        else:
            evidence = f"Their work is recognized at top venues."
        chunks[1] = evidence

    if random.random() < P_WRONG_COUNT:
        wrong_pubs = maybe_wrong_count(b.pubs)
        chunks[1] += f" They have around {wrong_pubs} publications."
    else:
        chunks[1] += f" They have around {b.pubs} publications."

    if random.random() < P_HALLUCINATION:
        fake_award = random.choice(["Best Innovator 2022", "Global Genius Prize", "World AI Medal"])
        chunks[1] += f" They also received the {fake_award}."

    if random.random() < P_STYLE_WEAK:
        chunks.insert(0, "I am pleased to write this letter. It is my pleasure to write this letter.")
    if random.random() < P_SECTION_MISS:
        # Remove either intro or closing randomly
        if random.random() < 0.5 and len(chunks) > 2:
            chunks.pop(0)  # lose intro
        else:
            chunks = chunks[:-1]  # lose closing

    # Occasionally misname or omit visa entirely to ding relevancy
    if random.random() < 0.08:
        wrong_visa = random.choice([v for v in VISA_TYPES if v != visa_type])
        chunks[-1] = f"I believe they are qualified for the {wrong_visa} visa."

    body = " ".join(chunks)
    # Add one more realistic flaw: passive voice + filler
    if random.random() < 0.35:
        body += " Their contributions have been considered impactful by many, in ways that are thought to be significant."

    return textwrap.fill(body, width=110)

def build_strings(visa_type: str, r: Person, b: BeneficiaryFacts) -> Tuple[str, str, str]:
    beneficiary_str = facts_to_string(b)
    recommender_str = recommender_to_string(r)
    rough = mk_rough_draft(visa_type, r, b)
    final = mk_final_draft(visa_type, r, b)
    return beneficiary_str, recommender_str, rough, final

# ------------- MAIN SYNTHESIS -------------
def make_rows(n_rows: int) -> List[Record]:
    rows: List[Record] = []
    for i in range(1, n_rows + 1):
        visa = random.choice(VISA_TYPES)
        b = mk_beneficiary(i)
        r = mk_recommender(i)

        beneficiary_str, recommender_str, rough, final = build_strings(visa, r, b)
        rows.append(Record(
            case_id=f"case_{i}",
            visa_type=visa,
            beneficiary_data=beneficiary_str,
            recommender_data=recommender_str,
            rough_draft=rough,
            final_draft=final
        ))
    return rows

def write_csv(rows: List[Record], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id","visa_type","beneficiary_data","recommender_data","rough_draft","final_draft"],
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

def summarize(rows: List[Record]):
    vis_counts = {}
    for r in rows:
        vis_counts[r.visa_type] = vis_counts.get(r.visa_type, 0) + 1
    print("✅ Wrote dataset")
    print("Visa distribution:", vis_counts)
    print("Example (first case):")
    first = rows[0]
    print("-" * 80)
    print("Beneficiary Data:\n", first.beneficiary_data)
    print("\nRecommender Data:\n", first.recommender_data)
    print("\nROUGH (imperfect):\n", first.rough_draft)
    print("\nFINAL (polished):\n", first.final_draft[:400], "...")
    print("-" * 80)

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    rows = make_rows(N_ROWS)
    write_csv(rows, OUT_PATH)
    summarize(rows)
    print(f"📄 File saved to: {os.path.abspath(OUT_PATH)}")

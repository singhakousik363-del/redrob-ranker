#!/usr/bin/env python3
"""
Redrob Ranker v2 - With Semantic Embeddings
Author: Kousik Singha
"""
import json, csv, numpy as np
from datetime import date, datetime
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────
GOOD_TITLE_KEYWORDS = ["machine learning","ml engineer","ai engineer","applied scientist","data scientist","nlp engineer","search engineer","ranking engineer","recommendation","research engineer","deep learning","retrieval","senior ai","applied ml","staff ml","staff ai","principal ml","principal ai","founding engineer"]
BAD_TITLE_KEYWORDS = ["marketing","sales","hr ","human resource","finance","accountant","civil","mechanical","graphic","customer support","business analyst","project manager","scrum master","product manager","ux designer","ui designer","recruiter","talent acquisition","operations manager","content writer","seo","social media"]
CONSULTING_COMPANIES = ["tcs","tata consultancy","infosys","wipro","accenture","cognizant","capgemini","mindtree","hcl","tech mahindra","mphasis","hexaware","ltimindtree","l&t infotech"]
CORE_SKILLS = ["python","embeddings","sentence-transformers","vector","faiss","elasticsearch","opensearch","pinecone","weaviate","qdrant","milvus","retrieval","ranking","recommendation","nlp","transformers","bert","llm","fine-tuning","pytorch","tensorflow","ndcg","mrr","search","bm25","rag","dense retrieval","reranking","learning to rank","semantic search","information retrieval"]

# ── LOAD EMBEDDINGS ─────────────────────────────────────────
def load_embeddings():
    if not Path("candidate_embeddings.npy").exists():
        print("WARNING: embeddings not found, run precompute.py first")
        return None, None, None
    print("Loading embeddings...", flush=True)
    jd_emb = np.load("jd_embedding.npy")
    cand_embs = np.load("candidate_embeddings.npy")
    cand_ids = np.load("candidate_ids.npy", allow_pickle=True)
    # Cosine similarity (already normalized, so just dot product)
    similarities = cand_embs @ jd_emb
    # Build lookup dict: candidate_id -> similarity score
    sim_lookup = {cid: float(sim) for cid, sim in zip(cand_ids, similarities)}
    print(f"Embeddings loaded. Shape: {cand_embs.shape}", flush=True)
    return sim_lookup

# ── HONEYPOT DETECTION ───────────────────────────────────────
def is_honeypot(c):
    skills = c["skills"]
    yoe = c["profile"].get("years_of_experience", 0)
    career = c["career_history"]
    sig = c.get("redrob_signals", {})

    # 1. Expert skill with 0 duration
    expert_zero = sum(1 for s in skills if s.get("proficiency") in ["expert","advanced"] and s.get("duration_months",1) == 0)
    if expert_zero >= 3: return True

    # 2. YOE vs career history mismatch
    total_months = sum(ch.get("duration_months", 0) for ch in career)
    if yoe > 2 and total_months > 0 and abs(yoe*12 - total_months) > 36: return True

    # 3. Senior title but very low YOE
    title = c["profile"].get("current_title","").lower()
    if any(w in title for w in ["senior","staff","principal","lead","head","director"]) and yoe < 3: return True

    # 4. Salary min > max
    salary = sig.get("expected_salary_range_inr_lpa", {})
    if salary.get("min",0) > salary.get("max",0) + 1: return True

    # 5. Many expert skills, zero endorsements, no assessments
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 8 and sum(s.get("endorsements",0) for s in expert_skills) == 0 and not sig.get("skill_assessment_scores",{}): return True

    # 6. Connection count 0 but high endorsements (impossible)
    if sig.get("connection_count",1) == 0 and sig.get("endorsements_received",0) > 20: return True

    # 7. Last active before signup (impossible)
    try:
        signup = datetime.strptime(sig.get("signup_date","2020-01-01"), "%Y-%m-%d").date()
        last = datetime.strptime(sig.get("last_active_date","2020-01-01"), "%Y-%m-%d").date()
        if last < signup: return True
    except: pass

    return False

# ── SCORING COMPONENTS ───────────────────────────────────────
def score_title(c):
    title = c["profile"].get("current_title","").lower()
    for bad in BAD_TITLE_KEYWORDS:
        if bad in title: return 0.0
    for good in GOOD_TITLE_KEYWORDS:
        if good in title: return 1.0
    for ch in c["career_history"][-3:]:
        for good in GOOD_TITLE_KEYWORDS:
            if good in ch.get("title","").lower(): return 0.55
    return 0.1

def score_career(c):
    evidence = ["ranking","retrieval","recommendation","search","embeddings","vector","nlp","machine learning","deep learning","production","deployed","scale","faiss","elasticsearch","bert","transformer","llm","a/b test","ndcg","pipeline","model","training","inference","real-time","latency","precision","recall","relevance","rerank","query","collaborative filtering","matrix factorization","lora","qlora","fine-tun","llama","mistral","sentence transformer","opensearch","haystack","weaviate","pinecone","qdrant","milvus","dense","sparse","hybrid","two-tower","cross-encoder","bi-encoder","ann","approximate nearest"]
    text = " ".join(ch.get("description","").lower() for ch in c["career_history"])
    text += " " + c["profile"].get("summary","").lower()
    found = sum(1 for kw in evidence if kw in text)
    score = min(found / 8.0, 1.0)
    companies = [ch.get("company","").lower() for ch in c["career_history"]]
    all_consulting = all(any(cc in co for cc in CONSULTING_COMPANIES) for co in companies if co)
    if all_consulting: score *= 0.3
    return score

def score_skills(c):
    skills = c["skills"]
    if not skills: return 0.0
    assessments = c.get("redrob_signals",{}).get("skill_assessment_scores",{})
    total, found = 0.0, 0
    for s in skills:
        name = s.get("name","").lower()
        if not any(core in name or name in core for core in CORE_SKILLS): continue
        found += 1
        prof = {"beginner":0.2,"intermediate":0.5,"advanced":0.8,"expert":1.0}.get(s.get("proficiency","beginner"),0.2)
        end_trust = min(s.get("endorsements",0)/15.0, 1.0)
        dur_trust = min(s.get("duration_months",0)/24.0, 1.0)
        assess = assessments.get(s.get("name",""),-1)
        assess_trust = assess/100.0 if assess >= 0 else 0.5
        trust = end_trust*0.4 + dur_trust*0.4 + assess_trust*0.2
        total += prof * trust
    return 0.0 if found == 0 else min(total/5.0, 1.0)

def score_experience(c):
    yoe = c["profile"].get("years_of_experience", 0)
    if yoe < 3: return 0.1
    elif yoe < 5: return 0.6
    elif yoe <= 9: return 1.0
    elif yoe <= 12: return 0.8
    else: return 0.6

def score_location(c):
    loc = c["profile"].get("location","").lower()
    country = c["profile"].get("country","").lower()
    relocate = c.get("redrob_signals",{}).get("willing_to_relocate", False)
    if country != "india": return 0.1
    for city in ["pune","noida","delhi","ncr","hyderabad","mumbai","bangalore","bengaluru","gurgaon","gurugram"]:
        if city in loc: return 1.0
    return 0.7 if relocate else 0.4

def score_education(c):
    edu = c.get("education",[])
    if not edu: return 0.3
    tier_map = {"tier_1":1.0,"tier_2":0.8,"tier_3":0.6,"tier_4":0.4,"unknown":0.5}
    best = max(tier_map.get(e.get("tier","unknown"),0.5) for e in edu)
    rel_fields = ["computer","software","information","data","mathematics","statistics","electrical","ai","ml"]
    if any(any(rf in e.get("field_of_study","").lower() for rf in rel_fields) for e in edu):
        best = min(best+0.1, 1.0)
    return best

def score_behavioral(c):
    sig = c.get("redrob_signals",{})
    m = 1.0
    last = sig.get("last_active_date","")
    if last:
        try:
            days = (date.today() - datetime.strptime(last,"%Y-%m-%d").date()).days
            if days > 180: m *= 0.5
            elif days > 90: m *= 0.7
            elif days < 30: m *= 1.1
        except: pass
    if not sig.get("open_to_work_flag", True): m *= 0.7
    else: m *= 1.05
    rr = sig.get("recruiter_response_rate", 0.5)
    if rr < 0.1: m *= 0.6
    elif rr > 0.7: m *= 1.1
    ir = sig.get("interview_completion_rate", 0.5)
    if ir < 0.3: m *= 0.8
    elif ir > 0.8: m *= 1.05
    notice = sig.get("notice_period_days", 60)
    if notice <= 30: m *= 1.05
    elif notice > 90: m *= 0.9
    return max(0.3, min(m, 1.0))

# ── FINAL SCORING ────────────────────────────────────────────
def score_candidate(c, sim_score):
    if is_honeypot(c): return 0.0

    title    = score_title(c)       # 25%
    semantic = sim_score            # 20% ← NEW: embedding similarity
    career   = score_career(c)      # 20%
    skills   = score_skills(c)      # 15%
    exp      = score_experience(c)  # 10%
    loc      = score_location(c)    #  5%
    edu      = score_education(c)   #  5%

    github = c.get("redrob_signals",{}).get("github_activity_score",-1)
    github_score = (github/100.0) if github >= 0 else 0.0

    if title == 0.0:
        base = 0.04  # bad title = near zero regardless
    else:
        base = (
            title    * 0.20 +
            semantic * 0.30 +
            career   * 0.15 +
            skills   * 0.12 +
            exp      * 0.10 +
            loc      * 0.05 +
            edu      * 0.03 +
            github_score * 0.05  # github activity matters for AI engineers
        )

    return round(base * score_behavioral(c), 6)

# ── REASONING ────────────────────────────────────────────────
def generate_reasoning(c, score, sim_score):
    """
    Stage 4 compliant reasoning:
    - Specific facts from profile
    - JD connection
    - Honest concerns where applicable
    - No hallucination
    - Variation across candidates
    """
    p = c["profile"]
    sig = c.get("redrob_signals", {})
    title = p.get("current_title", "Unknown")
    yoe = p.get("years_of_experience", 0)
    company = p.get("current_company", "Unknown")
    location = p.get("location", "Unknown")
    country = p.get("country", "")

    # Get actual relevant skills from profile
    core_found = []
    for s in c.get("skills", []):
        if any(core in s["name"].lower() for core in CORE_SKILLS[:20]):
            prof = s.get("proficiency", "")
            months = s.get("duration_months", 0)
            end = s.get("endorsements", 0)
            core_found.append((s["name"], prof, months, end))

    # Career companies
    companies = [ch.get("company","") for ch in c.get("career_history",[])]
    company_str = " → ".join(companies[:3]) if companies else company

    # Behavioral signals
    rr = sig.get("recruiter_response_rate", 0)
    notice = sig.get("notice_period_days", 60)
    last = sig.get("last_active_date", "")
    otw = sig.get("open_to_work_flag", False)
    github = sig.get("github_activity_score", -1)
    interview_rate = sig.get("interview_completion_rate", 0.5)

    # Top skills string with evidence
    if core_found:
        top_skills = "; ".join(
            f"{name} ({prof}, {months}mo)" 
            for name, prof, months, end in core_found[:3]
        )
    else:
        top_skills = "no directly relevant ML/AI skills found"

    # Concerns
    concerns = []
    if notice > 90:
        concerns.append(f"long notice period ({notice}d)")
    if rr < 0.3:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    if not otw:
        concerns.append("not marked open to work")
    if country.lower() != "india":
        concerns.append("based outside India")
    if github < 0:
        concerns.append("no GitHub linked")
    elif github < 20:
        concerns.append(f"low GitHub activity ({github:.0f})")

    # Build reasoning based on rank tier
    if score >= 0.80:
        base = (f"{title} at {company} ({yoe:.1f} yrs); career: {company_str}. "
                f"Core skills: {top_skills}.")
        if concerns:
            base += f" Concern: {', '.join(concerns[:1])}."
    elif score >= 0.60:
        base = (f"{title}, {yoe:.1f} yrs exp, {location}. "
                f"Relevant skills: {top_skills}. "
                f"Response rate {rr:.0%}, notice {notice}d.")
        if concerns:
            base += f" Gap: {', '.join(concerns[:2])}."
    elif score >= 0.40:
        base = (f"{title} at {company}, {yoe:.1f} yrs. "
                f"Partial fit — {top_skills if core_found else 'limited ML/AI skill overlap with JD'}. "
                f"{'Open to work' if otw else 'Not marked open to work'}.")
    else:
        primary_issue = "title mismatch with Senior AI Engineer role" if not core_found else "weak behavioral signals and limited production ML evidence"
        base = (f"{title} at {company} ({yoe:.1f} yrs). "
                f"Low fit: {primary_issue}. "
                f"Included as rank-{int(score*100)} filler given {top_skills if core_found else 'adjacent skills'}.")

    return base

# ── MAIN ─────────────────────────────────────────────────────
def main():
    sim_lookup = load_embeddings()

    print("Scoring 100K candidates...", flush=True)
    scores = []
    honeypots = 0

    with open("candidates.jsonl") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            c = json.loads(line)
            cid = c["candidate_id"]
            sim = sim_lookup.get(cid, 0.5) if sim_lookup else 0.5
            s = score_candidate(c, sim)
            if s == 0.0: honeypots += 1
            scores.append((cid, s, sim, c))
            if (i+1) % 10000 == 0:
                print(f"  {i+1}/100000", flush=True)

    print(f"Done. Honeypots: {honeypots}")
    scores.sort(key=lambda x: (-x[1], x[0]))
    top100 = scores[:100]

    print("\nTop 10:")
    for i,(cid,s,sim,c) in enumerate(top100[:10]):
        print(f"  {i+1}. {cid} | {c['profile']['current_title'][:38]:<38} | score={s:.4f} | sem={sim:.3f}")

    with open("submission.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id","rank","score","reasoning"])
        for rank,(cid,s,sim,c) in enumerate(top100,1):
            w.writerow([cid, rank, s, generate_reasoning(c,s,sim)])

    print(f"\nHoneypot rate in top 100: {sum(1 for _,s,_,_ in top100 if s==0.0)}/100")
    print("Run: python3 validate_submission.py submission.csv")

if __name__ == "__main__":
    main()

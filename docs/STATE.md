# Project state brief — Amazon ML Challenge 2026, Business Entity Resolution

## Task (short)
3 sources of business records (entity_id, business_name, business_address, country). For every Source-1 (S1)
record, output all matching S2/S3 records (0..many). Metric: macro F0.5 per S1 entity (singleton S1 with empty
prediction = 1.0, any prediction on a singleton = 0.0). Train: US + India with labels. Test: US + India + **France
(no training data, 15% of test S1)**. No external data/APIs; model must be MIT/Apache, <=8B params.
Outputs: output/matching_results.tsv, output/candidate_pairs.tsv (final candidate set the model scores).

## Data facts (train)
- train S1 2,206,821 (US 1.32M, India 0.88M); pool S2 5.03M + S3 5.29M. Test S1 1,732,544 (India 810k, US 663k, France 259k); pool 9.97M.
- 7,638,365 true pairs; avg 3.46 matches per S1; 5.6% S1 singletons; match-count dist 0..11.
- **Every S2/S3 record belongs to at most ONE S1** (unique cand ids in GT). ~26% of S2/S3 are distractors (no S1).
- Country always agrees in positive pairs.
- Distractors are near-copies of an S1: house number shifted (80 vs 89), legal suffix added (Inc/Corp/Holding),
  last token changed (VI -> VIJ, II -> IN), different name at same address.
- Noise in true records: typos, leetspeak (C0ok, 5ervices), legal forms, prefixes (Smt/Shri/M/s/Dr), suffix junk
  ([Company], (ID: 123), #71076), DBA/formerly aliases, domain-style names (allpropertysolutions.com),
  Indic-script names (Devanagari/Kannada/Tamil/Telugu/Gujarati ~7% of pool names), address abbreviations,
  truncated addresses (only house no + city), empty addresses (~4.5% of candidates), state full vs abbrev.
- Transliteration dictionary learned from train GT (word-aligned, 1347 words, 96% test token coverage).

## Pipeline (code: sub/code/business_entity_resolution/src/)
1. prep.py / normalize.py / translit.py: name_norm, name_core (legal/honorific removed), name_alias (dba parts),
   addr_norm (abbrev expanded, state component stripped for us/india only), addr_nums, addr_state.
   Bug found (fixed in code, not yet in cached data/models): a component like "817 38ND" was treated as state ND and dropped.
2. candidates.py: per-country blocking passes, unioned:
   comb (word TF-IDF name+addr, top50), addr (top25), name_ch (char 4-gram nospace name, top12), rev (pool->S1 top5),
   nameaddr (whole name as one token + addr, top20), rev_na (top3), k_name_num (exact name+first number key), k_name (exact name, unique-ish).
   NOTE: TfidfVectorizer max_df=20000 absolute prunes common tokens (hurts big countries, e.g. India common name words).
3. stage0.py: LightGBM ranker on blocking scores + 6 cheap sims -> keep top-20 per S1 (= candidate_pairs.tsv).
4. features.py: ~50 pairwise features (rapidfuzz ratios, token set stats, TF-IDF cos, legal form, house number eq/lev,
   state eq, name frequency, alias best) + s0 competition stats (rank/gap within S1 group and within candidate group).
5. pipeline.py: stage1 LightGBM (255 leaves, 1000 rounds, lr .05) trained on 30% sample of S1 groups, 2-fold grouped OOF;
   stage2 LightGBM adds competition stats of stage-1 probs.
6. decide.py: exclusivity (each candidate -> only its best S1), then expected-F0.5 subset selection per S1
   (E[F_k]=1.25*sum p_i/(k+0.25*(sum p + miss)), empty if prod(1-p) * bias > best) — best rule so far.

## Results so far (validation = 2-fold grouped OOF over ALL 2.2M train S1)
| exp | blocking recall (final cands) | pair P | pair R | macro F0.5 | singleton acc | FP | FN | leaderboard |
|---|---|---|---|---|---|---|---|---|
| E01 stage1 | 95.37% | 99.48 | 92.82 | 0.9679 | 97.19 | 37k | 548k | |
| E01 stage2 | 95.37% | 99.48 | 93.03 | 0.9684 | 97.45 | 37k | 532k | **0.95 (actual)** |
| E02 stage1 (+nameaddr/key passes) | 97.51% | 99.45 | 94.61 | 0.9776 | 97.22 | 40k | 412k | not yet submitted |
Per-country in-domain E02 OOF: US 0.9800 (P 99.53, R 95.15, sing 97.61), India 0.9739 (P 99.34, R 93.79, sing 96.64).
E01 error analysis: FN 532k = 353k blocking failures (India 237k; native-script names with truncated address,
empty addresses) + 179k scoring failures (mostly empty-address candidates, number typos).
FP 37k: 78% distractors (no S1), 22% belong to another S1; causes: same name 17k, house number differs 8k, same address 4k.

## Unseen-country proxy (LOCO: train on US only, evaluate on India)
L01: macro F0.5 0.9512 (P 98.03, R 90.67, singleton 93.80, FP 56k) vs in-domain India 0.9739 (FP 19k).
=> on an unseen country the model is overconfident: false positives triple, singletons suffer.
L02 (monotone constraints) / L03 (per-country rank normalization of features): running.

## Leaderboard gap analysis
E01 validation 0.968 -> leaderboard 0.95. If US/India score ~their validation, France must be ~0.85
(0.85*0.968 + 0.15*F_fr = 0.95 -> F_fr ~ 0.85). France predictions in test: empty rate 4.2-4.6% (US 5.6%, India 5.8-7.3%),
avg matches 3.45-3.47 (US 3.37, India 3.17-3.29) => France likely OVER-matches (predicts more than train-like behaviour).
France predicted pairs: house-number mismatch only 4% (US 11%, India 20%); exact core-name match 70%.
France specifics seen: business names often contain the CITY name ("Nantes Culture SARL", "Pornic Collectif"),
generic French vocabulary (Club, Ecole, Amicale, Association, Pharmacie, SARL/SAS/SA/EI/EURL), small set of cities
(Nantes, Bordeaux, Lille, Roubaix, Tourcoing, Pessac, Merignac, Saint-Nazaire, La Teste-de-Buch, Calais, Dunkerque, ...),
region vs department in address (S1: "Hauts-de-France"; S2: "Nord"), abbreviations R./AV/All./Ch/Bd, "N°", "(6)", "0028",
"Bis". No state stripping for France (lookup only for us/india).
Sample France predictions (reports/france_sample.txt) show FPs like: "Association du Alt | 41 Chemin Moliere" matched
"Association du Alt Holding | 45 Chemin Moliere"; "Pornic Collectif" matched "PORNIC CENTRE" at same address.

## Compute constraints
Windows laptop, 16 GB RAM (swaps above ~14 GB), 16 threads, RTX 3050 4 GB, ~106 GB free disk.
Full train cycle ~2.5 h (blocking 60 min, stage0 15, features 17, stage1 train+predict 30, stage2 30), test cycle ~1.5 h.
Deadline: Sep 27 23:59 IST (now Sep 25 ~22:20). Leaderboard: public subset; final ranking on private subset.
Submissions: max 5/day. User does NOT want leaderboard probing submissions.
Goal: leaderboard > 0.98.

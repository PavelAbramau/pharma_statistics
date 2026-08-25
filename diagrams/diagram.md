---
title: pharma_statistics — ADC Silent Signals pipeline as built (2026-08-25)
---
flowchart TD

%% ============ SOURCES ============
subgraph SRC["① SOURCES — external"]
  direction LR
  CTG["<b>ClinicalTrials.gov</b><br/>API v2 &nbsp;·&nbsp; /studies, /studies/{nct}<br/>/api/int/studies/{nct}/history<br/><i>robots.txt allows /api/int/, Crawl-delay 1s</i>"]
  EDG["<b>SEC EDGAR</b><br/><i>not built</i>"]
  WBK["<b>Wayback / pipeline pages</b><br/><i>not built</i>"]
end

%% ============ INGEST ============
subgraph ING["② INGEST — clients + provenance"]
  CLI["<code>clients/ctgov.py</code><br/>CtgovClient · 1.05 s/req single-thread<br/>search_studies · get_study<br/>get_history · get_study_version"]
  SNP["<code>snapshot.py</code><br/>write verbatim body + URL + fetch_ts + SHA-256"]
  GUARD["<code>history/schema_guard.py</code><br/>targeted key/type asserts on /api/int<br/>hard-FAIL on parsed fields, WARN on shape drift"]
end

RAW[("<b>raw/</b> &nbsp;— immutable JSON<br/><b>59,330 files</b><br/>0 orphans · 0 hash mismatches")]
MAN[("<b>data/manifest.duckdb</b><br/>table <code>snapshots</code><br/>get_as_of(source,id,date) → 81/81 probes correct")]

%% ============ DISCOVERY ============
subgraph DIS["③ DISCOVERY — universe construction"]
  PAT["<code>discovery/patterns.py</code><br/>naming conventions · NON_ADC_DENYLIST<br/>generic-token + short-abbrev guards"]
  CAN["<code>discovery/candidates.py</code><br/>3 strategies: name-pattern · seed-expansion · sponsor-expansion<br/>union-find synonym clustering"]
  SEED["<code>discovery/seed_assets.json</code>"]
  DAU["<code>discovery/audit.py</code><br/>overlap split: 41 likely-genuine combos / 194 noise"]
end

CAND[("<b>asset_candidates</b> + synonyms<br/><b>964 candidates · 2,786 trials</b><br/>⚠ 964 unreviewed · sponsor_expansion = 89%")]
REV["<code>reports/universe_review_sponsor_first.csv</code>"]

%% ============ HISTORY ============
subgraph HIS["④ HISTORY — index + selective backfill"]
  IDX["<code>history/index.py</code><br/>version list only: version, posted_date,<br/>submitted_date, status, moduleLabels"]
  ORC["<code>history/orchestrator.py</code><br/>priority queue · checkpointed · resumable<br/>module filter: Study Design, Outcomes, Status, Sponsor"]
end

HIDX[("<b>history_index</b><br/>2,786/2,786 trials · contiguous from v0<br/>dates monotonic · 0 unknown module labels")]
BQ[("<b>backfill_queue</b><br/>done 2,745 · pending 0 · <b>error 41 (DNS)</b>")]
VERS[("<b>raw/ version bodies</b><br/>→ <b>50,207 version pairs</b>")]

%% ============ DIFFER ============
subgraph DIF["⑤ DIFFER — typed event extraction"]
  EXT["<code>differ/extract.py</code><br/>pull comparable fields per version<br/><i>single DuckDB conn — 1m45s full corpus</i>"]
  DFF["<code>differ/diff.py</code><br/>structural diff, consecutive pairs"]
  EVT["<code>differ/events.py</code><br/>typed + <b>signed</b> events<br/>ESTIMATED↔ACTUAL never crossed → *_finalized<br/>event_date = <b>posted_date</b>, never submitted"]
  DRP["<code>differ/report.py</code><br/>noise floor · negative control"]
end

EV[("<b>evidence_events</b><br/><b>35,557 events · 2,194 trials</b><br/>75.1% of pairs → 0 events<br/>max type: completion_date_pushed 11.7%")]

%% ============ LABELLING ============
subgraph LAB["⑥ LABELLING — the manual layer"]
  PRV["<code>labelling/provisional_programs.py</code><br/><b>v0 stand-in</b>: asset_candidate × trial cluster<br/>heuristic silence score + archetype<br/><i>no OncoTree / line normalisation yet</i>"]
  QUE["<code>labelling/queue.py</code><br/>stratified by score band + archetype<br/>10% silent repeats · skip/requeue"]
  APP["<code>labelling/app.py</code> + <code>static/</code><br/>localhost:8420 · blind mode default on<br/>keyboard-first · seconds-per-label"]
  STO["<code>labelling/store.py</code><br/>append-only writer + provenance block"]
  VOC["<code>labelling/vocab.py</code><br/>status · kill_reason · line enums"]
  STA["<code>labelling/stats.py</code>"]
end

PP[("<b>provisional_programs</b><br/>provisional key:<br/>asset|condition|line")]
GOLD[("<b>gold/labels.jsonl</b><br/><b>0 labels</b> — target 120–150<br/>evidence_date + confirmation_date required")]

%% ============ AUDIT ============
subgraph AUD["⑦ AUDIT — 10 stages, gates the pipeline"]
  AUDX["<code>audit/__main__.py --stage all</code><br/>provenance · universe · history · backfill · differ<br/>normalisation · gold_set · label_sufficiency · features · model<br/><b>FAIL gate: unreviewed candidates > 0</b>"]
  KNOWN["<code>tests/fixtures/known_adcs.txt</code><br/>64 assets, 5 external sources<br/><b>recall 38/64 — the check can now fail</b>"]
end

AUDR[("<b>audit/&lt;ts&gt;.md</b><br/>latest: 0 FAIL · 2 WARN · 11 INFO · 12 PASS")]

%% ============ NOT BUILT ============
subgraph TODO["⑧ NOT BUILT — everything downstream"]
  NRM["<b>normalisation</b><br/>five entities · OncoTree · HGNC<br/>ownership intervals · as-of reads only"]
  FEA["<b>features</b><br/>program × month panel<br/>knowability dates + leakage register"]
  MOD["<b>model</b><br/>discrete-time survival hazard<br/>must beat the heuristic on lead time"]
  PA["<b>Product A</b> — kill feed"]
  PB["<b>Product B</b> — sourcing screener<br/>kill-reason · recoverability · crowding"]
end

%% ============ FLOWS ============
CTG -->|"HTTPS JSON"| CLI
CLI --> SNP
CLI -.->|"/api/int shape check"| GUARD
SNP -->|"verbatim body"| RAW
SNP -->|"row per fetch"| MAN
RAW <-.->|"hash + as-of lookup"| MAN

RAW -->|"cached search snapshots<br/>replayed offline"| CAN
SEED --> CAN
PAT --> CAN
CAN --> CAND
CAND --> DAU
DAU --> REV
REV ==>|"⚠ MANUAL — blocking"| CAND

CAND -->|"2,786 NCT ids"| IDX
IDX --> HIDX
HIDX -->|"module filter"| ORC
ORC --> BQ
ORC -->|"selective version fetch"| CLI
CLI --> VERS

VERS --> EXT
HIDX --> EXT
EXT --> DFF --> EVT --> EV
EVT --> DRP

CAND --> PRV
EV -.->|"⚠ NOT WIRED YET"| PRV
PRV --> PP
PP --> QUE --> APP
VOC --> APP
APP --> STO --> GOLD
GOLD --> STA

MAN --> AUDX
CAND --> AUDX
HIDX --> AUDX
BQ --> AUDX
EV --> AUDX
GOLD --> AUDX
KNOWN --> AUDX
AUDX --> AUDR
AUDX ==>|"FAIL halts run"| CAND

EV -.-> NRM
GOLD -.-> MOD
NRM -.-> FEA -.-> MOD
MOD -.-> PA
MOD -.-> PB
EDG -.-> NRM
WBK -.-> NRM

%% ============ STYLES ============
classDef built fill:#E8F1EC,stroke:#2E6B4C,stroke-width:1px,color:#13161A
classDef store fill:#FFFFFF,stroke:#4A5157,stroke-width:1.5px,color:#13161A
classDef manual fill:#FBEDE6,stroke:#95322D,stroke-width:1.5px,color:#13161A
classDef notbuilt fill:#F2F3F1,stroke:#9AA2A8,stroke-width:1px,stroke-dasharray:4 3,color:#5C6873
classDef src fill:#EDF0F5,stroke:#3D4A5C,stroke-width:1px,color:#13161A

class CTG src
class EDG,WBK,NRM,FEA,MOD,PA,PB notbuilt
class CLI,SNP,GUARD,PAT,CAN,DAU,IDX,ORC,EXT,DFF,EVT,DRP,PRV,QUE,STO,VOC,STA,AUDX built
class APP,REV manual
class RAW,MAN,CAND,HIDX,BQ,VERS,EV,PP,GOLD,AUDR,SEED,KNOWN store
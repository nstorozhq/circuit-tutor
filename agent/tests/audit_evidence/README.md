# Independent offline audit evidence

These files record the independent claims/API audit. Live GigaChat runs were
performed by the application author and are separate evidence.

The scripts need the project's installed Python dependencies. They use temporary
SQLite databases, never load .env, never call GigaChat, and create no accounts.
They locate agent/ from their own path and can run from any working directory.
A failed assertion exits nonzero. The embedded schema checker only validates the
schema constructs this application generates; it is not provider-native validation.

From agent/:

```sh
GIGACHAT_CREDENTIALS='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python tests/audit_evidence/api_regressions.py
GIGACHAT_CREDENTIALS='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python tests/audit_evidence/replay_literal_pairs.py
GIGACHAT_CREDENTIALS='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python tests/audit_evidence/run_unit_audit.py
```

api_regressions.py runs the 12 agreed API cases and writes api-results.json.
Use --output PATH to retain that file and put a rerun elsewhere. Each result
contains an expectation, observed state, pass/fail, and current source hashes.

replay_literal_pairs.py replays the saved D01 extraction with empty nodes, plus
the saved concept-answer responses. Its diagnostic deliberately says contradiction.
It writes d01-replay-results.json. This is an offline replay, not a new live result.

manifest.json records app/KB hashes for the audited copy, evidence file hashes,
and the exact independently run unit modules (62 tests; excludes account tests).
independent-results.json contains compact IDs, expectations and observed outcomes
for the 16 claims probes, 2 positive examples and 5 coverage checks. The permanent
../audit_contract_cases.json and ../test_audit_regressions.py retain the agreed
contract fixtures/tests; the two literal-pair regressions are in ../test_claims.py.
unit-results.txt records the independently observed unit-run output.

Temporary audit snapshots are additional diagnostics only. None of the files here
needs a /tmp path or imports code from an audit snapshot. The original application
and knowledge base were not modified by the auditor.

live-content-review.json separately records the independent manual reading of
the author's completed full and narrow live runs, with source-file hashes and
separate progress/physics/directness judgements. It is not a live run by the auditor.

Current isolation: api_regressions.py suppresses dotenv.load_dotenv only during
app import and restores it, and keeps TUTOR_DB in a TemporaryDirectory until
process exit (then restores the previous value). The replay and unit runner
reuse those import guards. Earlier empty credentials alone did not prevent the
dotenv loader call; that earlier no-read claim was imprecise and is corrected.
compatibility-results.json separates independent Python 3.12.14 checks (62 unit,
12 API) from author-reported Python 3.10.12 checks (65 unit, 12 API) and public smoke.

# Final Specification

## Overview

System for evaluating CSP solver behavior across optimality gaps by:
1. Solving instances at various gaps, capturing search traces
2. Building search trees from traces (explored paths only)
3. We need to know if unexplored branches are SAT/UNSAT subtrees, we run complementary verifications

## Data Model

### Instance
- XCSP optimization problem (minimization only)
- Known optimal solution value
- Stored in `xcsp_solved.json`, imported to sqlite

```
[
  {
    "id": 151,
    "name": "AlteredStates-02",
    "fullname": "/cop/AlteredStates-02",
    "family": "AlteredStates",
    "competition_id": 7,
    "nb_variables": 1478,
    "nb_constraints": 1967,
    "best_bound": 7,
    "optim": 1,
    "info_domains": "#types:3 #values:13288 (#2:412 ... #26:416)",
    "info_constraints": "#extension:362 #intension:1186 #lex:7 #element:412 ",
    "useless_vars": 238,
    "type": "max SUM",
    "created_at": null,
    "updated_at": "2025-07-31T16:48:25.000000Z"
  },
```

### Resolution
- Single solver run: (instance, optimality_gap)
- Produces search trace with explored path
- Stops at first solution (SAT)
- Generates search tree pickled to `trees/resolution_{id}.pkl`

### Verification
- Single solver run to check if unexplored branch is SAT/UNSAT
- Input: (instance, optimality_gap, partial_assignment)
- Output: SAT or UNSAT only (no trace)
- Updates map when trees are processed

## Database Schema

```sql
CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    optimal INTEGER NOT NULL,
    UNIQUE (filename, problem_type)
);

CREATE TABLE batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    batch_type TEXT NOT NULL,  -- 'root' | 'root_retry' | 'verification' | 'verification_retry'
    script_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP,
    slurm_job_id TEXT
);

CREATE TABLE resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    instance_id INTEGER NOT NULL,
    gap REAL NOT NULL,
    objective_bound INTEGER NOT NULL,
    timeout_seconds INTEGER NOT NULL DEFAULT 3600,
    log_path TEXT NOT NULL,
    tree_path TEXT,
    result TEXT,  -- 'SAT' | 'UNSAT' | 'TIMEOUT' | 'ERROR' | NULL
    FOREIGN KEY (batch_id) REFERENCES batches(id),
    FOREIGN KEY (instance_id) REFERENCES instances(id)
);

CREATE TABLE verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    resolution_id INTEGER NOT NULL,
    partial_assignment TEXT NOT NULL,
    objective_bound INTEGER NOT NULL,
    timeout_seconds INTEGER NOT NULL DEFAULT 3600,
    result TEXT,  -- 'SAT' | 'UNSAT' | 'TIMEOUT' | 'ERROR' | NULL
    FOREIGN KEY (batch_id) REFERENCES batches(id),
    FOREIGN KEY (resolution_id) REFERENCES resolutions(id)
);
```

## Search Tree Structure

see search_tree.py

### Tree Construction from Resolution Trace
- Parse explored path from trace
- For each decision `x=v`:
  - Create explored child: `Node(decision=(x,'=',v), label=from_trace)`
  - Create unexplored sibling: `Node(decision=(x,'!=',v), label='UNKNOWN')`
- Pickle to `trees/resolution_{resolution_id}.pkl`

## Solver Interface

### Root Resolution Command
```bash
java -jar minicpbp.jar \
  --input data/{instance_file} \
  --bp-algorithm {config.algorithm} \
  --branching {config.branching} \
  --search-type dfs \
  --timeout {timeout_seconds} \
  --trace-search \
  --oracle-on-objective {config.oracle_on_objective} \
  --upper-bound {resolution.actual_bound}
```

Output: search trace to stdout with:
- `### branching on x[i,j]=v` lines
- `SAT` when a solution is found
- `TIMEOUT` when time limit exceeded

### Verification Command
```bash
java -jar minicpbp.jar \
  --verification \
  --input data/{instance_file} \
  --bp-algorithm no-bp \
  --branching dom-wdeg \
  --search-type dfs \
  --timeout {timeout_seconds} \
  --assignment "x[0,0]=1,x[0,1]!=2,..."
```

Output: `SAT`, `UNSAT`, or `TIMEOUT` in stdout

# State Machine Workflow

## State Detection Logic

### init_db
**Condition**: `experiments.db` does not exist
**Action**: Create database schema

### scan_instances
**Condition**: Database exists AND instances table is empty
**Action**: Parse `xcsp_solved.json` and populate instances table

### create_root_batch
**Condition**: Instances exist AND no root batch exists
**Action**: Create batch with (instance × gap) resolutions

### generate_root_script:{batch_id}
**Condition**: Latest root batch has no script_path
**Action**: Generate SLURM array job script

### submit_root_batch:{batch_id}
**Condition**: Script exists AND batch not submitted (submitted_at is NULL)
**Action**: Submit to SLURM, record job_id

### wait_root_batch:{batch_id}
**Condition**: Batch submitted AND `squeue -u $USER` shows running jobs
**Action**: None (informational state - SLURM jobs still executing)

### process_root_logs:{batch_id}
**Condition**: No SLURM jobs running AND some resolutions have result IS NULL
**Action**: Parse logs, extract result (SAT/UNSAT/TIMEOUT/ERROR), build trees with UNKNOWN siblings for SAT/UNSAT, pickle to trees/

### create_root_retry_batch:{batch_id}
**Condition**: Root batch processed AND some resolutions have result = 'TIMEOUT' AND timeout_seconds * 2 <= MAX_TIMEOUT
**Action**: Move TIMEOUT resolutions to new root_retry batch, double timeout_seconds, clear result/tree_path

### create_verification_batch
**Condition**: All root trees built (no more retryable timeouts) AND no verification batch exists
**Action**: Scan all trees for UNKNOWN nodes, create verification tasks

### generate_verification_script:{batch_id}
**Condition**: Latest verification batch has no script_path
**Action**: Generate SLURM array job script

### submit_verification_batch:{batch_id}
**Condition**: Script exists AND batch not submitted
**Action**: Submit to SLURM

### wait_verification_batch:{batch_id}
**Condition**: Batch submitted AND `squeue -u $USER` shows running jobs
**Action**: None (informational state - SLURM jobs still executing)

### process_verification_results:{batch_id}
**Condition**: No SLURM jobs running AND some verifications have result IS NULL
**Action**: Parse logs, extract result, update tree node labels for SAT/UNSAT, re-pickle trees

### create_verification_retry_batch:{batch_id}
**Condition**: Verification batch processed AND some verifications have result = 'TIMEOUT' AND timeout_seconds * 2 <= MAX_TIMEOUT
**Action**: Move TIMEOUT verifications to new verification_retry batch, double timeout_seconds, clear result

### complete
**Condition**: All verification results processed into trees (no more retryable timeouts)
**Action**: None

## Execution Pattern

```
init_db
  ↓
scan_instances
  ↓
create_root_batch
  ↓
generate_root_script
  ↓
submit_root_batch
  ↓
wait_root_batch (polls squeue until empty)
  ↓
process_root_logs
  ↓
create_root_retry_batch (if TIMEOUT results exist)
  ↓
generate_root_script (retry batch)
  ↓
submit_root_batch
  ↓
wait_root_batch
  ↓
process_root_logs (retry batch)
  ↓
[loop until no more retryable TIMEOUTs or MAX_TIMEOUT reached]
  ↓
create_verification_batch
  ↓
generate_verification_script
  ↓
submit_verification_batch
  ↓
wait_verification_batch (polls squeue until empty)
  ↓
process_verification_results
  ↓
create_verification_retry_batch (if TIMEOUT results exist)
  ↓
generate_verification_script (retry batch)
  ↓
submit_verification_batch
  ↓
wait_verification_batch
  ↓
process_verification_results (retry batch)
  ↓
[loop until no more retryable TIMEOUTs or MAX_TIMEOUT reached]
  ↓
complete
```

## Wait State Mechanism

Wait states check `squeue -u $USER`:
- If output has >1 line (header + job lines): jobs running → stay in wait state
- If output has ≤1 line (header only): no jobs → proceed to next state

The script exits during wait states. Re-run to check status and advance when ready.

## Usage

```bash
# Check current state
./schedule.py --dry-run

# Execute next action
./schedule.py

# Run repeatedly with periodic checks
while true; do ./schedule.py && sleep 60 || break; done
```

## Partial Assignment Format

Alphabetically sorted, comma-separated:
```
x[0,0]!=1,x[0,1]=3,x[0,2]=2
```

Both positive (`=`) and negative (`!=`) assignments included.

## Timeout Management

### Retry Mechanism
- When resolution or verification completes with `TIMEOUT` result:
  1. Row is moved to new retry batch (batch_type = 'root_retry' or 'verification_retry')
  2. `timeout_seconds` is doubled
  3. `result` and `tree_path` are cleared
  4. New log path is generated

### Retry Limits
- Maximum timeout: `MAX_TIMEOUT = 14400` (4 hours)
- Timeouts exceeding this limit are not retried
- Original resolution/verification ID is preserved

### State Derivation
No explicit status column - state is derived from:
- `result IS NULL` → pending log processing
- `result = 'TIMEOUT'` → needs retry (if timeout_seconds * 2 <= MAX_TIMEOUT)
- `result IN ('SAT', 'UNSAT')` → completed successfully
- `result = 'ERROR'` → parsing or solver error

## Additional Considerations

1. **Verification timeout handling**: TIMEOUT verifications are retried with 2x timeout up to MAX_TIMEOUT

2. **Tree update strategy**: Sequential processing sufficient

3. **Verification batch size limits**: Max array size is 1000 (SLURM limit)

4. **Error recovery**: ERROR results indicate parsing failures or solver crashes - manual investigation required

5. **Batch type tracking**: New batch types 'root_retry' and 'verification_retry' for audit trail

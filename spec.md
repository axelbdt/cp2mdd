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

see create_db.sql


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
  --timeout {seconds} \
  --trace-search \
  --oracle-on-objective {config.oracle_on_objective} \
  --upper-bound {resolution.actual_bound}
```

Output: search trace to stdout with:
- `### branching on x[i,j]=v` lines
- `SAT` when a solution is found

### Verification Command
```bash
java -jar minicpbp.jar \
  --verification \
  --input data/{instance_file} \
  --bp-algorithm no-bp \
  --branching dom-wdeg \
  --search-type dfs \
  --timeout {seconds} \
  --assignment "x[0,0]=1,x[0,1]!=2,..."
```

Output: `SAT` or `UNSAT` in stdout

# Workflow

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

### wait_root_batch:{batch_id}:{completed}/{total}
**Condition**: Batch submitted AND some resolutions lack result
**Action**: None (informational state)

### process_root_logs:{batch_id}
**Condition**: All resolutions have result AND some lack tree_path
**Action**: Parse logs, build trees with UNKNOWN siblings, pickle to trees/

### create_verification_batch
**Condition**: All root trees built AND no verification batch exists
**Action**: Scan all trees for UNKNOWN nodes, create verification tasks

### generate_verification_script:{batch_id}
**Condition**: Latest verification batch has no script_path
**Action**: Generate SLURM array job script

### submit_verification_batch:{batch_id}
**Condition**: Script exists AND batch not submitted
**Action**: Submit to SLURM

### wait_verification_batch:{batch_id}:{completed}/{total}
**Condition**: Batch submitted AND some verifications lack result
**Action**: None (informational state)

### process_verification_results:{batch_id}
**Condition**: All verifications have result AND trees need updating
**Action**: Update tree node labels from verification results, re-pickle

### complete
**Condition**: All verification results processed into trees
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
wait_root_batch (external: SLURM completes)
  ↓
process_root_logs
  ↓
create_verification_batch
  ↓
generate_verification_script
  ↓
submit_verification_batch
  ↓
wait_verification_batch (external: SLURM completes)
  ↓
process_verification_results
  ↓
complete
```

## Usage

```bash
# Check current state
./schedule.py --dry-run

# Execute next action
./schedule.py

# Run repeatedly until waiting or complete
while ./schedule.py; do sleep 1; done
```


## Partial Assignment Format

Alphabetically sorted, comma-separated:
```
x[0,0]!=1,x[0,1]=3,x[0,2]=2
```

Both positive (`=`) and negative (`!=`) assignments included.

## Additional Considerations

1. **Verification timeout handling**: TIMEOUT verifications should be retried with a 2x larger timeout

2. **Tree update strategy**: Sequential processing sufficient

4. **Verification batch size limits**: Max array size is 700 but with a configurable constant in script

5. **Error recovery**: If verification, parsing, ... fails should be marked error and retried

## Timeout management

Add a timeout to resolutions and verifications, if a timeout is reached, this should appear in the logs and result will be marked as timeout.
We'll generate


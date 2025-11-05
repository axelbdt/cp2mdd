# Final Specification

## Overview

System for evaluating CSP solver behavior across optimality gaps by:
1. Solving instances at various gaps, capturing search traces
2. Building search trees from traces (explored paths only)
3. We need to know if unexplored branches are SAT/UNSAT subtrees, we run complementary verifications
4. We build from all resolutions and verifications, a map of the "search space" for a given instance, in ordre to reuse data while labeling for other optimality gaps
5. We iterate over all this to annotate all choices of every resolutions

## Data Model

### Instance
- XCSP optimization problem (minimization only)
- Known optimal solution value
- Stored in `xcsp_solved.json`, imported to sqlite

### Resolution
- Single solver run: (instance, optimality_gap)
- Produces search trace with explored path
- Stops at first solution (SAT)
- Generates search tree pickled to `trees/resolution_{id}.pkl`
- Incrementally build a map of the search space pickled to `maps/instance_{id}.pkl`

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


## Map Structure

It is a tree structure where a partial assignment is a series of branches in lexicographic order.
We maintain for each node a lowest SAT (which is the score of the best solution found in the subtree), as well as the lowest upperbound for which the the subtree is UNSAT.

e.g. root node has lowest SAT = optimal, upper bound UNSAT = optimal - 1
for some other node we might find a solution with score 100, but find the subtree is UNSAT if we place an upperbound of 90. Then lowest SAT = 100, upper bound UNSAT = 90

### Map Construction

- For a search tree
  - for each node try to label from search trace or instance map
  - if not possible create a verification task
- Run all verifications
- Update map from all verifications


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

## Workflow Commands

Create database, popoulate instances from `xcsp_solved.json` and configurations from `config.py`

Generate resolutions : one for each instance and optimality gap and config

Generate verifications : parse search trees, label nodes from search traces and create verification tasks for unkown nodes

If a batch of verifications has been run and logs are not processed, process it, i.e. update the map and search trees

Generate verification batch : for each verification task never run, check if the node can be labeled from the instance map, if not put it in the next verification batch (until batch limit is reached)
generate verification batch script and schedule verifications on SLURM


## Partial Assignment Format

Alphabetically sorted, comma-separated:
```
x[0,0]!=1,x[0,1]=3,x[0,2]=2
```

Both positive (`=`) and negative (`!=`) assignments included.

## Additional Considerations

1. **Verification timeout handling**: Should TIMEOUT verifications be retried with a 2x larger timeout

2. **Tree update strategy**: Sequential processing sufficient

4. **Verification batch size limits**: Max array size is 700 but with a configurable constant in script

5. **Error recovery**: If verification, parsing, ... fails should be marked error and retried

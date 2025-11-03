# Final Specification

## Overview

System for evaluating CSP solver behavior across optimality gaps by:
1. Solving instances at various gaps, capturing search traces
2. Building search trees from traces (explored paths only)
3. Verifying unexplored branches (negations) to label all nodes
4. Analyzing complete trees for algorithm evaluation

## Data Model

### Instance
- XCSP optimization problem (minimization only)
- Known optimal solution value
- Stored in `xcsp_solved.json`, imported to sqlite

### Resolution
- Single solver run: (instance, optimality_gap)
- Produces search trace with explored path
- Generates search tree pickled to `trees/resolution_{id}.pkl`
- Stops at first solution (SAT)

### Verification
- Single solver run to check if unexplored branch is SAT/UNSAT
- Input: (instance, optimality_gap, partial_assignment)
- Output: SAT or UNSAT only (no trace)
- Updates parent resolution's pickled tree

## Database Schema

```sql
-- Instances: problem definitions
CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    optimal INTEGER NOT NULL,
    UNIQUE (filename, problem_type)
);

-- Batches: groups of solver runs
CREATE TABLE batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    batch_type TEXT NOT NULL,  -- 'root' | 'verification'
    script_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP,
    slurm_job_id TEXT
);

-- Resolutions: root solver runs
CREATE TABLE resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    instance_id INTEGER NOT NULL,
    gap REAL NOT NULL,
    objective_bound INTEGER NOT NULL,
    log_path TEXT NOT NULL,
    tree_path TEXT,  -- trees/resolution_{id}.pkl
    result TEXT,  -- 'SAT' | 'UNSAT' | 'TIMEOUT' | 'ERROR'
    FOREIGN KEY (batch_id) REFERENCES batches(id),
    FOREIGN KEY (instance_id) REFERENCES instances(id)
);

-- Verifications: unexplored branch checks
CREATE TABLE verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    resolution_id INTEGER NOT NULL,
    partial_assignment TEXT NOT NULL,
    objective_bound INTEGER NOT NULL,
    result TEXT,  -- 'SAT' | 'UNSAT' | 'TIMEOUT' | 'ERROR'
    FOREIGN KEY (batch_id) REFERENCES batches(id),
    FOREIGN KEY (resolution_id) REFERENCES resolutions(id)
);

CREATE INDEX idx_resolutions_batch ON resolutions(batch_id);
CREATE INDEX idx_resolutions_instance ON resolutions(instance_id);
CREATE INDEX idx_verifications_batch ON verifications(batch_id);
CREATE INDEX idx_verifications_resolution ON verifications(resolution_id);
```

## Search Tree Structure

```python
@dataclass
class Node:
    decision: Optional[Tuple[str, str, Any]]  # (var, op, val) or None for root
    label: Optional[str]  # 'SAT' | 'UNSAT' | 'UNKNOWN' | None
    children: List['Node']
    parent: Optional['Node']
    depth: int
    
    def path_from_root(self) -> List[Tuple[str, str, Any]]:
        """Full path including both x=v and x!=v decisions."""
        pass
```

### Tree Construction from Resolution Trace
- Parse explored path from trace
- For each decision `x=v`:
  - Create explored child: `Node(decision=(x,'=',v), label=from_trace)`
  - Create unexplored sibling: `Node(decision=(x,'!=',v), label='UNKNOWN')`
- Pickle to `trees/resolution_{resolution_id}.pkl`

### Tree Update from Verifications
- Load pickled tree
- Match verification partial_assignment to node path
- Update node label: UNKNOWN → SAT/UNSAT
- Pickle updated tree

## Solver Interface

### Root Resolution Command
```bash
java -jar minicpbp.jar \
  --input data/{instance_file} \
  --bp-algorithm max-product \
  --branching dom-wdeg \
  --search-type dfs \
  --timeout {seconds} \
  --trace-search \
  --oracle-on-objective {bound}
```

Output: search trace to stdout with:
- `### branching on x[i,j]=v` lines
- `SAT` or `UNSAT` leaf labels

### Verification Command
```bash
java -jar minicpbp.jar \
  --input data/{instance_file} \
  --bp-algorithm max-product \
  --branching dom-wdeg \
  --search-type dfs \
  --timeout {seconds} \
  --oracle-on-objective {bound} \
  --assignment "x[0,0]=1,x[0,1]!=2,..."
```

Output: `SAT` or `UNSAT` in stdout

## Workflow Commands

### Setup
```bash
schedule.py init-db                      # Create schema
schedule.py scan-instances xcsp_solved.json  # Import instances
```

### Phase 1: Root Solving
```bash
schedule.py create-root-batch [--limit N]
schedule.py generate-script <batch_id>
schedule.py submit-batch <batch_id>
# Wait for completion
schedule.py process-root-logs <batch_id>  # Parse traces → pickle trees
```

### Phase 2: Verification
```bash
schedule.py create-verification-batch     # Scan all trees for UNKNOWN nodes
schedule.py generate-script <batch_id>
schedule.py submit-batch <batch_id>
# Wait for completion
schedule.py process-verification-results <batch_id>  # Update trees
```

### Analysis
```bash
schedule.py extract-stats                 # Process all trees → stats (TBD)
schedule.py list-batches                  # Show batch status
```

## Partial Assignment Format

Alphabetically sorted, comma-separated:
```
x[0,0]!=1,x[0,1]=3,x[0,2]=2
```

Both positive (`=`) and negative (`!=`) assignments included.

## Open Questions for Implementation

1. **Verification timeout handling**: Should TIMEOUT verifications be retried, or left as UNKNOWN?

2. **Tree update strategy**: Sequential processing sufficient, or worth parallelizing?

3. **Stats schema**: What metrics per instance? Per (instance, gap)? Tree depth, branching factor, SAT/UNSAT ratio?

4. **Verification batch size limits**: Should there be a max array size for SLURM, requiring multiple verification batches?

5. **Error recovery**: If verification fails, should node remain UNKNOWN or be marked ERROR?

Ready to proceed with implementation?

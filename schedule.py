#!/usr/bin/env python3
"""
XCSP Optimization Gap Testing Framework
Manages root resolutions, verification, and result tracking.
"""

import argparse
import json
import pickle
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Gap values to test (percentages)
DEFAULT_GAPS = [0, 0.5, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]


@contextmanager
def db_connection(db_path: str = "experiments.db"):
    """Manage database connection with automatic commit/rollback."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database(db_path: str = "experiments.db"):
    """Initialize database schema."""
    schema = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS instances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        problem_type TEXT NOT NULL,
        optimal INTEGER NOT NULL,
        UNIQUE (filename, problem_type)
    );

    CREATE TABLE IF NOT EXISTS batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        batch_type TEXT NOT NULL,  -- 'root' | 'verification'
        script_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        submitted_at TIMESTAMP,
        slurm_job_id TEXT
    );

    CREATE TABLE IF NOT EXISTS resolutions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL,
        instance_id INTEGER NOT NULL,
        gap REAL NOT NULL,
        objective_bound INTEGER NOT NULL,
        log_path TEXT NOT NULL,
        tree_path TEXT,
        result TEXT,
        FOREIGN KEY (batch_id) REFERENCES batches(id),
        FOREIGN KEY (instance_id) REFERENCES instances(id)
    );

    CREATE TABLE IF NOT EXISTS verifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL,
        resolution_id INTEGER NOT NULL,
        partial_assignment TEXT NOT NULL,
        objective_bound INTEGER NOT NULL,
        result TEXT,
        FOREIGN KEY (batch_id) REFERENCES batches(id),
        FOREIGN KEY (resolution_id) REFERENCES resolutions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_resolutions_batch ON resolutions(batch_id);
    CREATE INDEX IF NOT EXISTS idx_resolutions_instance ON resolutions(instance_id);
    CREATE INDEX IF NOT EXISTS idx_verifications_batch ON verifications(batch_id);
    CREATE INDEX IF NOT EXISTS idx_verifications_resolution ON verifications(resolution_id);
    """

    with db_connection(db_path) as conn:
        conn.executescript(schema)

    print(f"Database initialized: {db_path}")


def scan_instances(json_file: str, db_path: str = "experiments.db"):
    """Scan xcsp_solved.json and populate instances table."""
    json_path = Path(json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_file}")
        return

    with open(json_path) as f:
        data = json.load(f)

    instances = []
    for entry in data:
        name = entry["name"]
        filename = f"cop/{name}_c25.xml.lzma"
        family = entry.get("family", "unknown")
        optimal = entry["best_bound"]
        instances.append((filename, family, optimal))

    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        inserted = 0
        for filename, family, optimal in instances:
            try:
                cursor.execute(
                    """
                    INSERT INTO instances (filename, problem_type, optimal)
                    VALUES (?, ?, ?)
                    ON CONFLICT(filename, problem_type) DO UPDATE SET optimal = ?
                    """,
                    (filename, family, optimal, optimal),
                )
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.IntegrityError:
                pass

        print(f"Scanned {len(instances)} instances, inserted/updated {inserted}")


def normalize_partial_assignment(assignment: str) -> str:
    """Normalize partial assignment to canonical form (alphabetically sorted)."""
    if not assignment:
        return ""
    parts = [p.strip() for p in assignment.split(",")]
    parts.sort()
    return ",".join(parts)


def format_partial_assignment_for_solver(assignment: str) -> str:
    """Format partial assignment for solver command line."""
    if not assignment:
        return ""
    return f'--assignment "{assignment}"'


def calculate_objective_bound(optimal: int, gap: float) -> int:
    """Calculate objective bound based on gap percentage."""
    if optimal >= 0:
        bound = optimal * (1 + gap)
    else:
        bound = optimal * (1 - gap)
    return int(bound)


def build_solver_command(
    instance_file: str,
    objective_bound: int,
    partial_assignment: Optional[str] = None,
    timeout: int = 3600,
    trace_search: bool = True,
    solver_jar: str = "minicpbp.jar",
    java_bin: str = "java",
) -> str:
    """Build complete solver command."""
    # Handle .lzma compressed files: decompress to temp file first
    if instance_file.endswith(".lzma"):
        cmd_parts = [
            f"lzcat {instance_file} > /tmp/instance_$SLURM_ARRAY_TASK_ID.xml &&",
            java_bin,
            "-jar",
            solver_jar,
            "--input",
            "/tmp/instance_$SLURM_ARRAY_TASK_ID.xml",
        ]
    else:
        cmd_parts = [
            java_bin,
            "-jar",
            solver_jar,
            "--input",
            instance_file,
        ]
    
    cmd_parts.extend([
        "--bp-algorithm",
        "max-product",
        "--branching",
        "dom-wdeg",
        "--search-type",
        "dfs",
        "--timeout",
        str(timeout),
        f"--oracle-on-objective {objective_bound}",
    ])

    if trace_search:
        cmd_parts.append("--trace-search")

    if partial_assignment:
        assignment_arg = format_partial_assignment_for_solver(partial_assignment)
        if assignment_arg:
            cmd_parts.append(assignment_arg)

    return " ".join(cmd_parts)


def create_root_batch(
    db_path: str = "experiments.db",
    gaps: Optional[List[float]] = None,
    instance_limit: Optional[int] = None,
    dry_run: bool = False,
):
    """Create batch for root resolutions (testing instances at various gaps)."""
    if gaps is None:
        gaps = DEFAULT_GAPS

    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        query = "SELECT id, filename, problem_type, optimal FROM instances"
        if instance_limit:
            query += f" LIMIT {instance_limit}"

        cursor.execute(query)
        instances = cursor.fetchall()

        if not instances:
            print("No instances found")
            return

        print(f"Found {len(instances)} instances")
        print(f"Testing {len(gaps)} gap values: {gaps}")
        print(f"Total resolutions: {len(instances) * len(gaps)}")

        if dry_run:
            print("\nDRY RUN - Sample resolutions:")
            for inst_id, filename, problem_type, optimal in instances[:3]:
                for gap in gaps[:3]:
                    bound = calculate_objective_bound(optimal, gap)
                    print(f"  {filename} @ gap={gap}: bound={bound}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"root_{timestamp}"

        cursor.execute(
            "INSERT INTO batches (name, batch_type) VALUES (?, 'root')", (batch_name,)
        )
        batch_id = cursor.lastrowid

        resolution_count = 0
        for inst_id, filename, problem_type, optimal in instances:
            for gap in gaps:
                bound = calculate_objective_bound(optimal, gap)
                log_path = f"logs/{batch_name}/resolution_{resolution_count}.log"

                cursor.execute(
                    """
                    INSERT INTO resolutions 
                    (batch_id, instance_id, gap, objective_bound, log_path)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (batch_id, inst_id, gap, bound, log_path),
                )
                resolution_count += 1

        print(f"Created batch {batch_id}: {batch_name}")
        print(f"  {resolution_count} resolutions scheduled")

        return batch_id


def generate_script(batch_id: int, db_path: str = "experiments.db") -> Path:
    """Generate SLURM array job script for batch."""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT name, batch_type FROM batches WHERE id = ?", (batch_id,))
        batch_row = cursor.fetchone()
        if not batch_row:
            raise ValueError(f"Batch {batch_id} not found")

        batch_name, batch_type = batch_row

        if batch_type == "root":
            cursor.execute(
                """
                SELECT r.id, r.objective_bound, r.log_path, i.filename
                FROM resolutions r
                JOIN instances i ON r.instance_id = i.id
                WHERE r.batch_id = ?
                ORDER BY r.id
                """,
                (batch_id,),
            )
            rows = cursor.fetchall()
            task_type = "resolution"
        elif batch_type == "verification":
            cursor.execute(
                """
                SELECT v.id, v.objective_bound, v.partial_assignment,
                       i.filename, r.log_path
                FROM verifications v
                JOIN resolutions r ON v.resolution_id = r.id
                JOIN instances i ON r.instance_id = i.id
                WHERE v.batch_id = ?
                ORDER BY v.id
                """,
                (batch_id,),
            )
            rows = cursor.fetchall()
            task_type = "verification"
        else:
            raise ValueError(f"Unknown batch_type: {batch_type}")

        if not rows:
            raise ValueError(f"No tasks found for batch {batch_id}")

        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={batch_name}",
            f"#SBATCH --output=logs/{batch_name}/slurm_%A_%a.out",
            f"#SBATCH --error=logs/{batch_name}/slurm_%A_%a.err",
            f"#SBATCH --array=0-{len(rows)-1}",
            "#SBATCH --time=02:00:00",
            "#SBATCH --mem=8G",
            "#SBATCH --cpus-per-task=1",
            "",
            "# Task ID array",
            "TASK_IDS=(",
        ]

        for row in rows:
            script_lines.append(f"  {row[0]}")
        script_lines.append(")")

        script_lines.extend(["", "# Command array", "COMMANDS=("])

        if task_type == "resolution":
            for task_id, bound, log_path, filename in rows:
                cmd = build_solver_command(
                    instance_file=filename,
                    objective_bound=bound,
                    trace_search=True,
                )
                cmd_escaped = cmd.replace('"', '\\"')
                script_lines.append(f'  "{cmd_escaped}"')
        else:  # verification
            for task_id, bound, assignment, filename, _ in rows:
                cmd = build_solver_command(
                    instance_file=filename,
                    objective_bound=bound,
                    partial_assignment=assignment,
                    trace_search=False,
                )
                cmd_escaped = cmd.replace('"', '\\"')
                script_lines.append(f'  "{cmd_escaped}"')

        script_lines.append(")")

        script_lines.extend(["", "# Log file array", "LOG_FILES=("])

        if task_type == "resolution":
            for _, _, log_path, _ in rows:
                script_lines.append(f'  "{log_path}"')
        else:  # verification
            for _, _, _, _, base_log_path in rows:
                log_dir = Path(base_log_path).parent
                log_path = log_dir / f"verification_{rows[0][0]}.log"
                script_lines.append(f'  "{log_path}"')

        script_lines.append(")")

        script_lines.extend(
            [
                "",
                "# Execute task",
                "TASK_ID=$SLURM_ARRAY_TASK_ID",
                'ID="${TASK_IDS[$TASK_ID]}"',
                'COMMAND="${COMMANDS[$TASK_ID]}"',
                'LOG_FILE="${LOG_FILES[$TASK_ID]}"',
                "",
                'echo "Task ID: $ID"',
                'echo "Log file: $LOG_FILE"',
                'echo "Command: $COMMAND"',
                "",
                'mkdir -p $(dirname "$LOG_FILE")',
                "",
                'eval $COMMAND > "$LOG_FILE" 2>&1',
                "",
                'echo "Completed at $(date)"',
            ]
        )

        script_dir = Path("batches")
        script_dir.mkdir(exist_ok=True)

        script_path = script_dir / f"{batch_name}.sh"
        script_path.write_text("\n".join(script_lines))
        script_path.chmod(0o755)

        cursor.execute(
            "UPDATE batches SET script_path = ? WHERE id = ?",
            (str(script_path), batch_id),
        )

        log_dir = Path("logs") / batch_name
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generated script: {script_path}")
        print(f"  Array size: {len(rows)}")

        return script_path


def submit_batch(batch_id: int, db_path: str = "experiments.db", dry_run: bool = False):
    """Submit batch to SLURM."""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name, script_path FROM batches WHERE id = ?", (batch_id,)
        )
        batch_row = cursor.fetchone()
        if not batch_row:
            print(f"Batch {batch_id} not found")
            return False

        batch_name, script_path = batch_row

        if not script_path or not Path(script_path).exists():
            print(f"Script not found: {script_path}")
            print("Run: schedule.py generate-script <batch_id>")
            return False

        if dry_run:
            print(f"DRY RUN: Would submit {batch_name}")
            print(f"  Command: sbatch {script_path}")
            return True

        try:
            result = subprocess.run(
                ["sbatch", script_path], capture_output=True, text=True, check=True
            )

            job_id = result.stdout.strip().split()[-1]

            cursor.execute(
                """
                UPDATE batches
                SET submitted_at = CURRENT_TIMESTAMP, slurm_job_id = ?
                WHERE id = ?
                """,
                (job_id, batch_id),
            )

            print(f"Submitted batch {batch_id}: {batch_name}")
            print(f"  SLURM job ID: {job_id}")
            return True

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or str(e)
            print(f"Submission failed: {error_msg}")
            return False


def process_root_logs(batch_id: int, db_path: str = "experiments.db"):
    """Process resolution logs, build trees, pickle to trees/ directory."""
    try:
        from search import SearchTree, Node
    except ImportError:
        print("Error: search.py not found")
        return

    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT r.id, r.instance_id, r.log_path, r.objective_bound
            FROM resolutions r
            WHERE r.batch_id = ?
            """,
            (batch_id,),
        )
        resolutions = cursor.fetchall()

        print(f"Processing {len(resolutions)} resolution logs...")

        trees_dir = Path("trees")
        trees_dir.mkdir(exist_ok=True)

        processed = 0
        errors = 0

        for res_id, inst_id, log_path, bound in resolutions:
            log_file = Path(log_path)

            if not log_file.exists():
                print(f"  Warning: Log not found: {log_path}")
                continue

            try:
                log_text = log_file.read_text()

                if "SAT" in log_text:
                    result = "SAT"
                elif "UNSAT" in log_text:
                    result = "UNSAT"
                elif "TIMEOUT" in log_text or "timeout" in log_text.lower():
                    result = "TIMEOUT"
                else:
                    result = "ERROR"

                tree = SearchTree.from_log(log_text)
                
                # Add unexplored siblings for each positive decision
                tree_with_unknowns = add_unexplored_siblings(tree.root)

                tree_path = trees_dir / f"resolution_{res_id}.pkl"
                with open(tree_path, "wb") as f:
                    pickle.dump(tree_with_unknowns, f)

                cursor.execute(
                    "UPDATE resolutions SET result = ?, tree_path = ? WHERE id = ?",
                    (result, str(tree_path), res_id),
                )

                processed += 1

            except Exception as e:
                print(f"  Error processing {log_path}: {e}")
                cursor.execute(
                    "UPDATE resolutions SET result = 'ERROR' WHERE id = ?", (res_id,)
                )
                errors += 1

        print(f"Processed {processed} logs, {errors} errors")


def add_unexplored_siblings(root):
    """Add UNKNOWN sibling nodes for each explored positive decision."""
    from search import Node
    
    def process_node(node):
        if node.decision is not None:
            var, op, val = node.decision
            if op == '=':
                # Add unexplored sibling: x!=v
                sibling = Node(
                    decision=(var, '!=', val),
                    label='UNKNOWN',
                    parent=node.parent,
                    depth=node.depth
                )
                if node.parent:
                    # Insert sibling after current node
                    idx = node.parent.children.index(node)
                    node.parent.children.insert(idx + 1, sibling)
        
        for child in list(node.children):
            process_node(child)
    
    process_node(root)
    return root


def create_verification_batch(db_path: str = "experiments.db", dry_run: bool = False):
    """Create batch to verify all UNKNOWN nodes across all resolution trees."""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, instance_id, tree_path, objective_bound
            FROM resolutions
            WHERE tree_path IS NOT NULL
            """
        )
        resolutions = cursor.fetchall()

        if not resolutions:
            print("No resolutions with trees found")
            return

        print(f"Scanning {len(resolutions)} trees for UNKNOWN nodes...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"verification_{timestamp}"

        cursor.execute(
            "INSERT INTO batches (name, batch_type) VALUES (?, 'verification')",
            (batch_name,),
        )
        batch_id = cursor.lastrowid

        verification_count = 0
        for res_id, inst_id, tree_path, bound in resolutions:
            try:
                with open(tree_path, "rb") as f:
                    root = pickle.load(f)

                unknown_nodes = find_unknown_nodes(root)

                for node in unknown_nodes:
                    path = node.path_from_root()
                    assignment_parts = [f"{var}{op}{val}" for var, op, val in path]
                    assignment = ",".join(assignment_parts)
                    normalized = normalize_partial_assignment(assignment)

                    cursor.execute(
                        """
                        INSERT INTO verifications
                        (batch_id, resolution_id, partial_assignment, objective_bound)
                        VALUES (?, ?, ?, ?)
                        """,
                        (batch_id, res_id, normalized, bound),
                    )
                    verification_count += 1

            except Exception as e:
                print(f"  Error scanning tree {tree_path}: {e}")
                continue

        print(f"Created batch {batch_id}: {batch_name}")
        print(f"  {verification_count} verifications scheduled")

        if dry_run:
            print("\nDRY RUN - would create verification batch")

        return batch_id


def find_unknown_nodes(root):
    """Recursively find all nodes with label='UNKNOWN'."""
    unknown = []

    def traverse(node):
        if node.label == "UNKNOWN":
            unknown.append(node)
        for child in node.children:
            traverse(child)

    traverse(root)
    return unknown


def process_verification_results(batch_id: int, db_path: str = "experiments.db"):
    """Process verification results and update pickled trees."""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT v.id, v.resolution_id, v.partial_assignment
            FROM verifications v
            WHERE v.batch_id = ?
            """,
            (batch_id,),
        )
        verifications = cursor.fetchall()

        print(f"Processing {len(verifications)} verification results...")

        # Group by resolution_id
        resolution_map = {}
        for ver_id, res_id, assignment in verifications:
            if res_id not in resolution_map:
                resolution_map[res_id] = []
            resolution_map[res_id].append((ver_id, assignment))

        for res_id, ver_list in resolution_map.items():
            cursor.execute(
                "SELECT tree_path FROM resolutions WHERE id = ?", (res_id,)
            )
            tree_path_row = cursor.fetchone()
            if not tree_path_row or not tree_path_row[0]:
                continue

            tree_path = Path(tree_path_row[0])
            if not tree_path.exists():
                continue

            try:
                with open(tree_path, "rb") as f:
                    root = pickle.load(f)

                for ver_id, assignment in ver_list:
                    # Find verification log
                    log_path = tree_path.parent / f"verification_{ver_id}.log"
                    if not log_path.exists():
                        continue

                    log_text = log_path.read_text()

                    if "SAT" in log_text:
                        result = "SAT"
                    elif "UNSAT" in log_text:
                        result = "UNSAT"
                    elif "TIMEOUT" in log_text or "timeout" in log_text.lower():
                        result = "TIMEOUT"
                    else:
                        result = "ERROR"

                    cursor.execute(
                        "UPDATE verifications SET result = ? WHERE id = ?",
                        (result, ver_id),
                    )

                    # Update tree node
                    node = find_node_by_path(root, assignment)
                    if node and result in ["SAT", "UNSAT"]:
                        node.label = result

                # Save updated tree
                with open(tree_path, "wb") as f:
                    pickle.dump(root, f)

            except Exception as e:
                print(f"  Error updating tree {tree_path}: {e}")
                continue

        print("Verification results processed and trees updated")


def find_node_by_path(root, partial_assignment: str):
    """Find node in tree matching partial assignment path."""
    if not partial_assignment:
        return root

    parts = [p.strip() for p in partial_assignment.split(",")]
    target_path = []
    for part in parts:
        if "!=" in part:
            var, val = part.split("!=")
            target_path.append((var.strip(), "!=", val.strip()))
        elif "=" in part:
            var, val = part.split("=")
            target_path.append((var.strip(), "=", val.strip()))

    def traverse(node, depth):
        if depth == len(target_path):
            return node

        if depth < len(target_path):
            target_var, target_op, target_val = target_path[depth]
            for child in node.children:
                if child.decision:
                    var, op, val = child.decision
                    if var == target_var and op == target_op and str(val) == target_val:
                        result = traverse(child, depth + 1)
                        if result:
                            return result
        return None

    return traverse(root, 0)


def list_batches(db_path: str = "experiments.db"):
    """List all batches with status."""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                id, name, batch_type,
                script_path IS NOT NULL as has_script,
                submitted_at IS NOT NULL as submitted,
                slurm_job_id
            FROM batches
            ORDER BY created_at DESC
            LIMIT 50
            """
        )

        batches = cursor.fetchall()

        if not batches:
            print("No batches found")
            return

        print(
            f"{'ID':<5} {'Name':<30} {'Type':<12} {'Script':<7} {'Submit':<7} {'Job ID'}"
        )
        print("-" * 90)

        for batch_id, name, btype, has_script, submitted, job_id in batches:
            script_check = "✓" if has_script else "✗"
            submit_check = "✓" if submitted else "✗"
            job_display = (job_id or "")[:12]

            print(
                f"{batch_id:<5} {name:<30} {btype:<12} {script_check:<7} {submit_check:<7} {job_display}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="XCSP Optimization Gap Testing Framework"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("init-db", help="Initialize database schema")

    scan_parser = subparsers.add_parser(
        "scan-instances", help="Scan xcsp_solved.json for instances"
    )
    scan_parser.add_argument("json_file", help="xcsp_solved.json file")

    root_parser = subparsers.add_parser(
        "create-root-batch", help="Create root resolution batch"
    )
    root_parser.add_argument("--limit", type=int, help="Limit number of instances")
    root_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )

    gen_parser = subparsers.add_parser(
        "generate-script", help="Generate SLURM script for batch"
    )
    gen_parser.add_argument("batch_id", type=int, help="Batch ID")

    submit_parser = subparsers.add_parser("submit-batch", help="Submit batch to SLURM")
    submit_parser.add_argument("batch_id", type=int, help="Batch ID")
    submit_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )

    root_logs_parser = subparsers.add_parser(
        "process-root-logs", help="Process resolution logs and build trees"
    )
    root_logs_parser.add_argument("batch_id", type=int, help="Batch ID")

    ver_parser = subparsers.add_parser(
        "create-verification-batch", help="Create verification batch for UNKNOWN nodes"
    )
    ver_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )

    ver_results_parser = subparsers.add_parser(
        "process-verification-results", help="Process verification results"
    )
    ver_results_parser.add_argument("batch_id", type=int, help="Batch ID")

    subparsers.add_parser("list-batches", help="List all batches")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "init-db":
        init_database()

    elif args.command == "scan-instances":
        scan_instances(args.json_file)

    elif args.command == "create-root-batch":
        create_root_batch(instance_limit=args.limit, dry_run=args.dry_run)

    elif args.command == "generate-script":
        generate_script(args.batch_id)

    elif args.command == "submit-batch":
        submit_batch(args.batch_id, dry_run=args.dry_run)

    elif args.command == "process-root-logs":
        process_root_logs(args.batch_id)

    elif args.command == "create-verification-batch":
        create_verification_batch(dry_run=args.dry_run)

    elif args.command == "process-verification-results":
        process_verification_results(args.batch_id)

    elif args.command == "list-batches":
        list_batches()


if __name__ == "__main__":
    main()

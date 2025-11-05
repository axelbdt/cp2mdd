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

# Configuration
DEFAULT_GAPS = [0, 0.5, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
BATCH_SIZE = 1000


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
        batch_type TEXT NOT NULL,
        script_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        submitted_at TIMESTAMP,
        slurm_job_id TEXT
    );

    CREATE TABLE IF NOT EXISTS resolutions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER,
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
        batch_id INTEGER,
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


def calculate_objective_bound(optimal: int, gap: float) -> int:
    """Calculate objective bound based on gap percentage."""
    if optimal >= 0:
        bound = optimal * (1 + gap)
    else:
        bound = optimal * (1 - gap)
    return int(bound)


def create_instances_and_resolutions(
    json_file: str, db_path: str = "experiments.db", gaps: Optional[List[float]] = None
):
    """Scan xcsp_solved.json, populate instances, and create all resolutions."""
    if gaps is None:
        gaps = DEFAULT_GAPS

    json_path = Path(json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_file}")
        return

    with open(json_path) as f:
        data = json.load(f)

    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Insert/update instances
        instances = []
        for entry in data:
            name = entry["name"]
            filename = f"cop/{name}_c25.xml.lzma"
            family = entry.get("family", "unknown")
            optimal = entry["best_bound"]

            cursor.execute(
                """
                INSERT INTO instances (filename, problem_type, optimal)
                VALUES (?, ?, ?)
                ON CONFLICT(filename, problem_type) DO UPDATE SET optimal = ?
                """,
                (filename, family, optimal, optimal),
            )
            inst_id = cursor.lastrowid
            if cursor.rowcount > 0:
                instances.append((inst_id, filename, family, optimal))

        # Fetch all instance IDs
        cursor.execute("SELECT id, filename, problem_type, optimal FROM instances")
        all_instances = cursor.fetchall()

        # Create resolutions for all instances at all gaps
        resolution_count = 0
        for inst_id, filename, problem_type, optimal in all_instances:
            for gap in gaps:
                bound = calculate_objective_bound(optimal, gap)
                log_path = f"logs/unscheduled/resolution_{resolution_count}.log"

                cursor.execute(
                    """
                    INSERT INTO resolutions 
                    (instance_id, gap, objective_bound, log_path)
                    VALUES (?, ?, ?, ?)
                    """,
                    (inst_id, gap, bound, log_path),
                )
                resolution_count += 1

        print(f"Processed {len(all_instances)} instances")
        print(f"Created {resolution_count} resolutions ({len(gaps)} gaps each)")


def queue_empty() -> bool:
    """Check if SLURM queue is empty for current user."""
    try:
        import os

        user = os.environ.get("USER", os.environ.get("USERNAME", ""))
        if not user:
            print("Warning: Cannot determine username, assuming queue is empty")
            return True

        result = subprocess.run(
            ["squeue", "-u", user], capture_output=True, text=True, check=False
        )
        lines = result.stdout.strip().split("\n")
        # If only header line present, queue is empty
        return len(lines) <= 1
    except FileNotFoundError:
        print("Warning: 'squeue' command not found, assuming queue is empty")
        return True


def submit_resolution_batch(
    db_path: str = "experiments.db", batch_size: int = BATCH_SIZE
):
    """Create, generate script, and submit batch for unscheduled resolutions."""

    if not queue_empty():
        print("Queue is not empty. Not scheduling new batch.")
        return

    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Find unscheduled resolutions
        cursor.execute(
            """
            SELECT r.id, r.instance_id, r.gap, r.objective_bound, r.log_path, i.filename
            FROM resolutions r
            JOIN instances i ON r.instance_id = i.id
            WHERE r.batch_id IS NULL
            ORDER BY r.id
            LIMIT ?
            """,
            (batch_size,),
        )
        resolutions = cursor.fetchall()

        if not resolutions:
            print("No unscheduled resolutions found")
            return

        # Create batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"root_{timestamp}"

        cursor.execute(
            "INSERT INTO batches (name, batch_type) VALUES (?, 'root')", (batch_name,)
        )
        batch_id = cursor.lastrowid

        # Assign resolutions to batch
        resolution_ids = [r[0] for r in resolutions]
        cursor.execute(
            f"UPDATE resolutions SET batch_id = ? WHERE id IN ({','.join('?' * len(resolution_ids))})",
            [batch_id] + resolution_ids,
        )

        print(f"Created batch {batch_id}: {batch_name}")
        print(f"  {len(resolutions)} resolutions")

        # Generate script
        script_path = generate_resolution_script(batch_id, batch_name, resolutions)

        cursor.execute(
            "UPDATE batches SET script_path = ? WHERE id = ?",
            (str(script_path), batch_id),
        )

        # Create log directory
        log_dir = Path("logs") / batch_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Submit to SLURM
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)], capture_output=True, text=True, check=True
            )

            job_id = result.stdout.strip().split()[-1]

            cursor.execute(
                "UPDATE batches SET submitted_at = CURRENT_TIMESTAMP, slurm_job_id = ? WHERE id = ?",
                (job_id, batch_id),
            )

            print(f"Submitted to SLURM: job {job_id}")

        except subprocess.CalledProcessError as e:
            print(f"Submission failed: {e.stderr or str(e)}")
            return


def generate_resolution_script(
    batch_id: int, batch_name: str, resolutions: List[Tuple]
) -> Path:
    """Generate SLURM script for resolution batch."""

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={batch_name}",
        f"#SBATCH --output=logs/{batch_name}/slurm_%A_%a.out",
        f"#SBATCH --error=logs/{batch_name}/slurm_%A_%a.err",
        f"#SBATCH --array=0-{len(resolutions)-1}",
        "#SBATCH --time=02:00:00",
        "#SBATCH --mem=8G",
        "#SBATCH --cpus-per-task=1",
        "",
        "TASK_IDS=(",
    ]

    for res_id, _, _, _, _, _ in resolutions:
        script_lines.append(f"  {res_id}")
    script_lines.append(")")

    script_lines.extend(["", "COMMANDS=("])

    for _, _, _, bound, _, filename in resolutions:
        cmd = build_solver_command(
            instance_file=filename,
            objective_bound=bound,
            trace_search=True,
        )
        cmd_escaped = cmd.replace('"', '\\"')
        script_lines.append(f'  "{cmd_escaped}"')

    script_lines.append(")")

    script_lines.extend(["", "LOG_FILES=("])

    for _, _, _, _, log_path, _ in resolutions:
        script_lines.append(f'  "{log_path}"')

    script_lines.append(")")

    script_lines.extend(
        [
            "",
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

    print(f"Generated script: {script_path}")
    return script_path


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
    cmd_parts = [
        java_bin,
        "-jar",
        solver_jar,
        "--input",
        instance_file,
    ]

    cmd_parts.extend(
        [
            "--bp-algorithm",
            "max-product",
            "--branching",
            "dom-wdeg",
            "--search-type",
            "dfs",
            "--timeout",
            str(timeout),
            f"--oracle-on-objective {objective_bound}",
        ]
    )

    if trace_search:
        cmd_parts.append("--trace-search")

    if partial_assignment:
        cmd_parts.append(f'--assignment "{partial_assignment}"')

    return " ".join(cmd_parts)


def process_root_logs(batch_id: int, db_path: str = "experiments.db"):
    """Process resolution logs, build trees, create verifications."""
    try:
        from search import Node, SearchTree
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
                add_unexplored_siblings(tree.root)

                tree_path = trees_dir / f"resolution_{res_id}.pkl"
                with open(tree_path, "wb") as f:
                    pickle.dump(tree.root, f)

                cursor.execute(
                    "UPDATE resolutions SET result = ?, tree_path = ? WHERE id = ?",
                    (result, str(tree_path), res_id),
                )

                # Create verifications for UNKNOWN nodes
                unknown_nodes = find_unknown_nodes(tree.root)
                for node in unknown_nodes:
                    path = node.path_from_root()
                    assignment_parts = [f"{var}{op}{val}" for var, op, val in path]
                    assignment = ",".join(sorted(assignment_parts))

                    cursor.execute(
                        """
                        INSERT INTO verifications
                        (resolution_id, partial_assignment, objective_bound)
                        VALUES (?, ?, ?)
                        """,
                        (res_id, assignment, bound),
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
            if op == "=":
                sibling = Node(
                    decision=(var, "!=", val),
                    label="UNKNOWN",
                    parent=node.parent,
                    depth=node.depth,
                )
                if node.parent:
                    idx = node.parent.children.index(node)
                    node.parent.children.insert(idx + 1, sibling)

        for child in list(node.children):
            process_node(child)

    process_node(root)


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


def submit_verification_batch(
    db_path: str = "experiments.db", batch_size: int = BATCH_SIZE
):
    """Create, generate script, and submit batch for unscheduled verifications."""

    if not queue_empty():
        print("Queue is not empty. Not scheduling new batch.")
        return

    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT v.id, v.resolution_id, v.partial_assignment, v.objective_bound,
                   i.filename
            FROM verifications v
            JOIN resolutions r ON v.resolution_id = r.id
            JOIN instances i ON r.instance_id = i.id
            WHERE v.batch_id IS NULL
            ORDER BY v.id
            LIMIT ?
            """,
            (batch_size,),
        )
        verifications = cursor.fetchall()

        if not verifications:
            print("No unscheduled verifications found")
            return

        # Create batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"verification_{timestamp}"

        cursor.execute(
            "INSERT INTO batches (name, batch_type) VALUES (?, 'verification')",
            (batch_name,),
        )
        batch_id = cursor.lastrowid

        # Assign verifications to batch
        verification_ids = [v[0] for v in verifications]
        cursor.execute(
            f"UPDATE verifications SET batch_id = ? WHERE id IN ({','.join('?' * len(verification_ids))})",
            [batch_id] + verification_ids,
        )

        print(f"Created batch {batch_id}: {batch_name}")
        print(f"  {len(verifications)} verifications")

        # Generate script
        script_path = generate_verification_script(batch_id, batch_name, verifications)

        cursor.execute(
            "UPDATE batches SET script_path = ? WHERE id = ?",
            (str(script_path), batch_id),
        )

        # Create log directory
        log_dir = Path("logs") / batch_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Submit to SLURM
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)], capture_output=True, text=True, check=True
            )

            job_id = result.stdout.strip().split()[-1]

            cursor.execute(
                "UPDATE batches SET submitted_at = CURRENT_TIMESTAMP, slurm_job_id = ? WHERE id = ?",
                (job_id, batch_id),
            )

            print(f"Submitted to SLURM: job {job_id}")

        except subprocess.CalledProcessError as e:
            print(f"Submission failed: {e.stderr or str(e)}")
            return


def generate_verification_script(
    batch_id: int, batch_name: str, verifications: List[Tuple]
) -> Path:
    """Generate SLURM script for verification batch."""

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={batch_name}",
        f"#SBATCH --output=logs/{batch_name}/slurm_%A_%a.out",
        f"#SBATCH --error=logs/{batch_name}/slurm_%A_%a.err",
        f"#SBATCH --array=0-{len(verifications)-1}",
        "#SBATCH --time=02:00:00",
        "#SBATCH --mem=8G",
        "#SBATCH --cpus-per-task=1",
        "",
        "TASK_IDS=(",
    ]

    for ver_id, _, _, _, _ in verifications:
        script_lines.append(f"  {ver_id}")
    script_lines.append(")")

    script_lines.extend(["", "COMMANDS=("])

    for _, _, assignment, bound, filename in verifications:
        cmd = build_solver_command(
            instance_file=filename,
            objective_bound=bound,
            partial_assignment=assignment,
            trace_search=False,
        )
        cmd_escaped = cmd.replace('"', '\\"')
        script_lines.append(f'  "{cmd_escaped}"')

    script_lines.append(")")

    script_lines.extend(["", "LOG_FILES=("])

    for ver_id, _, _, _, _ in verifications:
        log_path = f"logs/{batch_name}/verification_{ver_id}.log"
        script_lines.append(f'  "{log_path}"')

    script_lines.append(")")

    script_lines.extend(
        [
            "",
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

    print(f"Generated script: {script_path}")
    return script_path


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

        cursor.execute("SELECT name FROM batches WHERE id = ?", (batch_id,))
        batch_name = cursor.fetchone()[0]

        for res_id, ver_list in resolution_map.items():
            cursor.execute("SELECT tree_path FROM resolutions WHERE id = ?", (res_id,))
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
                    log_path = Path("logs") / batch_name / f"verification_{ver_id}.log"
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

                    node = find_node_by_path(root, assignment)
                    if node and result in ["SAT", "UNSAT"]:
                        node.label = result

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
            f"{'ID':<5} {'Name':<30} {'Type':<15} {'Script':<7} {'Submit':<7} {'Job ID'}"
        )
        print("-" * 90)

        for batch_id, name, btype, has_script, submitted, job_id in batches:
            script_check = "✓" if has_script else "✗"
            submit_check = "✓" if submitted else "✗"
            job_display = (job_id or "")[:12]

            print(
                f"{batch_id:<5} {name:<30} {btype:<15} {script_check:<7} {submit_check:<7} {job_display}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="XCSP Optimization Gap Testing Framework"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("init-db", help="Initialize database schema")

    scan_parser = subparsers.add_parser(
        "create-instances", help="Scan JSON and create instances and resolutions"
    )
    scan_parser.add_argument("json_file", help="xcsp_solved.json file")

    subparsers.add_parser(
        "submit-resolutions", help="Create and submit resolution batch"
    )

    root_logs_parser = subparsers.add_parser(
        "process-root-logs",
        help="Process resolution logs, build trees, create verifications",
    )
    root_logs_parser.add_argument("batch_id", type=int, help="Batch ID")

    subparsers.add_parser(
        "submit-verifications", help="Create and submit verification batch"
    )

    ver_results_parser = subparsers.add_parser(
        "process-verification-results",
        help="Process verification results and update trees",
    )
    ver_results_parser.add_argument("batch_id", type=int, help="Batch ID")

    subparsers.add_parser("list-batches", help="List all batches")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "init-db":
        init_database()

    elif args.command == "create-instances":
        create_instances_and_resolutions(args.json_file)

    elif args.command == "submit-resolutions":
        submit_resolution_batch()

    elif args.command == "process-root-logs":
        process_root_logs(args.batch_id)

    elif args.command == "submit-verifications":
        submit_verification_batch()

    elif args.command == "process-verification-results":
        process_verification_results(args.batch_id)

    elif args.command == "list-batches":
        list_batches()


if __name__ == "__main__":
    main()

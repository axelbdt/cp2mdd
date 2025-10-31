#!/usr/bin/env python3
"""
XCSP Optimization Gap Testing Framework
Manages root resolutions, subtree testing, and result tracking.
"""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Gap values to test (percentages)
DEFAULT_GAPS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.5, 1]


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


def init_database(db_path: str = "experiments.db", schema_file: str = "create_db.sql"):
    """Initialize database schema."""
    schema_path = Path(schema_file)

    if not schema_path.exists():
        print(f"Error: Schema file not found: {schema_file}")
        return

    schema = schema_path.read_text()

    with db_connection(db_path) as conn:
        conn.executescript(schema)

    print(f"Database initialized: {db_path}")


def scan_instances(json_file: str, db_path: str = "experiments.db"):
    """Import instances from xcsp_solved.json (minimization problems only)."""

    json_path = Path(json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_file}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: Expected JSON array of instances")
        return

    instances = []
    skipped = 0
    for item in data:
        name = item["name"]
        family = item["family"]
        optimal = item["best_bound"]
        type_str = item["type"]

        # Filter: only minimization problems
        if not type_str.startswith("min"):
            skipped += 1
            continue

        # Construct filename
        filename = f"cop/{name}_c25.xml.lzma"

        instances.append((filename, family, optimal))

    if not instances:
        print("No minimization instances found in JSON")
        return

    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        inserted = 0
        for filename, problem_type, optimal in instances:
            try:
                cursor.execute(
                    """
                    INSERT INTO instances (filename, problem_type, optimal)
                    VALUES (?, ?, ?)
                    ON CONFLICT(filename, problem_type) DO UPDATE SET
                        optimal = excluded.optimal
                    """,
                    (filename, problem_type, optimal),
                )
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.IntegrityError as e:
                print(f"Error inserting {filename}: {e}")

        print(
            f"Processed {len(instances)} minimization instances, inserted/updated {inserted}"
        )
        if skipped:
            print(f"Skipped {skipped} non-minimization instances")


def normalize_partial_assignment(assignment: str) -> str:
    """Normalize partial assignment to canonical form (alphabetically sorted)."""
    if not assignment:
        return ""

    # Parse individual assignments
    parts = [p.strip() for p in assignment.split(",")]

    # Sort alphabetically
    parts.sort()

    return ",".join(parts)


def format_partial_assignment_for_solver(assignment: str) -> str:
    """Format partial assignment for solver command line."""
    if not assignment:
        return ""

    # Assuming format: --assignment "x[0,0]!=1,x[0,1]=3"
    # TODO: Verify actual solver syntax when available
    return f'--assignment "{assignment}"'


def calculate_objective_bound(optimal: int, gap: float) -> int:
    """Calculate objective bound based on gap percentage (minimization only)."""
    bound = optimal * (1 + gap)
    return int(bound)


def build_solver_command(
    instance_file: str,
    objective_bound: int,
    partial_assignment: Optional[str] = None,
    timeout: int = 3600,
    solver_jar: str = "minicpbp.jar",
    java_bin: str = "java",
) -> str:
    """Build complete solver command."""

    # Base command
    cmd_parts = [
        java_bin,
        "-jar",
        solver_jar,
        "--input",
        instance_file,
        "--bp-algorithm",
        "max-product",
        "--branching",
        "dom-wdeg",
        "--search-type",
        "dfs",
        "--timeout",
        str(timeout),
        "--trace-search",  # Generates search trace to stdout
        f"--oracle-on-objective {objective_bound}",
    ]

    # Add partial assignment if provided
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

        # Get instances
        query = """
            SELECT id, filename, problem_type, optimal
            FROM instances
        """
        if instance_limit:
            query += f" LIMIT {instance_limit}"

        cursor.execute(query)
        instances = cursor.fetchall()

        if not instances:
            print("No instances found")
            print("Run: schedule.py scan-instances xcsp_solved.json")
            return

        print(f"Found {len(instances)} instances")
        print(f"Testing {len(gaps)} gap values: {gaps}")
        print(f"Total resolutions: {len(instances) * len(gaps)}")

        if dry_run:
            print("\nDRY RUN - Sample resolutions:")
            for i, (inst_id, filename, problem_type, optimal) in enumerate(
                instances[:3]
            ):
                for gap in gaps[:3]:
                    bound = calculate_objective_bound(optimal, gap)
                    print(f"  {filename} @ gap={gap}: bound={bound}")
            return

        # Create batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"root_{timestamp}"

        cursor.execute(
            "INSERT INTO batches (name, batch_type) VALUES (?, 'root')", (batch_name,)
        )
        batch_id = cursor.lastrowid

        # Create resolutions
        resolution_count = 0
        for inst_id, filename, problem_type, optimal in instances:
            for gap in gaps:
                bound = calculate_objective_bound(optimal, gap)
                log_path = f"logs/{batch_name}/{batch_name}_{resolution_count}.log"

                cursor.execute(
                    """
                    INSERT INTO resolutions 
                    (batch_id, instance_id, gap, objective_bound, partial_assignment, log_path)
                    VALUES (?, ?, ?, ?, NULL, ?)
                    """,
                    (batch_id, inst_id, gap, bound, log_path),
                )
                resolution_count += 1

        print(f"Created batch {batch_id}: {batch_name}")
        print(f"  {resolution_count} resolutions scheduled")

        return batch_id


def generate_slurm_script(batch_id: int, db_path: str = "experiments.db") -> Path:
    """Generate SLURM array job script for batch."""

    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Get batch info
        cursor.execute("SELECT name, batch_type FROM batches WHERE id = ?", (batch_id,))
        batch_row = cursor.fetchone()
        if not batch_row:
            raise ValueError(f"Batch {batch_id} not found")

        batch_name, batch_type = batch_row

        # Get resolutions
        cursor.execute(
            """
            SELECT 
                r.id, r.objective_bound, r.partial_assignment, r.log_path,
                i.filename
            FROM resolutions r
            JOIN instances i ON r.instance_id = i.id
            WHERE r.batch_id = ?
            ORDER BY r.id
            """,
            (batch_id,),
        )
        resolutions = cursor.fetchall()

        if not resolutions:
            raise ValueError(f"No resolutions found for batch {batch_id}")

        # Build script content
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
            "# Resolution ID array",
            "RESOLUTION_IDS=(",
        ]

        # Add resolution IDs
        for res_id, _, _, _, _ in resolutions:
            script_lines.append(f"  {res_id}")
        script_lines.append(")")

        script_lines.extend(["", "# Command array", "COMMANDS=("])

        # Add commands
        for res_id, bound, assignment, log_path, filename in resolutions:
            cmd = build_solver_command(
                instance_file=f"data/{filename}",
                objective_bound=bound,
                partial_assignment=assignment,
            )
            # Escape for bash array
            cmd_escaped = cmd.replace('"', '\\"')
            script_lines.append(f'  "{cmd_escaped}"')

        script_lines.append(")")

        script_lines.extend(["", "# Log file array", "LOG_FILES=("])

        for _, _, _, log_path, _ in resolutions:
            script_lines.append(f'  "{log_path}"')

        script_lines.append(")")

        script_lines.extend(
            [
                "",
                "# Execute task",
                "TASK_ID=$SLURM_ARRAY_TASK_ID",
                'RESOLUTION_ID="${RESOLUTION_IDS[$TASK_ID]}"',
                'COMMAND="${COMMANDS[$TASK_ID]}"',
                'LOG_FILE="${LOG_FILES[$TASK_ID]}"',
                "",
                'echo "Resolution ID: $RESOLUTION_ID"',
                'echo "Log file: $LOG_FILE"',
                'echo "Command: $COMMAND"',
                "",
                "# Create log directory",
                'mkdir -p $(dirname "$LOG_FILE")',
                "",
                "# Run solver, redirect trace to log file",
                'eval $COMMAND > "$LOG_FILE" 2>&1',
                "",
                'echo "Completed at $(date)"',
            ]
        )

        # Write script
        script_dir = Path("batches")
        script_dir.mkdir(exist_ok=True)

        script_path = script_dir / f"{batch_name}.sh"
        script_path.write_text("\n".join(script_lines))
        script_path.chmod(0o755)

        # Update database
        cursor.execute(
            "UPDATE batches SET script_path = ? WHERE id = ?",
            (str(script_path), batch_id),
        )

        # Create log directory
        log_dir = Path("logs") / batch_name
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generated script: {script_path}")
        print(f"  Array size: {len(resolutions)}")

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

        # Submit to SLURM
        try:
            result = subprocess.run(
                ["sbatch", script_path], capture_output=True, text=True, check=True
            )

            # Parse job ID
            job_id = result.stdout.strip().split()[-1]

            cursor.execute(
                """
                UPDATE batches
                SET submitted_at = CURRENT_TIMESTAMP,
                    slurm_job_id = ?,
                    submission_attempts = submission_attempts + 1
                WHERE id = ?
                """,
                (job_id, batch_id),
            )

            print(f"Submitted batch {batch_id}: {batch_name}")
            print(f"  SLURM job ID: {job_id}")
            return True

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or str(e)
            cursor.execute(
                """
                UPDATE batches
                SET last_error = ?,
                    submission_attempts = submission_attempts + 1
                WHERE id = ?
                """,
                (error_msg, batch_id),
            )
            print(f"Submission failed: {error_msg}")
            return False


def process_logs(batch_id: int, db_path: str = "experiments.db"):
    """Process search trace logs and populate subtrees table."""

    # Import here to use the existing search.py parser
    try:
        from search import SearchTree, extract_paths
    except ImportError:
        print("Error: search.py not found in current directory")
        print("Ensure search.py is available for parsing search traces")
        return

    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Get batch info
        cursor.execute(
            "SELECT name, logs_processed FROM batches WHERE id = ?", (batch_id,)
        )
        batch_row = cursor.fetchone()
        if not batch_row:
            print(f"Batch {batch_id} not found")
            return

        batch_name, logs_processed = batch_row

        if logs_processed:
            print(f"Batch {batch_id} already processed")
            return

        # Get resolutions
        cursor.execute(
            """
            SELECT r.id, r.instance_id, r.log_path, r.objective_bound
            FROM resolutions r
            WHERE r.batch_id = ?
            """,
            (batch_id,),
        )
        resolutions = cursor.fetchall()

        print(f"Processing {len(resolutions)} log files...")

        processed = 0
        errors = 0
        subtrees_found = 0

        for res_id, inst_id, log_path, bound in resolutions:
            log_file = Path(log_path)

            if not log_file.exists():
                print(f"  Warning: Log file not found: {log_path}")
                continue

            try:
                # Read log
                log_text = log_file.read_text()

                # Determine result
                if "SAT" in log_text:
                    result = "SAT"
                elif "UNSAT" in log_text:
                    result = "UNSAT"
                elif "TIMEOUT" in log_text or "timeout" in log_text.lower():
                    result = "TIMEOUT"
                else:
                    result = "ERROR"

                # Update resolution
                cursor.execute(
                    "UPDATE resolutions SET result = ? WHERE id = ?", (result, res_id)
                )

                # Parse search tree
                tree = SearchTree.from_log(log_text)

                # Extract all paths with positive decisions only
                for path, label in extract_paths(tree.root, positive_only=True):
                    if not path:  # Skip empty paths
                        continue

                    # Format partial assignment
                    assignment_parts = [f"{var}{op}{val}" for var, op, val in path]
                    assignment = ",".join(assignment_parts)
                    normalized = normalize_partial_assignment(assignment)

                    # Update or insert subtree
                    cursor.execute(
                        """
                        INSERT INTO subtrees (instance_id, partial_assignment, lowest_sat, highest_unsat)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(instance_id, partial_assignment) DO UPDATE SET
                            lowest_sat = CASE 
                                WHEN ? = 'SAT' THEN MIN(COALESCE(lowest_sat, 999999999), ?)
                                ELSE lowest_sat
                            END,
                            highest_unsat = CASE
                                WHEN ? = 'UNSAT' THEN MAX(COALESCE(highest_unsat, -999999999), ?)
                                ELSE highest_unsat
                            END,
                            last_tested_at = CURRENT_TIMESTAMP
                        """,
                        (
                            inst_id,
                            normalized,
                            bound if label == "SAT" else None,
                            bound if label == "UNSAT" else None,
                            label,
                            bound,
                            label,
                            bound,
                        ),
                    )
                    subtrees_found += 1

                processed += 1

            except Exception as e:
                print(f"  Error processing {log_path}: {e}")
                cursor.execute(
                    "UPDATE resolutions SET result = 'ERROR' WHERE id = ?", (res_id,)
                )
                errors += 1

        # Mark batch as processed
        cursor.execute(
            "UPDATE batches SET logs_processed = 1 WHERE id = ?", (batch_id,)
        )

        print(f"Processed {processed} logs, {errors} errors")
        print(f"Found {subtrees_found} subtree paths")


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
                logs_processed,
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
            f"{'ID':<5} {'Name':<25} {'Type':<8} {'Script':<7} {'Submit':<7} {'Logs':<7} {'Job ID'}"
        )
        print("-" * 90)

        for batch_id, name, btype, has_script, submitted, logs_proc, job_id in batches:
            script_check = "✓" if has_script else "✗"
            submit_check = "✓" if submitted else "✗"
            logs_check = "✓" if logs_proc else "✗"
            job_display = (job_id or "")[:12]

            print(
                f"{batch_id:<5} {name:<25} {btype:<8} {script_check:<7} {submit_check:<7} {logs_check:<7} {job_display}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="XCSP Optimization Gap Testing Framework"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init-db
    subparsers.add_parser("init-db", help="Initialize database schema")

    # scan-instances
    scan_parser = subparsers.add_parser(
        "scan-instances", help="Import instances from xcsp_solved.json"
    )
    scan_parser.add_argument("json_file", help="Path to xcsp_solved.json")

    # create-root-batch
    root_parser = subparsers.add_parser(
        "create-root-batch", help="Create root resolution batch"
    )
    root_parser.add_argument("--limit", type=int, help="Limit number of instances")
    root_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )

    # generate-script
    gen_parser = subparsers.add_parser(
        "generate-script", help="Generate SLURM script for batch"
    )
    gen_parser.add_argument("batch_id", type=int, help="Batch ID")

    # submit-batch
    submit_parser = subparsers.add_parser("submit-batch", help="Submit batch to SLURM")
    submit_parser.add_argument("batch_id", type=int, help="Batch ID")
    submit_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )

    # process-logs
    logs_parser = subparsers.add_parser(
        "process-logs", help="Process search trace logs"
    )
    logs_parser.add_argument("batch_id", type=int, help="Batch ID")

    # list-batches
    subparsers.add_parser("list-batches", help="List all batches")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == "init-db":
        init_database()

    elif args.command == "scan-instances":
        scan_instances(args.json_file)

    elif args.command == "create-root-batch":
        create_root_batch(instance_limit=args.limit, dry_run=args.dry_run)

    elif args.command == "generate-script":
        generate_slurm_script(args.batch_id)

    elif args.command == "submit-batch":
        submit_batch(args.batch_id, dry_run=args.dry_run)

    elif args.command == "process-logs":
        process_logs(args.batch_id)

    elif args.command == "list-batches":
        list_batches()


if __name__ == "__main__":
    main()

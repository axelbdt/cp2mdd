#!/usr/bin/env python3
"""
XCSP Optimization Gap Testing Framework
Manages root resolutions, subtree testing, and result tracking.
"""

import argparse
import re
import sqlite3
import subprocess
import sys
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
        optimal_score INTEGER,  -- NULL until populated
        highest_unsat INTEGER,
        lowest_sat INTEGER,
        UNIQUE (filename, problem_type)
    );

    CREATE TABLE IF NOT EXISTS batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        batch_type TEXT NOT NULL,  -- 'root' or 'subtree'
        script_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        submitted_at TIMESTAMP,
        submission_attempts INTEGER DEFAULT 0,
        last_error TEXT,
        slurm_job_id TEXT,
        logs_processed BOOLEAN DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS resolutions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL,
        instance_id INTEGER NOT NULL,
        gap REAL NOT NULL,
        objective_bound INTEGER NOT NULL,
        partial_assignment TEXT,  -- NULL for root, comma-separated for subtrees
        log_path TEXT NOT NULL,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        timeout BOOLEAN DEFAULT 0,
        result TEXT,  -- 'SAT', 'UNSAT', 'TIMEOUT', 'ERROR', NULL
        FOREIGN KEY (batch_id) REFERENCES batches(id),
        FOREIGN KEY (instance_id) REFERENCES instances(id)
    );

    CREATE TABLE IF NOT EXISTS subtrees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instance_id INTEGER NOT NULL,
        partial_assignment TEXT NOT NULL,  -- normalized: alphabetically sorted
        lowest_sat_bound INTEGER,
        highest_unsat_bound INTEGER,
        last_tested_at TIMESTAMP,
        FOREIGN KEY (instance_id) REFERENCES instances(id),
        UNIQUE (instance_id, partial_assignment)
    );

    CREATE INDEX IF NOT EXISTS idx_resolutions_batch ON resolutions(batch_id);
    CREATE INDEX IF NOT EXISTS idx_resolutions_instance ON resolutions(instance_id);
    CREATE INDEX IF NOT EXISTS idx_subtrees_instance ON subtrees(instance_id);
    """
    
    with db_connection(db_path) as conn:
        conn.executescript(schema)
    
    print(f"Database initialized: {db_path}")


def scan_instances(data_dir: str, db_path: str = "experiments.db"):
    """Scan directory structure and populate instances table."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    instances = []
    
    # Recursively find all .xml files (XCSP format)
    for xml_file in data_path.rglob("*.xml"):
        # Extract problem type from directory structure
        # Assumes structure like: data/{problem_type}/instance.xml
        relative = xml_file.relative_to(data_path)
        problem_type = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        
        instances.append((str(relative), problem_type))
    
    if not instances:
        print(f"No XCSP files found in {data_dir}")
        return
    
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        inserted = 0
        for filename, problem_type in instances:
            try:
                cursor.execute(
                    """
                    INSERT INTO instances (filename, problem_type)
                    VALUES (?, ?)
                    ON CONFLICT(filename, problem_type) DO NOTHING
                    """,
                    (filename, problem_type)
                )
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.IntegrityError:
                pass
        
        print(f"Scanned {len(instances)} files, inserted {inserted} new instances")


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


def calculate_objective_bound(optimal_score: int, gap: float) -> int:
    """Calculate objective bound based on gap percentage."""
    if optimal_score >= 0:
        bound = optimal_score * (1 + gap)
    else:
        bound = optimal_score * (1 - gap)
    
    return int(bound)


def build_solver_command(
    instance_file: str,
    objective_bound: int,
    partial_assignment: Optional[str] = None,
    timeout: int = 3600,
    solver_jar: str = "minicpbp.jar",
    java_bin: str = "java"
) -> str:
    """Build complete solver command."""
    
    # Base command
    cmd_parts = [
        java_bin,
        "-jar", solver_jar,
        "--input", instance_file,
        "--bp-algorithm", "max-product",
        "--branching", "dom-wdeg",
        "--search-type", "dfs",
        "--timeout", str(timeout),
        "--trace-search",  # Generates search trace to stdout
        f"--oracle-on-objective {objective_bound}"
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
    dry_run: bool = False
):
    """Create batch for root resolutions (testing instances at various gaps)."""
    
    if gaps is None:
        gaps = DEFAULT_GAPS
    
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Get instances with optimal scores
        query = """
            SELECT id, filename, problem_type, optimal_score
            FROM instances
            WHERE optimal_score IS NOT NULL
        """
        if instance_limit:
            query += f" LIMIT {instance_limit}"
        
        cursor.execute(query)
        instances = cursor.fetchall()
        
        if not instances:
            print("No instances with optimal scores found")
            print("Run: schedule.py import-optimal-scores <csv_file>")
            return
        
        print(f"Found {len(instances)} instances with optimal scores")
        print(f"Testing {len(gaps)} gap values: {gaps}")
        print(f"Total resolutions: {len(instances) * len(gaps)}")
        
        if dry_run:
            print("\nDRY RUN - Sample resolutions:")
            for i, (inst_id, filename, problem_type, optimal) in enumerate(instances[:3]):
                for gap in gaps[:3]:
                    bound = calculate_objective_bound(optimal, gap)
                    print(f"  {filename} @ gap={gap}: bound={bound}")
            return
        
        # Create batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"root_{timestamp}"
        
        cursor.execute(
            "INSERT INTO batches (name, batch_type) VALUES (?, 'root')",
            (batch_name,)
        )
        batch_id = cursor.lastrowid
        
        # Create resolutions
        resolution_count = 0
        for inst_id, filename, problem_type, optimal_score in instances:
            for gap in gaps:
                bound = calculate_objective_bound(optimal_score, gap)
                log_path = f"logs/{batch_name}/{batch_name}_{resolution_count}.log"
                
                cursor.execute(
                    """
                    INSERT INTO resolutions 
                    (batch_id, instance_id, gap, objective_bound, partial_assignment, log_path)
                    VALUES (?, ?, ?, ?, NULL, ?)
                    """,
                    (batch_id, inst_id, gap, bound, log_path)
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
        cursor.execute(
            "SELECT name, batch_type FROM batches WHERE id = ?",
            (batch_id,)
        )
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
            (batch_id,)
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
            "RESOLUTION_IDS=("
        ]
        
        # Add resolution IDs
        for res_id, _, _, _, _ in resolutions:
            script_lines.append(f"  {res_id}")
        script_lines.append(")")
        
        script_lines.extend([
            "",
            "# Command array",
            "COMMANDS=("
        ])
        
        # Add commands
        for res_id, bound, assignment, log_path, filename in resolutions:
            cmd = build_solver_command(
                instance_file=f"data/{filename}",
                objective_bound=bound,
                partial_assignment=assignment
            )
            # Escape for bash array
            cmd_escaped = cmd.replace('"', '\\"')
            script_lines.append(f'  "{cmd_escaped}"')
        
        script_lines.append(")")
        
        script_lines.extend([
            "",
            "# Log file array",
            "LOG_FILES=("
        ])
        
        for _, _, _, log_path, _ in resolutions:
            script_lines.append(f'  "{log_path}"')
        
        script_lines.append(")")
        
        script_lines.extend([
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
        ])
        
        # Write script
        script_dir = Path("batches")
        script_dir.mkdir(exist_ok=True)
        
        script_path = script_dir / f"{batch_name}.sh"
        script_path.write_text("\n".join(script_lines))
        script_path.chmod(0o755)
        
        # Update database
        cursor.execute(
            "UPDATE batches SET script_path = ? WHERE id = ?",
            (str(script_path), batch_id)
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
            "SELECT name, script_path FROM batches WHERE id = ?",
            (batch_id,)
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
                ["sbatch", script_path],
                capture_output=True,
                text=True,
                check=True
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
                (job_id, batch_id)
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
                (error_msg, batch_id)
            )
            print(f"Submission failed: {error_msg}")
            return False


def process_logs(batch_id: int, db_path: str = "experiments.db"):
    """Process search trace logs and populate subtrees table."""
    
    # Import here to use the existing search.py parser
    try:
        from search import SearchTree
    except ImportError:
        print("Error: search.py not found in current directory")
        print("Ensure search.py is available for parsing search traces")
        return
    
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Get batch info
        cursor.execute(
            "SELECT name, logs_processed FROM batches WHERE id = ?",
            (batch_id,)
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
            (batch_id,)
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
                    "UPDATE resolutions SET result = ? WHERE id = ?",
                    (result, res_id)
                )
                
                # Parse search tree
                tree = SearchTree.from_log(log_text)
                
                # Extract all paths with positive decisions only
                for path, label in tree.extract_positive_only_paths():
                    if not path:  # Skip empty paths
                        continue
                    
                    # Format partial assignment
                    assignment_parts = [f"{var}{op}{val}" for var, op, val in path]
                    assignment = ",".join(assignment_parts)
                    normalized = normalize_partial_assignment(assignment)
                    
                    # Update or insert subtree
                    cursor.execute(
                        """
                        INSERT INTO subtrees (instance_id, partial_assignment, lowest_sat_bound, highest_unsat_bound)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(instance_id, partial_assignment) DO UPDATE SET
                            lowest_sat_bound = CASE 
                                WHEN ? = 'SAT' THEN MIN(COALESCE(lowest_sat_bound, 999999999), ?)
                                ELSE lowest_sat_bound
                            END,
                            highest_unsat_bound = CASE
                                WHEN ? = 'UNSAT' THEN MAX(COALESCE(highest_unsat_bound, -999999999), ?)
                                ELSE highest_unsat_bound
                            END,
                            last_tested_at = CURRENT_TIMESTAMP
                        """,
                        (
                            inst_id, normalized,
                            bound if label == 'SAT' else None,
                            bound if label == 'UNSAT' else None,
                            label, bound,
                            label, bound
                        )
                    )
                    subtrees_found += 1
                
                processed += 1
                
            except Exception as e:
                print(f"  Error processing {log_path}: {e}")
                cursor.execute(
                    "UPDATE resolutions SET result = 'ERROR' WHERE id = ?",
                    (res_id,)
                )
                errors += 1
        
        # Mark batch as processed
        cursor.execute(
            "UPDATE batches SET logs_processed = 1 WHERE id = ?",
            (batch_id,)
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
        
        print(f"{'ID':<5} {'Name':<25} {'Type':<8} {'Script':<7} {'Submit':<7} {'Logs':<7} {'Job ID'}")
        print("-" * 90)
        
        for batch_id, name, btype, has_script, submitted, logs_proc, job_id in batches:
            script_check = "✓" if has_script else "✗"
            submit_check = "✓" if submitted else "✗"
            logs_check = "✓" if logs_proc else "✗"
            job_display = (job_id or "")[:12]
            
            print(f"{batch_id:<5} {name:<25} {btype:<8} {script_check:<7} {submit_check:<7} {logs_check:<7} {job_display}")


def import_optimal_scores(csv_file: str, db_path: str = "experiments.db"):
    """Import optimal scores from CSV file.
    
    Expected CSV format:
    filename,optimal_score
    problem1.xml,42
    problem2.xml,137
    """
    import csv
    
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_file}")
        return
    
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            updated = 0
            for row in reader:
                filename = row['filename']
                optimal_score = int(row['optimal_score'])
                
                cursor.execute(
                    """
                    UPDATE instances
                    SET optimal_score = ?
                    WHERE filename = ?
                    """,
                    (optimal_score, filename)
                )
                
                if cursor.rowcount > 0:
                    updated += 1
        
        print(f"Updated {updated} instances with optimal scores")


def main():
    parser = argparse.ArgumentParser(
        description="XCSP Optimization Gap Testing Framework"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # init-db
    subparsers.add_parser("init-db", help="Initialize database schema")
    
    # scan-instances
    scan_parser = subparsers.add_parser("scan-instances", help="Scan data directory for instances")
    scan_parser.add_argument("data_dir", help="Root data directory")
    
    # import-optimal-scores
    import_parser = subparsers.add_parser("import-optimal-scores", help="Import optimal scores from CSV")
    import_parser.add_argument("csv_file", help="CSV file with filename,optimal_score")
    
    # create-root-batch
    root_parser = subparsers.add_parser("create-root-batch", help="Create root resolution batch")
    root_parser.add_argument("--limit", type=int, help="Limit number of instances")
    root_parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    
    # generate-script
    gen_parser = subparsers.add_parser("generate-script", help="Generate SLURM script for batch")
    gen_parser.add_argument("batch_id", type=int, help="Batch ID")
    
    # submit-batch
    submit_parser = subparsers.add_parser("submit-batch", help="Submit batch to SLURM")
    submit_parser.add_argument("batch_id", type=int, help="Batch ID")
    submit_parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    
    # process-logs
    logs_parser = subparsers.add_parser("process-logs", help="Process search trace logs")
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
        scan_instances(args.data_dir)
    
    elif args.command == "import-optimal-scores":
        import_optimal_scores(args.csv_file)
    
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

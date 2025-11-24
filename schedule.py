#!/usr/bin/env python3
"""Minimal state-driven CSP experiment scheduler with timeout retry support."""

import json
import pickle
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from search import Node, SearchTree

# Constants
DB_PATH = "experiments.db"
GAPS = [0, 0.5, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
TIMEOUT = 3600
MAX_TIMEOUT = 14400  # 4 hours - don't retry beyond this
SOLVER_JAR = "minicpbp.jar"


@contextmanager
def db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
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
        batch_id INTEGER NOT NULL,
        instance_id INTEGER NOT NULL,
        gap REAL NOT NULL,
        objective_bound INTEGER NOT NULL,
        timeout_seconds INTEGER NOT NULL DEFAULT 3600,
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
        timeout_seconds INTEGER NOT NULL DEFAULT 3600,
        result TEXT,
        FOREIGN KEY (batch_id) REFERENCES batches(id),
        FOREIGN KEY (resolution_id) REFERENCES resolutions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_resolutions_batch ON resolutions(batch_id);
    CREATE INDEX IF NOT EXISTS idx_resolutions_instance ON resolutions(instance_id);
    CREATE INDEX IF NOT EXISTS idx_verifications_batch ON verifications(batch_id);
    CREATE INDEX IF NOT EXISTS idx_verifications_resolution ON verifications(resolution_id);
    """
    with db_connection() as conn:
        conn.executescript(schema)
    print(f"Database initialized: {DB_PATH}")


def scan_instances():
    json_path = Path("xcsp_solved.json")
    if not json_path.exists():
        print(f"Error: xcsp_solved.json not found")
        return 0

    with open(json_path) as f:
        data = json.load(f)

    instances = []
    for entry in data:
        name = entry["name"]
        filename = f"cop/{name}_c25.xml.lzma"
        family = entry.get("family", "unknown")
        optimal = entry["best_bound"]
        instances.append((filename, family, optimal))

    with db_connection() as conn:
        cursor = conn.cursor()
        inserted = 0
        for filename, family, optimal in instances:
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

    print(f"Scanned {len(instances)} instances, inserted/updated {inserted}")
    return inserted


def calculate_objective_bound(optimal: int, gap: float) -> int:
    if optimal >= 0:
        bound = optimal * (1 + gap)
    else:
        bound = optimal * (1 - gap)
    return int(bound)


def create_root_batch():
    with db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT id, filename, problem_type, optimal FROM instances")
        instances = cursor.fetchall()

        if not instances:
            print("No instances found")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"root_{timestamp}"

        cursor.execute(
            "INSERT INTO batches (name, batch_type) VALUES (?, 'root')", (batch_name,)
        )
        batch_id = cursor.lastrowid

        resolution_count = 0
        for inst_id, filename, problem_type, optimal in instances:
            for gap in GAPS:
                bound = calculate_objective_bound(optimal, gap)
                log_path = f"logs/{batch_name}/resolution_{resolution_count}.log"

                cursor.execute(
                    """
                    INSERT INTO resolutions 
                    (batch_id, instance_id, gap, objective_bound, timeout_seconds, log_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (batch_id, inst_id, gap, bound, TIMEOUT, log_path),
                )
                resolution_count += 1

        print(f"Created batch {batch_id}: {batch_name}")
        print(f"  {resolution_count} resolutions")
        return batch_id


def build_solver_command(
    instance_file: str,
    objective_bound: int,
    timeout_seconds: int,
    partial_assignment: Optional[str] = None,
    trace_search: bool = True,
) -> str:
    if instance_file.endswith(".lzma"):
        cmd_parts = [
            f"lzcat {instance_file} > /tmp/instance_$SLURM_ARRAY_TASK_ID.xml &&",
            "java",
            "-jar",
            SOLVER_JAR,
            "--input",
            "/tmp/instance_$SLURM_ARRAY_TASK_ID.xml",
        ]
    else:
        cmd_parts = ["java", "-jar", SOLVER_JAR, "--input", instance_file]

    cmd_parts.extend(
        [
            "--bp-algorithm",
            "max-product",
            "--branching",
            "dom-wdeg",
            "--search-type",
            "dfs",
            "--timeout",
            str(timeout_seconds),
            f"--oracle-on-objective {objective_bound}",
        ]
    )

    if trace_search:
        cmd_parts.append("--trace-search")

    if partial_assignment:
        cmd_parts.append(f'--assignment "{partial_assignment}"')

    return " ".join(cmd_parts)


def generate_script(batch_id: int) -> Path:
    with db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT name, batch_type FROM batches WHERE id = ?", (batch_id,))
        batch_row = cursor.fetchone()
        if not batch_row:
            raise ValueError(f"Batch {batch_id} not found")

        batch_name, batch_type = batch_row

        if batch_type in ["root", "root_retry"]:
            cursor.execute(
                """
                SELECT r.id, r.objective_bound, r.timeout_seconds, r.log_path, i.filename
                FROM resolutions r
                JOIN instances i ON r.instance_id = i.id
                WHERE r.batch_id = ?
                ORDER BY r.id
                """,
                (batch_id,),
            )
        else:  # verification or verification_retry
            cursor.execute(
                """
                SELECT v.id, v.objective_bound, v.timeout_seconds, v.partial_assignment,
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
        if not rows:
            raise ValueError(f"No tasks for batch {batch_id}")

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
            "TASK_IDS=(",
        ]

        for row in rows:
            script_lines.append(f"  {row[0]}")
        script_lines.append(")")

        script_lines.extend(["", "COMMANDS=("])

        if batch_type in ["root", "root_retry"]:
            for task_id, bound, timeout, log_path, filename in rows:
                cmd = build_solver_command(filename, bound, timeout, trace_search=True)
                script_lines.append(f'  "{cmd.replace(chr(34), chr(92)+chr(34))}"')
        else:
            for task_id, bound, timeout, assignment, filename, _ in rows:
                cmd = build_solver_command(
                    filename, bound, timeout, assignment, trace_search=False
                )
                script_lines.append(f'  "{cmd.replace(chr(34), chr(92)+chr(34))}"')

        script_lines.append(")")
        script_lines.extend(["", "LOG_FILES=("])

        if batch_type in ["root", "root_retry"]:
            for _, _, _, log_path, _ in rows:
                script_lines.append(f'  "{log_path}"')
        else:
            for task_id, _, _, _, _, base_log_path in rows:
                log_dir = Path(base_log_path).parent
                log_path = log_dir / f"verification_{task_id}.log"
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
                'mkdir -p $(dirname "$LOG_FILE")',
                'eval $COMMAND > "$LOG_FILE" 2>&1',
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


def submit_batch(batch_id: int):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, script_path FROM batches WHERE id = ?", (batch_id,)
        )
        batch_row = cursor.fetchone()
        if not batch_row:
            raise ValueError(f"Batch {batch_id} not found")

        batch_name, script_path = batch_row
        if not script_path or not Path(script_path).exists():
            raise ValueError(f"Script not found: {script_path}")

        result = subprocess.run(
            ["sbatch", script_path], capture_output=True, text=True, check=True
        )
        job_id = result.stdout.strip().split()[-1]

        cursor.execute(
            "UPDATE batches SET submitted_at = CURRENT_TIMESTAMP, slurm_job_id = ? WHERE id = ?",
            (job_id, batch_id),
        )

        print(f"Submitted batch {batch_id}: {batch_name}")
        print(f"  SLURM job ID: {job_id}")


def add_unexplored_siblings(root: Node) -> Node:
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
    return root


def process_root_logs(batch_id: int):
    with db_connection() as conn:
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

        print(f"Processing {len(resolutions)} resolution logs")

        trees_dir = Path("trees")
        trees_dir.mkdir(exist_ok=True)

        processed = 0
        errors = 0

        for res_id, inst_id, log_path, bound in resolutions:
            log_file = Path(log_path)
            if not log_file.exists():
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

                if result in ["SAT", "UNSAT"]:
                    tree = SearchTree.from_log(log_text)
                    tree_with_unknowns = add_unexplored_siblings(tree.root)

                    tree_path = trees_dir / f"resolution_{res_id}.pkl"
                    with open(tree_path, "wb") as f:
                        pickle.dump(tree_with_unknowns, f)

                    cursor.execute(
                        "UPDATE resolutions SET result = ?, tree_path = ? WHERE id = ?",
                        (result, str(tree_path), res_id),
                    )
                else:
                    # TIMEOUT or ERROR - no tree
                    cursor.execute(
                        "UPDATE resolutions SET result = ? WHERE id = ?",
                        (result, res_id),
                    )

                processed += 1

            except Exception as e:
                print(f"  Error processing {log_path}: {e}")
                cursor.execute(
                    "UPDATE resolutions SET result = 'ERROR' WHERE id = ?", (res_id,)
                )
                errors += 1

        print(f"Processed {processed} logs, {errors} errors")


def create_root_retry_batch(original_batch_id: int):
    with db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT r.id, r.instance_id, r.gap, r.objective_bound, r.timeout_seconds
            FROM resolutions r
            WHERE r.batch_id = ? AND r.result = 'TIMEOUT'
            """,
            (original_batch_id,),
        )
        timeouts = cursor.fetchall()

        if not timeouts:
            print("No timeouts to retry")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"root_retry_{timestamp}"

        cursor.execute(
            "INSERT INTO batches (name, batch_type) VALUES (?, 'root_retry')",
            (batch_name,),
        )
        retry_batch_id = cursor.lastrowid

        retry_count = 0
        skip_count = 0

        for res_id, inst_id, gap, bound, old_timeout in timeouts:
            new_timeout = old_timeout * 2

            if new_timeout > MAX_TIMEOUT:
                skip_count += 1
                continue

            new_log_path = f"logs/{batch_name}/resolution_{res_id}.log"

            cursor.execute(
                """
                UPDATE resolutions
                SET batch_id = ?,
                    timeout_seconds = ?,
                    log_path = ?,
                    result = NULL,
                    tree_path = NULL
                WHERE id = ?
                """,
                (retry_batch_id, new_timeout, new_log_path, res_id),
            )
            retry_count += 1

        print(f"Created retry batch {retry_batch_id}: {batch_name}")
        print(f"  {retry_count} retries, {skip_count} skipped (max timeout reached)")
        return retry_batch_id


def find_unknown_nodes(root: Node) -> List[Node]:
    unknown = []

    def traverse(node):
        if node.label == "UNKNOWN":
            unknown.append(node)
        for child in node.children:
            traverse(child)

    traverse(root)
    return unknown


def normalize_partial_assignment(assignment: str) -> str:
    if not assignment:
        return ""
    parts = [p.strip() for p in assignment.split(",")]
    parts.sort()
    return ",".join(parts)


def create_verification_batch():
    with db_connection() as conn:
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
            return None

        print(f"Scanning {len(resolutions)} trees for UNKNOWN nodes")

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
                        (batch_id, resolution_id, partial_assignment, objective_bound, timeout_seconds)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (batch_id, res_id, normalized, bound, TIMEOUT),
                    )
                    verification_count += 1

            except Exception as e:
                print(f"  Error scanning tree {tree_path}: {e}")
                continue

        print(f"Created batch {batch_id}: {batch_name}")
        print(f"  {verification_count} verifications")
        return batch_id


def find_node_by_path(root: Node, partial_assignment: str) -> Optional[Node]:
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


def process_verification_results(batch_id: int):
    with db_connection() as conn:
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

        print(f"Processing {len(verifications)} verification results")

        resolution_map = {}
        for ver_id, res_id, assignment in verifications:
            if res_id not in resolution_map:
                resolution_map[res_id] = []
            resolution_map[res_id].append((ver_id, assignment))

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

                    if result in ["SAT", "UNSAT"]:
                        node = find_node_by_path(root, assignment)
                        if node:
                            node.label = result

                with open(tree_path, "wb") as f:
                    pickle.dump(root, f)

            except Exception as e:
                print(f"  Error updating tree {tree_path}: {e}")
                continue

        print("Verification results processed")


def create_verification_retry_batch(original_batch_id: int):
    with db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT v.id, v.resolution_id, v.partial_assignment, 
                   v.objective_bound, v.timeout_seconds
            FROM verifications v
            WHERE v.batch_id = ? AND v.result = 'TIMEOUT'
            """,
            (original_batch_id,),
        )
        timeouts = cursor.fetchall()

        if not timeouts:
            print("No verification timeouts to retry")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"verification_retry_{timestamp}"

        cursor.execute(
            "INSERT INTO batches (name, batch_type) VALUES (?, 'verification_retry')",
            (batch_name,),
        )
        retry_batch_id = cursor.lastrowid

        retry_count = 0
        skip_count = 0

        for ver_id, res_id, assignment, bound, old_timeout in timeouts:
            new_timeout = old_timeout * 2

            if new_timeout > MAX_TIMEOUT:
                skip_count += 1
                continue

            cursor.execute(
                """
                UPDATE verifications
                SET batch_id = ?,
                    timeout_seconds = ?,
                    result = NULL
                WHERE id = ?
                """,
                (retry_batch_id, new_timeout, ver_id),
            )
            retry_count += 1

        print(f"Created verification retry batch {retry_batch_id}: {batch_name}")
        print(f"  {retry_count} retries, {skip_count} skipped (max timeout reached)")
        return retry_batch_id


def has_running_jobs() -> bool:
    """Check if any SLURM jobs are running for current user."""
    try:
        user = subprocess.check_output(["whoami"], text=True).strip()
        result = subprocess.run(
            ["squeue", "-u", user], capture_output=True, text=True, timeout=10
        )
        lines = [line for line in result.stdout.strip().split("\n") if line]
        return len(lines) > 1
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        return True


def detect_state():
    """Determine current workflow state from filesystem."""
    db_exists = Path(DB_PATH).exists()

    if not db_exists:
        return "init_db"

    with db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM instances")
        instance_count = cursor.fetchone()[0]

        if instance_count == 0:
            return "scan_instances"

        cursor.execute(
            "SELECT id FROM batches WHERE batch_type = 'root' ORDER BY created_at DESC LIMIT 1"
        )
        latest_root_batch = cursor.fetchone()

        if not latest_root_batch:
            return "create_root_batch"

        batch_id = latest_root_batch[0]

        cursor.execute(
            "SELECT script_path, submitted_at FROM batches WHERE id = ?", (batch_id,)
        )
        script_path, submitted_at = cursor.fetchone()

        if not script_path:
            return f"generate_root_script:{batch_id}"

        if not submitted_at:
            return f"submit_root_batch:{batch_id}"

        if has_running_jobs():
            return f"wait_root_batch:{batch_id}"

        cursor.execute(
            "SELECT COUNT(*) FROM resolutions WHERE batch_id = ? AND result IS NULL",
            (batch_id,),
        )
        unprocessed_logs = cursor.fetchone()[0]

        if unprocessed_logs > 0:
            return f"process_root_logs:{batch_id}"

        # Check for root timeouts needing retry
        cursor.execute(
            "SELECT COUNT(*) FROM resolutions WHERE batch_id = ? AND result = 'TIMEOUT'",
            (batch_id,),
        )
        if cursor.fetchone()[0] > 0:
            return f"create_root_retry_batch:{batch_id}"

        # Check for root retry batch
        cursor.execute(
            "SELECT id FROM batches WHERE batch_type = 'root_retry' ORDER BY created_at DESC LIMIT 1"
        )
        retry_batch = cursor.fetchone()

        if retry_batch:
            retry_batch_id = retry_batch[0]

            cursor.execute(
                "SELECT script_path, submitted_at FROM batches WHERE id = ?",
                (retry_batch_id,),
            )
            retry_script_path, retry_submitted_at = cursor.fetchone()

            if not retry_script_path:
                return f"generate_root_script:{retry_batch_id}"

            if not retry_submitted_at:
                return f"submit_root_batch:{retry_batch_id}"

            if has_running_jobs():
                return f"wait_root_batch:{retry_batch_id}"

            cursor.execute(
                "SELECT COUNT(*) FROM resolutions WHERE batch_id = ? AND result IS NULL",
                (retry_batch_id,),
            )
            if cursor.fetchone()[0] > 0:
                return f"process_root_logs:{retry_batch_id}"

            # Check if retry batch has more timeouts
            cursor.execute(
                "SELECT COUNT(*) FROM resolutions WHERE batch_id = ? AND result = 'TIMEOUT'",
                (retry_batch_id,),
            )
            if cursor.fetchone()[0] > 0:
                return f"create_root_retry_batch:{retry_batch_id}"

        cursor.execute(
            "SELECT id FROM batches WHERE batch_type = 'verification' ORDER BY created_at DESC LIMIT 1"
        )
        latest_ver_batch = cursor.fetchone()

        if not latest_ver_batch:
            return "create_verification_batch"

        ver_batch_id = latest_ver_batch[0]

        cursor.execute(
            "SELECT script_path, submitted_at FROM batches WHERE id = ?",
            (ver_batch_id,),
        )
        ver_script_path, ver_submitted_at = cursor.fetchone()

        if not ver_script_path:
            return f"generate_verification_script:{ver_batch_id}"

        if not ver_submitted_at:
            return f"submit_verification_batch:{ver_batch_id}"

        if has_running_jobs():
            return f"wait_verification_batch:{ver_batch_id}"

        cursor.execute(
            "SELECT COUNT(*) FROM verifications WHERE batch_id = ? AND result IS NULL",
            (ver_batch_id,),
        )
        if cursor.fetchone()[0] > 0:
            return f"process_verification_results:{ver_batch_id}"

        # Check for verification timeouts needing retry
        cursor.execute(
            "SELECT COUNT(*) FROM verifications WHERE batch_id = ? AND result = 'TIMEOUT'",
            (ver_batch_id,),
        )
        if cursor.fetchone()[0] > 0:
            return f"create_verification_retry_batch:{ver_batch_id}"

        # Check for verification retry batch
        cursor.execute(
            "SELECT id FROM batches WHERE batch_type = 'verification_retry' ORDER BY created_at DESC LIMIT 1"
        )
        ver_retry_batch = cursor.fetchone()

        if ver_retry_batch:
            ver_retry_batch_id = ver_retry_batch[0]

            cursor.execute(
                "SELECT script_path, submitted_at FROM batches WHERE id = ?",
                (ver_retry_batch_id,),
            )
            ver_retry_script_path, ver_retry_submitted_at = cursor.fetchone()

            if not ver_retry_script_path:
                return f"generate_verification_script:{ver_retry_batch_id}"

            if not ver_retry_submitted_at:
                return f"submit_verification_batch:{ver_retry_batch_id}"

            if has_running_jobs():
                return f"wait_verification_batch:{ver_retry_batch_id}"

            cursor.execute(
                "SELECT COUNT(*) FROM verifications WHERE batch_id = ? AND result IS NULL",
                (ver_retry_batch_id,),
            )
            if cursor.fetchone()[0] > 0:
                return f"process_verification_results:{ver_retry_batch_id}"

            # Check if retry batch has more timeouts
            cursor.execute(
                "SELECT COUNT(*) FROM verifications WHERE batch_id = ? AND result = 'TIMEOUT'",
                (ver_retry_batch_id,),
            )
            if cursor.fetchone()[0] > 0:
                return f"create_verification_retry_batch:{ver_retry_batch_id}"

        return "complete"


def main():
    dry_run = "--dry-run" in sys.argv

    state = detect_state()
    print(f"State: {state}")

    if dry_run:
        print("DRY RUN - no actions performed")
        return

    if state == "init_db":
        init_database()

    elif state == "scan_instances":
        scan_instances()

    elif state == "create_root_batch":
        create_root_batch()

    elif state.startswith("generate_root_script:"):
        batch_id = int(state.split(":")[1])
        generate_script(batch_id)

    elif state.startswith("submit_root_batch:"):
        batch_id = int(state.split(":")[1])
        submit_batch(batch_id)

    elif state.startswith("wait_root_batch:"):
        print("Waiting for SLURM jobs to complete")

    elif state.startswith("process_root_logs:"):
        batch_id = int(state.split(":")[1])
        process_root_logs(batch_id)

    elif state.startswith("create_root_retry_batch:"):
        batch_id = int(state.split(":")[1])
        create_root_retry_batch(batch_id)

    elif state == "create_verification_batch":
        create_verification_batch()

    elif state.startswith("generate_verification_script:"):
        batch_id = int(state.split(":")[1])
        generate_script(batch_id)

    elif state.startswith("submit_verification_batch:"):
        batch_id = int(state.split(":")[1])
        submit_batch(batch_id)

    elif state.startswith("wait_verification_batch:"):
        print("Waiting for SLURM jobs to complete")

    elif state.startswith("process_verification_results:"):
        batch_id = int(state.split(":")[1])
        process_verification_results(batch_id)

    elif state.startswith("create_verification_retry_batch:"):
        batch_id = int(state.split(":")[1])
        create_verification_retry_batch(batch_id)

    elif state == "complete":
        print("All batches complete")

    else:
        print(f"Unknown state: {state}")


if __name__ == "__main__":
    main()

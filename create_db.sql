-- experiments.db schema
-- SQLite database for CPBP algorithm evaluation experiments

PRAGMA foreign_keys = ON;

-- Instances table: problem instances to be solved
CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    best_bound INTEGER NOT NULL,
    optimization_type TEXT NOT NULL,
    highest_unsat INTEGER,
    lowest_sat INTEGER,
    UNIQUE (filename, problem_type)
);

-- Batches table: groups of resolutions executed together
CREATE TABLE batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    batch_type TEXT NOT NULL,
    script_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP,
    submission_attempts INTEGER DEFAULT 0,
    last_error TEXT,
    slurm_job_id TEXT,
    logs_processed BOOLEAN DEFAULT 0
);

-- Resolutions table: individual solver runs
CREATE TABLE resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    instance_id INTEGER NOT NULL,
    gap REAL NOT NULL,
    objective_bound INTEGER NOT NULL,
    partial_assignment TEXT,
    log_path TEXT NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    timeout BOOLEAN DEFAULT 0,
    result TEXT,
    FOREIGN KEY (batch_id) REFERENCES batches(id),
    FOREIGN KEY (instance_id) REFERENCES instances(id)
);

-- Subtrees table: partial assignment results across resolutions
CREATE TABLE subtrees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    partial_assignment TEXT NOT NULL,
    best_sat_bound INTEGER,
    worst_unsat_bound INTEGER,
    last_tested_at TIMESTAMP,
    FOREIGN KEY (instance_id) REFERENCES instances(id),
    UNIQUE (instance_id, partial_assignment)
);

CREATE INDEX idx_resolutions_batch ON resolutions(batch_id);
CREATE INDEX idx_resolutions_instance ON resolutions(instance_id);
CREATE INDEX idx_subtrees_instance ON subtrees(instance_id);

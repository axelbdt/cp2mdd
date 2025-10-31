
-- experiments.db schema
-- SQLite database for CPBP algorithm evaluation experiments

PRAGMA foreign_keys = ON;

-- Instances table: problem instances to be solved
CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    problem_type STRING NOT NULL,
    highest_unsat INTEGER,  -- highest objective found across all resolutions, computed during log processing
    lowest_sat INTEGER,   -- lowest objective found across all resolutions, computed during log processing
    UNIQUE (filename, problem_type)
);

-- Batches table: groups of resolutions executed together
CREATE TABLE batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    script_path TEXT,  -- path to generated .sh file
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP,  -- when successfully submitted to SLURM
    submission_attempts INTEGER DEFAULT 0,
    last_error TEXT,  -- last error message
    slurm_job_id TEXT,  -- SLURM job ID if submitted
    logs_processed BOOLEAN DEFAULT 0
);

-- TODO: Create table for partial assignment subtrees, did we test them ? Are they sat, unsat ?


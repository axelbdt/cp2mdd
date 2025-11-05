-- Instances: problem definitions
CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    optimal INTEGER NOT NULL,
    map_path TEXT,
    UNIQUE (filename, problem_type)
);

-- Configurations: solver parameters
CREATE TABLE configurations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    oracle_on_objective INTEGER NOT NULL,
)

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
    configuration_id INTEGER NOT NULL,
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

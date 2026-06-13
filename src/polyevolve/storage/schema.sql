-- v0 schema for superpod
-- Three logical layers: raw_fetches (audit) -> normalized -> decision

-- ============================================================
-- RAW LAYER (audit, append-only)
-- ============================================================

CREATE TABLE IF NOT EXISTS raw_fetches (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT        NOT NULL,
    endpoint    TEXT        NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload     JSONB       NOT NULL,
    checksum    TEXT        NOT NULL,
    UNIQUE (source, endpoint, checksum)
);

CREATE INDEX IF NOT EXISTS idx_raw_fetches_source_time
    ON raw_fetches (source, fetched_at DESC);

-- ============================================================
-- NORMALIZED LAYER
-- ============================================================

CREATE TABLE IF NOT EXISTS markets (
    id                BIGSERIAL PRIMARY KEY,
    venue             TEXT        NOT NULL,
    external_id       TEXT        NOT NULL,
    cross_venue_id    TEXT,
    question          TEXT        NOT NULL,
    description       TEXT,
    category          TEXT,
    close_time        TIMESTAMPTZ,
    status            TEXT        NOT NULL,
    raw_fetch_id      BIGINT      REFERENCES raw_fetches(id),
    first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata          JSONB,
    UNIQUE (venue, external_id)
);

CREATE INDEX IF NOT EXISTS idx_markets_venue_status ON markets (venue, status);
CREATE INDEX IF NOT EXISTS idx_markets_cross_venue
    ON markets (cross_venue_id) WHERE cross_venue_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_markets_category ON markets (category);

-- Signals: features extracted from raw_fetches that agents consume
CREATE TABLE IF NOT EXISTS signals (
    id            BIGSERIAL PRIMARY KEY,
    name          TEXT        NOT NULL,
    subject       TEXT        NOT NULL,
    value         JSONB       NOT NULL,
    raw_fetch_id  BIGINT      REFERENCES raw_fetches(id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_name_subject
    ON signals (name, subject, created_at DESC);

-- ============================================================
-- DECISION LAYER
-- ============================================================

CREATE TABLE IF NOT EXISTS predictions (
    id                            BIGSERIAL PRIMARY KEY,
    market_id                     BIGINT       NOT NULL REFERENCES markets(id),
    agent_name                    TEXT         NOT NULL,
    model_name                    TEXT         NOT NULL,
    probability_yes               NUMERIC(5,4) NOT NULL
        CHECK (probability_yes BETWEEN 0 AND 1),
    confidence                    NUMERIC(5,4) NOT NULL
        CHECK (confidence BETWEEN 0 AND 1),
    market_price_at_prediction    NUMERIC(5,4),
    reasoning                     TEXT         NOT NULL,
    key_factors                   JSONB,
    uncertainty_drivers           JSONB,
    data_sources_used             JSONB,
    created_at                    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_market
    ON predictions (market_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_agent
    ON predictions (agent_name, created_at DESC);

CREATE TABLE IF NOT EXISTS resolutions (
    market_id     BIGINT PRIMARY KEY REFERENCES markets(id),
    outcome       TEXT         NOT NULL,
    resolved_at   TIMESTAMPTZ  NOT NULL,
    raw_fetch_id  BIGINT       REFERENCES raw_fetches(id)
);

-- ============================================================
-- CALIBRATION VIEW
-- ============================================================
-- Decile-bucketed: for each agent/model, what fraction of predictions
-- in the 0.6-0.7 bucket actually resolved YES?
-- Brier component lets us track total predictive quality over time.

CREATE OR REPLACE VIEW calibration AS
SELECT
    p.agent_name,
    p.model_name,
    FLOOR(p.probability_yes * 10) / 10 AS probability_bucket,
    COUNT(*)                            AS n,
    AVG(p.probability_yes)              AS avg_predicted,
    AVG(CASE WHEN r.outcome = 'YES' THEN 1.0 ELSE 0.0 END) AS actual_yes_rate,
    AVG(POWER(p.probability_yes -
              CASE WHEN r.outcome = 'YES' THEN 1.0 ELSE 0.0 END, 2)) AS brier_component
FROM predictions p
JOIN resolutions r ON r.market_id = p.market_id
WHERE r.outcome IN ('YES', 'NO')
GROUP BY p.agent_name, p.model_name, FLOOR(p.probability_yes * 10) / 10
ORDER BY p.agent_name, p.model_name, probability_bucket;

-- ============================================================
-- LLM CALL TRACING (observability)
-- ============================================================
-- Every model call is recorded here: prompt, response, tokens, cache stats,
-- latency, estimated cost. This is the SQL/CLI substrate for LLM observability;
-- Langfuse (optional) layers on top when configured.

CREATE TABLE IF NOT EXISTS llm_calls (
    id                     BIGSERIAL PRIMARY KEY,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    agent_name             TEXT,
    model_name             TEXT         NOT NULL,
    market_external_id     TEXT,
    latency_ms             INTEGER,
    input_tokens           INTEGER,
    output_tokens          INTEGER,
    cache_read_tokens      INTEGER,
    cache_creation_tokens  INTEGER,
    estimated_cost_usd     NUMERIC(12,6),
    system_prompt_chars    INTEGER,
    user_prompt            TEXT,
    response               JSONB,
    error                  TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_calls_market ON llm_calls (market_external_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls (model_name, created_at DESC);

-- ============================================================
-- INSPECTION VIEWS (consumed by cli.py)
-- ============================================================

-- Recent predictions with market context.
CREATE OR REPLACE VIEW v_recent_predictions AS
SELECT
    p.id,
    p.created_at,
    p.agent_name,
    p.model_name,
    ROUND(p.probability_yes, 3)            AS p_yes,
    ROUND(p.confidence, 3)                 AS conf,
    ROUND(p.market_price_at_prediction, 3) AS mkt_price,
    m.venue,
    m.external_id                          AS market_id,
    LEFT(m.question, 70)                   AS question
FROM predictions p
JOIN markets m ON m.id = p.market_id
ORDER BY p.id DESC;

-- Per-run summary (grouped by calendar day + model).
CREATE OR REPLACE VIEW v_run_summary AS
SELECT
    DATE(p.created_at)               AS run_date,
    p.model_name,
    COUNT(*)                         AS predictions,
    COUNT(DISTINCT p.market_id)      AS markets,
    ROUND(AVG(p.confidence), 3)      AS avg_confidence,
    ROUND(AVG(p.probability_yes), 3) AS avg_p_yes
FROM predictions p
GROUP BY DATE(p.created_at), p.model_name
ORDER BY run_date DESC, p.model_name;

-- Cost + token usage from llm_calls, per day + model.
CREATE OR REPLACE VIEW v_cost AS
SELECT
    DATE(created_at)                  AS call_date,
    model_name,
    COUNT(*)                         AS calls,
    SUM(input_tokens)                AS input_tokens,
    SUM(output_tokens)               AS output_tokens,
    SUM(cache_read_tokens)           AS cache_read_tokens,
    ROUND(SUM(estimated_cost_usd), 4) AS cost_usd,
    ROUND(AVG(latency_ms))           AS avg_latency_ms,
    COUNT(*) FILTER (WHERE error IS NOT NULL) AS errors
FROM llm_calls
GROUP BY DATE(created_at), model_name
ORDER BY call_date DESC, model_name;

-- Market coverage: how many markets tracked, predicted, resolved.
CREATE OR REPLACE VIEW v_market_coverage AS
SELECT
    m.status,
    COUNT(DISTINCT m.id)                          AS markets,
    COUNT(DISTINCT p.market_id)                   AS with_prediction,
    COUNT(DISTINCT r.market_id)                   AS resolved
FROM markets m
LEFT JOIN predictions p ON p.market_id = m.id
LEFT JOIN resolutions r ON r.market_id = m.id
GROUP BY m.status
ORDER BY m.status;

-- Calibration including edge-over-market (per RESEARCH_CONTRACT objective).
CREATE OR REPLACE VIEW v_calibration_vs_market AS
SELECT
    p.agent_name,
    p.model_name,
    COUNT(*) AS n_resolved,
    ROUND(AVG(POWER(
        p.probability_yes - CASE WHEN r.outcome = 'YES' THEN 1.0 ELSE 0.0 END, 2
    )), 4) AS brier_agent,
    ROUND(AVG(POWER(
        p.market_price_at_prediction - CASE WHEN r.outcome = 'YES' THEN 1.0 ELSE 0.0 END, 2
    )), 4) AS brier_market,
    ROUND(AVG(POWER(
        p.market_price_at_prediction - CASE WHEN r.outcome = 'YES' THEN 1.0 ELSE 0.0 END, 2
    )) - AVG(POWER(
        p.probability_yes - CASE WHEN r.outcome = 'YES' THEN 1.0 ELSE 0.0 END, 2
    )), 4) AS edge
FROM predictions p
JOIN resolutions r ON r.market_id = p.market_id
WHERE r.outcome IN ('YES', 'NO')
  AND p.market_price_at_prediction IS NOT NULL
GROUP BY p.agent_name, p.model_name;

-- ============================================================
-- BACKTEST (point-in-time replay against resolved markets)
-- ============================================================
-- Separate from `predictions` because backtests are retrospective replays, not
-- forward predictions. Each row pairs an agent prediction (made with as_of-
-- windowed data) against the KNOWN outcome and the historical market price at
-- as_of. `is_clean` flags whether the market resolved after the model's training
-- cutoff - contaminated rows are for harness dev only and must NEVER be used as
-- a fitness/validation signal.

CREATE TABLE IF NOT EXISTS backtests (
    id                     BIGSERIAL PRIMARY KEY,
    run_id                 TEXT         NOT NULL,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    agent_name             TEXT         NOT NULL,
    model_name             TEXT         NOT NULL,
    market_external_id     TEXT         NOT NULL,
    question               TEXT,
    as_of                  TIMESTAMPTZ  NOT NULL,
    resolved_at            TIMESTAMPTZ,
    outcome                TEXT         NOT NULL,            -- YES | NO
    probability_yes        NUMERIC(5,4) NOT NULL,
    confidence             NUMERIC(5,4),
    market_price_at_as_of  NUMERIC(5,4),                    -- historical, may be NULL
    is_clean               BOOLEAN      NOT NULL,           -- post model-cutoff?
    split                  TEXT,                            -- train | holdout (clean rows only)
    brier_agent            NUMERIC(6,5),
    brier_market           NUMERIC(6,5),
    reasoning              TEXT,
    data_sources_used      JSONB,
    UNIQUE (run_id, market_external_id)
);

CREATE INDEX IF NOT EXISTS idx_backtests_run ON backtests (run_id);
CREATE INDEX IF NOT EXISTS idx_backtests_clean ON backtests (model_name, is_clean);

-- Backtest calibration partitioned by contamination AND split. The clean +
-- holdout cohort is the ONLY one whose numbers we trust for go/no-go.
CREATE OR REPLACE VIEW v_backtest_calibration AS
SELECT
    run_id,
    model_name,
    is_clean,
    COALESCE(split, 'n/a')                          AS split,
    COUNT(*)                                        AS n,
    ROUND(AVG(brier_agent), 4)                      AS brier_agent,
    ROUND(AVG(brier_market), 4)                     AS brier_market,
    ROUND(AVG(brier_market) - AVG(brier_agent), 4)  AS edge,
    COUNT(*) FILTER (WHERE market_price_at_as_of IS NULL) AS missing_price
FROM backtests
GROUP BY run_id, model_name, is_clean, COALESCE(split, 'n/a')
ORDER BY run_id, is_clean DESC, split;

-- ============================================================
-- EVAL SNAPSHOT (frozen, model-agnostic evaluation set)
-- ============================================================
-- The fast-iteration / evolution loop evaluates candidates against a FROZEN set
-- of resolved markets. Built once: each row carries the market, its point-in-
-- time as_of, the GDELT research context AND historical market price AS OF that
-- instant, and the known outcome. Crucially model-AGNOSTIC - clean/contaminated
-- and train/holdout are computed per-model at eval time (a 2024 market is clean
-- for qwen but contaminated for a 2025-cutoff model). This makes evaluation
-- network-free, reproducible, and identical across candidates.

CREATE TABLE IF NOT EXISTS eval_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    snapshot_set    TEXT         NOT NULL,           -- named set (e.g. "fp_v1")
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    market_external_id TEXT      NOT NULL,
    venue           TEXT         NOT NULL,
    question        TEXT         NOT NULL,
    event_title     TEXT,
    tags            JSONB,
    as_of           TIMESTAMPTZ  NOT NULL,           -- prediction instant
    resolved_at     TIMESTAMPTZ  NOT NULL,           -- actual resolution (closedTime)
    created_market_at TIMESTAMPTZ,                   -- market creation
    lead_days       INTEGER      NOT NULL,
    outcome         TEXT         NOT NULL,           -- YES | NO
    market_price_at_as_of NUMERIC(5,4),             -- historical, may be NULL
    research_context JSONB       NOT NULL,           -- {source: rendered_text} frozen at as_of
    UNIQUE (snapshot_set, market_external_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_snapshots_set ON eval_snapshots (snapshot_set);

-- Prediction cache: a candidate is (genome, model). Re-evaluating an unchanged
-- (snapshot row, model, genome) is a cache hit - no LLM call. This is what
-- makes the evolution loop affordable: only changed genomes pay inference.
CREATE TABLE IF NOT EXISTS prediction_cache (
    id              BIGSERIAL PRIMARY KEY,
    snapshot_id     BIGINT       NOT NULL REFERENCES eval_snapshots(id),
    model_name      TEXT         NOT NULL,
    genome_hash     TEXT         NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    probability_yes NUMERIC(5,4) NOT NULL,
    confidence      NUMERIC(5,4),
    reasoning       TEXT,
    UNIQUE (snapshot_id, model_name, genome_hash)
);

CREATE INDEX IF NOT EXISTS idx_prediction_cache_lookup
    ON prediction_cache (model_name, genome_hash);

-- Experiment ledger: one row per evolution run (or eval). Records what was
-- tried (snapshot, model, lead, split config), the champion selected by
-- VALIDATION, and its metrics on train/validation/the pristine TEST set, plus
-- the seed's test edge for comparison. Local + $0; evolve_task/EXPERIMENTS.md is
-- a human-readable render of this table, regenerated by scripts/experiment_log.py.
CREATE TABLE IF NOT EXISTS experiments (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    name            TEXT         NOT NULL,           -- run/results-dir label
    git_sha         TEXT,                            -- code version
    snapshot_set    TEXT         NOT NULL,
    model_name      TEXT,
    lead_days       INT,                             -- days before resolution priced
    generations     INT,
    holdout_frac    REAL,
    test_frac       REAL,                            -- 0 => two-way (no pristine test)
    champion_label  TEXT,                            -- e.g. gen_10
    champion_hash   TEXT,
    n_train         INT,
    n_val           INT,
    n_test          INT,
    edge_train      REAL,
    edge_val        REAL,                            -- selection set
    edge_test       REAL,                            -- pristine, trustworthy
    brier_test      REAL,
    seed_edge_test  REAL,                            -- gen_0 test edge, for transfer check
    notes           TEXT,
    config          JSONB                            -- catch-all
);

CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments (created_at DESC);

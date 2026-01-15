-- Knowledge Graph Schema Migration
-- Implements claim-centric knowledge graph per PRD_Knowledge_Graph.md
-- Tables prefixed with kg_ to avoid conflicts with existing schema

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- TABLE: kg_claims
-- The only first-class nodes in the knowledge graph
-- ============================================================================
CREATE TABLE kg_claims (
    id TEXT PRIMARY KEY,  -- Format: "claim_{hash}" where hash = first 12 chars of SHA256(text)
    
    -- Core content
    text TEXT NOT NULL,
    original_text TEXT,
    
    -- Classification
    type TEXT NOT NULL CHECK (type IN ('empirical', 'ground_truth', 'unsupported')),
    confidence FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Source paper metadata (NOT a separate node - embedded as JSONB)
    source_paper JSONB NOT NULL,
    -- Expected structure:
    -- {
    --   "arxiv_id": "2301.12345",
    --   "title": "Paper Title",
    --   "authors": ["Author 1", "Author 2"],
    --   "year": 2023,
    --   "venue": "Nature",
    --   "section": "Results",
    --   "url": "https://arxiv.org/abs/2301.12345"
    -- }
    
    -- External source (for non-arXiv papers)
    external_source JSONB,
    -- {
    --   "url": "https://...",
    --   "retrieval_method": "semantic_scholar" | "web_search" | "manual",
    --   "retrieval_date": "2025-01-15",
    --   "verified": true
    -- }
    
    -- Extraction metadata
    extraction JSONB NOT NULL,
    -- {
    --   "model": "gpt-4o-mini",
    --   "timestamp": "2025-01-15T10:30:00Z",
    --   "prompt_version": "1.0",
    --   "context_snippet": "..."
    -- }
    
    -- Vector embedding (1536 dimensions for text-embedding-3-small)
    embedding vector(1536),
    
    -- Computed fields (updated by triggers or application)
    support_count INTEGER DEFAULT 0,
    contradict_count INTEGER DEFAULT 0,
    depth_to_ground_truth INTEGER,  -- NULL if ungrounded
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- TABLE: kg_edges
-- Relationships between claims: supports or contradicts
-- ============================================================================
CREATE TABLE kg_edges (
    id TEXT PRIMARY KEY,  -- Format: "edge_{source_id}_{target_id}_{type}"
    source_id TEXT NOT NULL REFERENCES kg_claims(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES kg_claims(id) ON DELETE CASCADE,
    
    type TEXT NOT NULL CHECK (type IN ('supports', 'contradicts')),
    weight FLOAT NOT NULL DEFAULT 0.5 CHECK (weight >= 0 AND weight <= 1),
    
    -- Justification
    reasoning TEXT,
    
    -- Metadata
    model TEXT,  -- Which LLM created this edge
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Prevent duplicate edges (same source, target, type)
    UNIQUE(source_id, target_id, type)
);

-- ============================================================================
-- TABLE: kg_build_runs
-- Track pipeline executions for reproducibility and debugging
-- ============================================================================
CREATE TABLE kg_build_runs (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status TEXT CHECK (status IN ('running', 'completed', 'failed')),
    
    -- Configuration
    anchor_papers JSONB,  -- List of arXiv IDs used as seeds
    config JSONB,  -- { max_depth, max_claims, parallel_workers, ... }
    
    -- Progress tracking
    claims_extracted INTEGER DEFAULT 0,
    edges_created INTEGER DEFAULT 0,
    papers_processed INTEGER DEFAULT 0,
    current_depth INTEGER DEFAULT 0,
    
    -- Error tracking
    errors JSONB  -- Array of error objects
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Claims indexes
CREATE INDEX kg_claims_type_idx ON kg_claims(type);
CREATE INDEX kg_claims_confidence_idx ON kg_claims(confidence);
CREATE INDEX kg_claims_source_paper_idx ON kg_claims USING GIN (source_paper);
CREATE INDEX kg_claims_created_at_idx ON kg_claims(created_at);

-- Full-text search on claim text
CREATE INDEX kg_claims_text_search_idx ON kg_claims USING GIN (to_tsvector('english', text));

-- Edges indexes
CREATE INDEX kg_edges_source_idx ON kg_edges(source_id);
CREATE INDEX kg_edges_target_idx ON kg_edges(target_id);
CREATE INDEX kg_edges_type_idx ON kg_edges(type);

-- pgvector index for similarity search (IVFFlat for speed)
-- Note: lists=100 is good for up to ~100k vectors; adjust if graph grows larger
CREATE INDEX kg_claims_embedding_idx ON kg_claims 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Semantic search for claims by embedding
CREATE OR REPLACE FUNCTION search_claims_by_embedding(
    query_embedding vector(1536),
    match_count INT DEFAULT 10,
    filter_type TEXT DEFAULT NULL,
    min_confidence FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    id TEXT,
    text TEXT,
    type TEXT,
    confidence FLOAT,
    source_paper JSONB,
    support_count INTEGER,
    contradict_count INTEGER,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.text,
        c.type,
        c.confidence,
        c.source_paper,
        c.support_count,
        c.contradict_count,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM kg_claims c
    WHERE 
        (filter_type IS NULL OR c.type = filter_type)
        AND c.confidence >= min_confidence
        AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function: Get claim with computed support/contradict counts
CREATE OR REPLACE FUNCTION get_claim_with_counts(claim_id_param TEXT)
RETURNS TABLE (
    id TEXT,
    text TEXT,
    original_text TEXT,
    type TEXT,
    confidence FLOAT,
    source_paper JSONB,
    external_source JSONB,
    extraction JSONB,
    depth_to_ground_truth INTEGER,
    support_count BIGINT,
    contradict_count BIGINT,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.text,
        c.original_text,
        c.type,
        c.confidence,
        c.source_paper,
        c.external_source,
        c.extraction,
        c.depth_to_ground_truth,
        (SELECT COUNT(*) FROM kg_edges e WHERE e.target_id = c.id AND e.type = 'supports'),
        (SELECT COUNT(*) FROM kg_edges e WHERE e.target_id = c.id AND e.type = 'contradicts'),
        c.created_at
    FROM kg_claims c
    WHERE c.id = claim_id_param;
END;
$$;

-- Function: Full-text keyword search for claims
CREATE OR REPLACE FUNCTION search_claims_by_keyword(
    search_query TEXT,
    match_count INT DEFAULT 10,
    filter_type TEXT DEFAULT NULL,
    min_confidence FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    id TEXT,
    text TEXT,
    type TEXT,
    confidence FLOAT,
    source_paper JSONB,
    support_count INTEGER,
    contradict_count INTEGER,
    rank FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.text,
        c.type,
        c.confidence,
        c.source_paper,
        c.support_count,
        c.contradict_count,
        ts_rank(to_tsvector('english', c.text), plainto_tsquery('english', search_query)) AS rank
    FROM kg_claims c
    WHERE 
        to_tsvector('english', c.text) @@ plainto_tsquery('english', search_query)
        AND (filter_type IS NULL OR c.type = filter_type)
        AND c.confidence >= min_confidence
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- Function: Get incoming edges (claims that support/contradict this claim)
CREATE OR REPLACE FUNCTION get_incoming_edges(claim_id_param TEXT)
RETURNS TABLE (
    edge_id TEXT,
    source_claim_id TEXT,
    source_claim_text TEXT,
    source_claim_type TEXT,
    edge_type TEXT,
    edge_weight FLOAT,
    reasoning TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id AS edge_id,
        e.source_id AS source_claim_id,
        c.text AS source_claim_text,
        c.type AS source_claim_type,
        e.type AS edge_type,
        e.weight AS edge_weight,
        e.reasoning
    FROM kg_edges e
    JOIN kg_claims c ON c.id = e.source_id
    WHERE e.target_id = claim_id_param;
END;
$$;

-- Function: Get outgoing edges (claims this claim supports/contradicts)
CREATE OR REPLACE FUNCTION get_outgoing_edges(claim_id_param TEXT)
RETURNS TABLE (
    edge_id TEXT,
    target_claim_id TEXT,
    target_claim_text TEXT,
    target_claim_type TEXT,
    edge_type TEXT,
    edge_weight FLOAT,
    reasoning TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id AS edge_id,
        e.target_id AS target_claim_id,
        c.text AS target_claim_text,
        c.type AS target_claim_type,
        e.type AS edge_type,
        e.weight AS edge_weight,
        e.reasoning
    FROM kg_edges e
    JOIN kg_claims c ON c.id = e.target_id
    WHERE e.source_id = claim_id_param;
END;
$$;

-- Function: Get graph statistics
CREATE OR REPLACE FUNCTION get_kg_statistics()
RETURNS TABLE (
    total_claims BIGINT,
    empirical_claims BIGINT,
    ground_truth_claims BIGINT,
    unsupported_claims BIGINT,
    total_edges BIGINT,
    support_edges BIGINT,
    contradict_edges BIGINT,
    grounded_claims BIGINT,
    ungrounded_claims BIGINT,
    avg_confidence FLOAT,
    papers_indexed BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM kg_claims),
        (SELECT COUNT(*) FROM kg_claims WHERE type = 'empirical'),
        (SELECT COUNT(*) FROM kg_claims WHERE type = 'ground_truth'),
        (SELECT COUNT(*) FROM kg_claims WHERE type = 'unsupported'),
        (SELECT COUNT(*) FROM kg_edges),
        (SELECT COUNT(*) FROM kg_edges WHERE type = 'supports'),
        (SELECT COUNT(*) FROM kg_edges WHERE type = 'contradicts'),
        (SELECT COUNT(*) FROM kg_claims WHERE depth_to_ground_truth IS NOT NULL),
        (SELECT COUNT(*) FROM kg_claims WHERE depth_to_ground_truth IS NULL AND type != 'ground_truth'),
        (SELECT AVG(confidence) FROM kg_claims),
        (SELECT COUNT(DISTINCT source_paper->>'arxiv_id') FROM kg_claims WHERE source_paper->>'arxiv_id' IS NOT NULL);
END;
$$;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger: Update support/contradict counts on edge insert/delete
CREATE OR REPLACE FUNCTION update_claim_edge_counts()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.type = 'supports' THEN
            UPDATE kg_claims SET support_count = support_count + 1 WHERE id = NEW.target_id;
        ELSIF NEW.type = 'contradicts' THEN
            UPDATE kg_claims SET contradict_count = contradict_count + 1 WHERE id = NEW.target_id;
        END IF;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        IF OLD.type = 'supports' THEN
            UPDATE kg_claims SET support_count = GREATEST(0, support_count - 1) WHERE id = OLD.target_id;
        ELSIF OLD.type = 'contradicts' THEN
            UPDATE kg_claims SET contradict_count = GREATEST(0, contradict_count - 1) WHERE id = OLD.target_id;
        END IF;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_claim_counts
    AFTER INSERT OR DELETE ON kg_edges
    FOR EACH ROW
    EXECUTE FUNCTION update_claim_edge_counts();

-- Trigger: Update updated_at timestamp on claim modification
CREATE OR REPLACE FUNCTION update_claim_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_claim_timestamp
    BEFORE UPDATE ON kg_claims
    FOR EACH ROW
    EXECUTE FUNCTION update_claim_timestamp();

-- ============================================================================
-- ROW LEVEL SECURITY (Optional - disabled for now)
-- ============================================================================

-- Enable RLS on tables
ALTER TABLE kg_claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_build_runs ENABLE ROW LEVEL SECURITY;

-- Allow all access for now (can restrict later)
CREATE POLICY "Allow all access to kg_claims" ON kg_claims FOR ALL USING (true);
CREATE POLICY "Allow all access to kg_edges" ON kg_edges FOR ALL USING (true);
CREATE POLICY "Allow all access to kg_build_runs" ON kg_build_runs FOR ALL USING (true);

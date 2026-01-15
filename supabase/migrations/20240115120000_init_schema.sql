-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Table: papers
create table papers (
  id uuid primary key default gen_random_uuid(),
  openalex_id text unique,
  doi text unique,
  title text not null,
  abstract text,
  authors jsonb default '[]'::jsonb,
  year int,
  trust_score float default 1.0,
  is_retracted boolean default false,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Table: claims
create table claims (
  id uuid primary key default gen_random_uuid(),
  paper_id uuid references papers(id) on delete cascade not null,
  text text not null,
  claim_type text,
  evidence_type text,
  confidence float default 0.8,
  validation_status text default 'pending',
  embedding vector(1536), -- Assuming OpenAI embeddings
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Table: citations (Evidence/Validation results)
create table citations (
  id uuid primary key default gen_random_uuid(),
  claim_id uuid references claims(id) on delete cascade not null,
  citation_doi text,
  is_relevant boolean,
  is_circular boolean default false,
  relevance_score float,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Table: graph_edges
create table graph_edges (
  id uuid primary key default gen_random_uuid(),
  source text not null,
  target text not null,
  type text not null,
  weight float default 1.0,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Table: conversations
create table conversations (
  id uuid primary key default gen_random_uuid(),
  title text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Table: messages
create table messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid references conversations(id) on delete cascade not null,
  role text not null, -- 'user' or 'assistant'
  content jsonb not null, -- Stores the actual message content, can be text or structured
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create indexes for better performance
create index idx_papers_doi on papers(doi);
create index idx_claims_paper_id on claims(paper_id);
create index idx_messages_conversation_id on messages(conversation_id);

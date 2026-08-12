# BANT Evidence Extraction: Epics and User Stories

This is a discussion draft based only on:

- `ARCHITECTURE.md`
- `HIERARCHICAL-CHUNKING.md`
- `HARNESS.md`

The scope is backend evidence extraction. The system does not decide whether a lead qualifies.

## Epic 1: Strategic Hierarchical Chunking

**Goal:** Prepare long transcripts as structured, bounded, overlapping context while preserving the original evidence.

### Epic 1 Data Schema: TranscriptSegment

The qualification subsystem normalizes each existing transcript entry into an
immutable source unit. The existing `output/transcript.json` is not modified.

```json
{
  "segment_id": "seg_000001",
  "speaker": "SPEAKER_00",
  "start": 12.4,
  "end": 15.8,
  "text": "We need Finance approval.",
  "words": [
    {
      "word": "We",
      "start": 12.4,
      "end": 12.7
    }
  ]
}
```

**Schema rules:**

- `segment_id` is deterministic and unique within the transcript.
- One segment represents one original speaker utterance.
- `speaker`, `start`, `end`, and `text` are copied exactly from the source transcript.
- `words` is preserved when available and may be empty or unavailable.
- The qualification subsystem does not summarize, rewrite, or merge source segment text.
- A segment is not split unless it exceeds the configured chunk size limit.
- If an oversized segment is split, each piece retains the original `segment_id` and records its word range.
- Conversation context is created by grouping segments into sections and chunks; speakers are not merged into one source segment.

### Epic 1 Data Schema: ConversationSection

A conversation section is a chronological parent container. It groups multiple
source segments, usually from both speakers, and provides broad context for
child extraction chunks.

```json
{
  "section_id": "section_000001",
  "sequence": 1,
  "start": 0.0,
  "end": 480.0,
  "segment_ids": [
    "seg_000001",
    "seg_000002",
    "seg_000003"
  ],
  "overlap_segment_ids": [],
  "chunk_ids": []
}
```

**Schema rules:**

- `section_id` is deterministic and unique within the transcript.
- `sequence` defines the chronological order of sections.
- `start` is the earliest source segment start time in the section.
- `end` is the latest source segment end time in the section.
- `segment_ids` are ordered chronologically and reference original source segments.
- A section may contain segments from multiple speakers.
- `overlap_segment_ids` identifies segments intentionally shared with another section.
- `chunk_ids` references the child extraction chunks belonging to the section.
- Sections are organizational only and do not have a BANT label.
- Sections do not contain rewritten or summarized transcript text.
- Every source segment belongs to at least one section.

The section is not the primary LLM input. The child `ExtractionChunk` contains
the rendered conversation text, with speaker labels and timestamps, that is
sent to the topic agents. Section and segment IDs are included alongside that
text for grounding and traceability. The chunker processes the complete
hierarchy; it does not use vector search to select only the most relevant
sections.

### Epic 1 Data Schema: ExtractionChunk

An extraction chunk is the bounded child context sent to a BANT topic agent.
It contains rendered conversation text plus references to the original source
segments.

```json
{
  "chunk_id": "chunk_000001",
  "section_id": "section_000001",
  "sequence": 1,
  "start": 12.4,
  "end": 20.1,
  "segment_ids": [
    "seg_000001",
    "seg_000002"
  ],
  "overlap_segment_ids": [],
  "token_count": 24,
  "text": "REP [00:00:12]: Who approves this internally?\nPROSPECT [00:00:17]: Our VP of Finance."
}
```

**Schema rules:**

- `chunk_id` is deterministic and unique within the transcript.
- `section_id` identifies the parent conversation section.
- `sequence` defines the chunk order within its parent section.
- `start` is the first source timestamp represented in the chunk.
- `end` is the last source timestamp represented in the chunk.
- `segment_ids` are ordered references to the original source segments.
- `overlap_segment_ids` identifies segments intentionally shared with neighboring chunks.
- `token_count` records the size of the rendered chunk text using the configured tokenizer.
- `text` is rendered from the original segments and is sent to the LLM.
- Rendered text includes speaker labels and timestamps.
- The chunk does not replace the original transcript.
- A chunk normally contains multiple speakers when the conversation includes both speakers.
- Chunks remain within the configured maximum token limit, except when handling an individual oversized source segment according to the defined word-range rule.

### Epic 1 Data Schema: CoverageRecord

A coverage record proves that the complete transcript is represented in the
hierarchy and makes intentional overlap visible before processing begins.

```json
{
  "coverage_id": "coverage_000001",
  "source_segment_count": 4,
  "section_count": 1,
  "chunk_count": 2,
  "covered_segment_ids": [
    "seg_000001",
    "seg_000002",
    "seg_000003",
    "seg_000004"
  ],
  "omitted_segment_ids": [],
  "shared_segment_ids": [
    "seg_000003"
  ],
  "valid": true,
  "errors": []
}
```

**Schema rules:**

- `coverage_id` is deterministic for the chunking run.
- `source_segment_count` equals the number of normalized transcript segments.
- `section_count` records the number of generated parent sections.
- `chunk_count` records the number of generated child chunks.
- `covered_segment_ids` contains every segment included in at least one extraction chunk.
- `omitted_segment_ids` contains source segments not included in any extraction chunk.
- `shared_segment_ids` contains segments intentionally repeated because of overlap.
- `valid` is `true` only when no source segment is unexpectedly omitted.
- `errors` records structural problems such as missing references or invalid ordering.
- An overlapping segment is not considered omitted or incorrectly duplicated.
- Coverage validation happens before chunks are sent to the BANT topic agents.

### Epic 1 Data Schema: ChunkingConfiguration

The chunking configuration makes hierarchy creation explicit and reproducible.
It uses approximate time windows for broad sections, a hard token limit for
LLM chunks, and complete speaker turns for conversational overlap.

```json
{
  "section_target_seconds": 480,
  "section_overlap_seconds": 30,
  "max_chunk_tokens": 1200,
  "chunk_overlap_turns": 2,
  "max_overlap_tokens": 250,
  "tokenizer": "configured-tokenizer-v1"
}
```

**Configuration rules:**

- `section_target_seconds` is the approximate target duration for a parent section.
- `section_overlap_seconds` controls the intentional time overlap between neighboring sections.
- `max_chunk_tokens` is the hard target boundary for rendered LLM input.
- `chunk_overlap_turns` is the preferred number of complete speaker turns shared between neighboring chunks.
- `max_overlap_tokens` limits the preferred size of shared chunk context.
- Chunk overlap is selected by complete source segments and speaker turns, not by cutting text at a timestamp or token boundary.
- If one indivisible speaker turn exceeds `max_overlap_tokens`, it remains intact and the resulting exception is recorded in chunk metadata or validation errors.
- `tokenizer` identifies the tokenizer used to calculate `token_count` and enforce chunk limits.
- The configuration is stored with the extraction run so the hierarchy can be reproduced.
- Time is used for section organization; token count is used for LLM safety; speaker turns are used for chunk overlap.

### US-1.1: Preserve Transcript Source Units

**Status: Done**

Implemented by `pipeline/qualification/schemas.py` and covered by unit and
integration tests in `tests/qualification/`. The normalizer is non-destructive,
assigns deterministic source IDs, preserves exact source fields and word
metadata, and records word ranges for oversized pieces.

**As the extraction harness,** I want every transcript segment to have a stable source identity, **so that** extracted evidence can always reference the original call.

**Acceptance criteria:**

- Every segment has a stable `segment_id`.
- Segment IDs are deterministic and unique within the transcript.
- Speaker, start time, end time, and exact text are preserved.
- Original transcript text is not summarized or rewritten.
- Word-level timestamps remain available for oversized segments.
- The qualification subsystem normalizes source segments without modifying `output/transcript.json`.
- Oversized segment pieces retain the original `segment_id` and their word-range reference.

### US-1.2: Create Conversation Sections

**Status: Done**

Implemented by `pipeline/qualification/chunking.py` (`create_sections`) and the
`ChunkingConfiguration` / `ConversationSection` schemas in
`pipeline/qualification/schemas.py`, covered by unit and integration tests in
`tests/qualification/`. Section windows are deterministic, overlap by exactly
the configured `section_overlap_seconds`, and every source segment belongs to
at least one section.

**As the extraction harness,** I want the transcript grouped into chronological conversation sections, **so that** long calls can be processed with broader context.

**Acceptance criteria:**

- Sections preserve chronological order.
- Every transcript segment belongs to at least one section.
- Sections may overlap where needed to preserve conversational context.
- Sections are organizational and are not qualification labels.
- Section IDs and sequence numbers are deterministic.
- Each section records its start and end timestamps, ordered source segment IDs, overlap segment IDs, and child chunk IDs.
- Section overlap is created according to the configured section overlap value.
- A section may contain source segments from multiple speakers.

### US-1.3: Create Bounded Extraction Chunks

**As a BANT topic agent,** I want bounded chunks made from complete transcript segments and speaker turns, **so that** I can process manageable context without losing meaning.

**Acceptance criteria:**

- Chunks do not arbitrarily cut normal transcript segments.
- Related speaker turns are kept together when possible.
- Every chunk has a maximum size.
- Neighboring chunks include overlap.
- Every chunk retains its section ID and source segment IDs.
- Every chunk has start and end timestamps.
- Chunk text is rendered from the original source segments with speaker labels and timestamps.
- Each chunk records its deterministic ID, sequence, ordered segment IDs, overlap segment IDs, rendered text, and `token_count`.
- The configured maximum token limit is enforced for rendered chunk text.
- Chunk overlap uses complete speaker turns and respects the configured overlap target where possible.
- Normal transcript segments are never cut to satisfy an overlap limit.

### US-1.4: Support Strategic Chunking Infrastructure

**As the extraction harness,** I want parent sections and child extraction chunks, **so that** the transcript hierarchy remains traceable from broad context to exact evidence.

**Acceptance criteria:**

- Parent sections contain child extraction chunks.
- Child chunks retain original transcript segment references.
- Parent-child relationships are available to downstream processing.
- Framework behavior cannot silently discard transcript metadata.
- The hierarchy preserves speakers, timestamps, and exact text.
- Every chunk can be resolved from its chunk ID to its parent section and original source segments.
- Rendered chunk text can be reconstructed from the referenced source segments.
- The hierarchy does not require vector search to select which transcript chunks are processed.

### US-1.5: Cover the Complete Transcript

**As the extraction harness,** I want every transcript section and chunk processed, **so that** relevant BANT evidence is not missed because it was outside a selected window.

**Acceptance criteria:**

- The chunk structure covers the complete transcript.
- No transcript segment is silently omitted.
- Overlapping segments are intentionally marked as shared context.
- Chunk creation produces a coverage record.
- The coverage record includes source, section, and chunk counts.
- The coverage record lists covered, omitted, and intentionally shared segment IDs.
- Coverage validation runs before any BANT topic agent receives chunks.
- Unexpected omissions, missing references, and invalid ordering make the coverage record invalid and are recorded as errors.

### US-1.6: Reproduce Chunking Configuration

**As the extraction harness,** I want the chunking configuration stored with the run, **so that** the transcript hierarchy can be reproduced and audited.

**Acceptance criteria:**

- The configuration records section target duration and section overlap duration.
- The configuration records the maximum chunk token limit.
- The configuration records the preferred number of overlapping speaker turns and maximum overlap tokens.
- The tokenizer identity used to calculate `token_count` is recorded.
- The same transcript and configuration produce the same IDs, section membership, chunk membership, rendered text, and coverage record.
- Configuration values are applied consistently to every section and chunk in the run.

### Epic 1 Data Schema: EvaluationQuestion

An evaluation question is a manually reviewed question-answer case used to
compare chunking configurations against known transcript evidence.

```yaml
id: q_authority_001
topic: Authority
question: "Who needs to approve the purchase?"
expected_answer: "The VP of Finance."
required_segment_ids:
  - seg_000041
  - seg_000042
acceptable_evidence:
  - "VP of Finance"
```

**Schema rules:**

- `id` is unique within the evaluation dataset.
- `topic` is one of Budget, Authority, Need, or Timeline.
- `question` is answerable from the transcript.
- `expected_answer` describes the evidence-based answer without applying a qualification decision.
- `required_segment_ids` identifies the source segments needed to answer the question correctly.
- `acceptable_evidence` contains important words or phrases expected in a grounded answer.
- Evaluation questions are manually reviewed before being used as quality benchmarks.

### US-1.7: Evaluate Chunking Strategy

**As the extraction harness,** I want candidate chunking configurations evaluated against known transcript questions and evidence, **so that** the selected strategy is supported by measured results rather than assumptions.

**Acceptance criteria:**

- The evaluation dataset contains manually reviewed questions, expected answers, topics, and required source segment IDs.
- The same evaluation questions are run against every candidate chunking configuration.
- The evaluation verifies that every required source segment appears in at least one chunk.
- The evaluation measures whether required question-and-answer segments appear together in at least one chunk.
- The evaluation measures whether the topic agent extracts the expected evidence.
- The evaluation verifies returned quotes, speakers, timestamps, and source IDs against the original transcript.
- The evaluation records unsupported or hallucinated evidence separately from missing evidence.
- Results include evidence coverage, context sufficiency, extraction recall, grounding precision, chunk count, total tokens, and overlap duplication.
- Results are saved in a machine-readable format for comparison across configurations.
- A chunking configuration is not selected solely because it produces a plausible answer from one question.

## Epic 2: Extract BANT Evidence

**Goal:** Run four parallel topic agents that extract all relevant transcript information without qualification decisions.

### US-2.1: Extract Budget Evidence

**As a company reviewing a sales call,** I want all transcript evidence related to Budget, **so that** I can apply my own budget KPIs.

**Acceptance criteria:**

- The Budget agent processes the complete transcript hierarchy.
- It extracts relevant statements about funds, budget, cost, limits, and financial constraints.
- Each item includes the exact quote, speaker, timestamp, and source IDs.
- The agent does not decide whether the budget is sufficient.

### US-2.2: Extract Authority Evidence

**As a company reviewing a sales call,** I want all transcript evidence related to Authority, **so that** I can apply my own decision-process KPIs.

**Acceptance criteria:**

- The Authority agent processes the complete transcript hierarchy.
- It extracts statements about decision-makers, approval, sign-off, Finance, Procurement, and stakeholders.
- Each item includes the exact quote, speaker, timestamp, and source IDs.
- The agent does not decide whether authority is confirmed.

### US-2.3: Extract Need Evidence

**As a company reviewing a sales call,** I want all transcript evidence related to Need, **so that** I can apply my own problem and outcome KPIs.

**Acceptance criteria:**

- The Need agent processes the complete transcript hierarchy.
- It extracts statements about problems, pain, desired outcomes, consequences, and reasons for seeking a solution.
- Each item includes the exact quote, speaker, timestamp, and source IDs.
- The agent does not decide whether the need is strong enough.

### US-2.4: Extract Timeline Evidence

**As a company reviewing a sales call,** I want all transcript evidence related to Timeline, **so that** I can apply my own timing KPIs.

**Acceptance criteria:**

- The Timeline agent processes the complete transcript hierarchy.
- It extracts statements about deadlines, implementation dates, buying windows, and procurement timing.
- Each item includes the exact quote, speaker, timestamp, and source IDs.
- The agent does not decide whether the timeline is acceptable.

### US-2.5: Run Topic Agents in Parallel

**As the extraction harness,** I want the four BANT agents to process their topics independently and in parallel, **so that** each topic receives focused extraction instructions without waiting for unrelated topics.

**Acceptance criteria:**

- Budget, Authority, Need, and Timeline have separate agent responsibilities.
- Each agent scans the complete transcript hierarchy.
- One agent's result does not replace another agent's result.
- A failure in one topic is recorded separately from successful topic results.

## Epic 3: Validate and Assemble Evidence

**Goal:** Ensure extracted information is grounded in the original transcript and combine results without losing evidence.

### US-3.1: Validate Evidence Against the Source

**As the extraction harness,** I want every extracted item checked against the original transcript, **so that** the company receives grounded information rather than invented or altered quotes.

**Acceptance criteria:**

- The referenced segment exists.
- The quote exists in the original transcript.
- The speaker matches the source segment.
- The timestamp is valid.
- The BANT topic is one of Budget, Authority, Need, or Timeline.

### US-3.2: Merge Overlapping Results

**As the evidence assembler,** I want duplicate results from overlapping chunks combined, **so that** the final artifact does not repeat the same evidence unnecessarily.

**Acceptance criteria:**

- Duplicate references are identified using source segment and timestamp information.
- Separate statements from different moments are preserved.
- The original transcript remains authoritative when overlapping results differ.
- Evidence is sorted chronologically within each BANT topic.

### US-3.3: Preserve Cross-Topic Evidence

**As the evidence assembler,** I want one transcript statement to appear under multiple BANT topics when relevant, **so that** no information is lost by forcing it into a single category.

**Acceptance criteria:**

- Evidence is not globally deduplicated across different BANT topics.
- The same source segment may appear under multiple topics.
- Each topic retains its own evidence association.

### US-3.4: Record Absence Separately From Failure

**As a company reviewing a call,** I want no-evidence results distinguished from processing failures, **so that** I know whether a topic was absent or simply not processed successfully.

**Acceptance criteria:**

- “No evidence found” is a valid result.
- Failed chunks or agents are recorded as processing failures.
- Failed work can be retried without rerunning successful work.
- Unsupported evidence is excluded from the final artifact and retained in audit information.

## Epic 4: Link Existing Call Signals

**Goal:** Attach already-produced measurements to each validated BANT evidence moment without rerunning signal analysis.

### US-4.1: Link Audio Measurements

**As a company reviewing evidence,** I want the existing audio measurements associated with each transcript moment, **so that** I receive the relevant call data alongside the quote.

**Acceptance criteria:**

- Audio measurements are linked using the evidence segment or timestamp.
- Original audio values are preserved.
- No audio model is executed again.

### US-4.2: Link Voice Measurements

**As a company reviewing evidence,** I want the existing voice measurements associated with each transcript moment, **so that** I receive the relevant voice data alongside the quote.

**Acceptance criteria:**

- Voice measurements are linked using the evidence segment or timestamp.
- Original valence, arousal, and dominance values are preserved.
- No voice model is executed again.

### US-4.3: Link Facial Measurements

**As a company reviewing evidence,** I want the nearest existing facial measurement associated with each transcript moment when available, **so that** facial evidence is provided without rerunning facial analysis.

**Acceptance criteria:**

- The facial sample timestamp is preserved.
- The nearest available sample is linked according to the defined temporal rule.
- Missing facial measurements remain explicitly unavailable.
- Missing data is not replaced with zero.

### US-4.4: Preserve Measurement Boundaries

**As a company reviewing evidence,** I want transcript evidence and measurements kept as separate data, **so that** I can interpret them according to my own rules.

**Acceptance criteria:**

- Measurements are attached as raw associated data.
- The linker does not interpret, score, or compare measurements.
- No field states that a signal supports or contradicts the transcript.

## Epic 5: Orchestrate and Deliver the Evidence Artifact

**Goal:** Execute the complete extraction workflow reliably and produce a structured backend artifact for the company's own qualification process.

### US-5.1: Initialize the Extraction Run

**As the harness,** I want to load the existing call outputs and initialize run state, **so that** the workflow can process one call consistently.

**Acceptance criteria:**

- The required transcript and signal inputs are checked before processing.
- The run has a call identifier.
- The run state records input availability and processing status.
- Missing required inputs fail clearly without rerunning the original pipeline.

### US-5.2: Retry Failed Work Locally

**As the harness,** I want failed chunk or agent work retried locally, **so that** one temporary failure does not restart the complete call.

**Acceptance criteria:**

- Failed work identifies its topic, chunk, and attempt.
- Successful work is not repeated unnecessarily.
- Retry exhaustion is recorded clearly.
- A failed retry does not create qualification conclusions.

### US-5.3: Produce the BANT Evidence Artifact

**As a company consuming call data,** I want one structured artifact containing all validated BANT evidence and linked measurements, **so that** I can apply my own qualification process.

**Acceptance criteria:**

- The artifact contains Budget, Authority, Need, and Timeline collections.
- Each evidence item includes quote, speaker, timestamps, and source references.
- Linked audio, voice, and facial measurements are included when available.
- Missing measurements are explicitly represented.
- The artifact contains no qualification verdict.

### US-5.4: Preserve Traceability

**As a company auditing extracted information,** I want every output item traceable to its source chunk and transcript segment, **so that** I can verify the information against the original call.

**Acceptance criteria:**

- Every evidence item retains source segment ID.
- Every evidence item retains source chunk ID.
- The artifact preserves timestamps and speaker identity.
- Processing failures and rejected unsupported candidates remain available in audit information.

## Out of Scope

These items are not part of this scope:

- Deciding whether a lead qualifies
- Applying company-specific KPIs
- Producing qualification scores or statuses
- CRM field export
- Frontend or report changes
- Re-running audio, voice, or facial analysis
- Building embeddings or a vector database
- Cross-call or account-level analysis

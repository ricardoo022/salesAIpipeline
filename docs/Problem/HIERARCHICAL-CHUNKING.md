# Hierarchical Transcript Chunking

## Purpose

The BANT extractor must process long sales-call transcripts without losing conversational context, exact quotes, speakers, or timestamps.

The solution is hierarchical chunking. The transcript is organized into parent conversation sections and smaller child extraction chunks. The original transcript segments remain the source of truth throughout the process.

## What Gets Chunked

Only the transcript is chunked.

The extraction operation reads the transcript to locate information related to:

- Budget
- Authority
- Need
- Timeline

The other pipeline outputs are not independently chunked because they are already measurements associated with the call's timeline:

- Audio measurements
- Voice measurements
- Facial measurements

Those measurements are linked to extracted transcript evidence afterward. They do not need to be sent through the extraction chunker or analyzed again.

## Hierarchy

```text
Full transcript
    |
    v
Conversation sections
    |
    v
Extraction chunks
    |
    v
Original transcript segments
```

### Level 0: Original Transcript Segments

The original transcript segments are the atomic source units. Each segment preserves:

- A stable segment ID
- Speaker
- Start timestamp
- End timestamp
- Exact transcript text

Segments should not be cut arbitrarily. If an individual segment is larger than the maximum allowed chunk size, its word-level timestamps can be used to split it while preserving the original segment reference.

### Level 1: Conversation Sections

Nearby transcript segments are grouped into larger chronological sections.

```text
Section 1: 00:00-08:00
Section 2: 07:30-16:00
Section 3: 15:30-24:00
```

Sections provide broad context for long calls. They do not need to be labelled as “Budget,” “Authority,” “Need,” or “Timeline,” because those labels could introduce an interpretation before extraction. They are organizational and chronological.

Sections may overlap slightly so that a conversation exchange is not separated completely between two parents.

### Level 2: Extraction Chunks

Each section is divided into smaller chunks suitable for extraction.

```text
Section 1
    Chunk 1A: segments 001-018
    Chunk 1B: segments 015-032

Section 2
    Chunk 2A: segments 033-050
    Chunk 2B: segments 047-064
```

Child chunks overlap by several transcript segments or speaker turns. The overlap protects context when a question and answer fall near a boundary.

## Chunking Rules

The chunker follows these rules:

1. Start with complete transcript segments.
2. Keep related speaker turns together when possible.
3. Prefer natural conversation boundaries.
4. Enforce a maximum chunk size.
5. Add overlap between neighboring chunks.
6. Preserve every segment ID and timestamp.
7. Never replace the original transcript with a summary.

The maximum size is a safety boundary. Even if the same topic continues, the chunk must be split before exceeding the limit.

## Why Overlap Is Required

Important meaning can depend on adjacent turns:

```text
Chunk 1:
REP: Who approves this internally?

Chunk 2:
PROSPECT: Our VP of Finance.
```

Without overlap, the extractor may receive only the question or only the answer. With overlap, both chunks can contain enough context to identify the evidence correctly.

The same evidence may therefore be extracted more than once. This is expected and handled during merging.

## Extraction Output From Each Chunk

Each extraction result must point back to the original source, not just to the chunk text.

```text
section_id
chunk_id
segment_id
speaker
start
end
exact_quote
BANT_field
```

The extractor only identifies information. It does not decide whether the information satisfies a qualification rule.

It must not produce conclusions such as:

- Budget confirmed
- Authority failed
- Lead qualified
- Timeline acceptable

## Merging Overlapping Results

After extraction, results from all child chunks are combined.

Duplicate evidence is identified using the original segment ID and source timestamps. The merge process:

- Combines results from every chunk
- Removes duplicate references caused by overlap
- Preserves separate statements from different moments
- Sorts evidence chronologically
- Validates that each quote exists in the original transcript

The original transcript remains authoritative if two overlapping chunks return slightly different versions of the same quote.

## Relationship With Other JSON Outputs

The signal files are not chunked independently.

The process is:

```text
transcript.json
    |
    v
hierarchical transcript chunks
    |
    v
BANT evidence: quote + speaker + timestamp
    |
    +-------------------+
    |                   |
    v                   v
audio_features.json  voice_emotion.json
    |                   |
    +---------+---------+
              |
              v
       face_emotion.json
              |
              v
    Signals linked to evidence
```

For each extracted evidence moment:

1. Use its timestamp to find the matching audio measurements.
2. Use its timestamp to find the matching voice measurements.
3. Use the nearest available facial sample when one exists.
4. Mark a measurement as unavailable if no matching signal exists.

No model is run again. No signal is interpreted during this linking step.

## Why We Do Not Chunk the Signal Files

The signal files do not contain independent prose that needs semantic extraction. They are measurements indexed by time and, for most records, by transcript segment.

Chunking them separately could create alignment problems and duplicate data. The transcript evidence provides the lookup point; the signal files provide the measurements for that point.

## Final Data Flow

```text
Full transcript
    -> conversation sections
    -> overlapping extraction chunks
    -> BANT evidence from original text
    -> merge and deduplicate by source segment
    -> attach existing audio, voice, and face measurements
    -> qualification.json
```

The final artifact contains extracted information and linked measurements. Qualification decisions remain outside this system.

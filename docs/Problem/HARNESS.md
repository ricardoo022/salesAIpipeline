# BANT Extraction Harness

## Purpose

The harness coordinates the extraction of BANT information from a long, already-processed sales call.

Its purpose is to provide the company with complete, grounded, timestamped information. It does not decide whether the lead qualifies.

The harness extracts and associates evidence only:

- Budget
- Authority
- Need
- Timeline

## Core Design

The harness uses four parallel topic agents. Each agent owns one BANT topic and performs both extraction and evidence validation for that topic.

```text
                         +----------------+
                         | Existing       |
                         | transcript     |
                         | hierarchy      |
                         +--------+-------+
                                  |
              +-------------------+-------------------+
              |                   |                   |
              v                   v                   v
       +-------------+     +-------------+     +-------------+
       | Budget      |     | Authority   |     | Need        |
       | agent       |     | agent       |     | agent       |
       |             |     |             |     |             |
       | Extract +   |     | Extract +   |     | Extract +   |
       | validate    |     | validate    |     | validate    |
       +------+------+     +------+------+     +------+------+
              |                   |                   |
              +-------------------+-------------------+
                                  |
                                  v
                         +----------------+
                         | Timeline agent |
                         |                |
                         | Extract +      |
                         | validate       |
                         +--------+-------+
                                  |
                                  v
                         +----------------+
                         | Evidence       |
                         | assembler      |
                         +--------+-------+
                                  |
                                  v
                         +----------------+
                         | Signal linker  |
                         |                |
                         | Existing audio |
                         | voice and face |
                         +--------+-------+
                                  |
                                  v
                         +----------------+
                         | qualification  |
                         | .json          |
                         +----------------+
```

The four agents are logically parallel. The final assembler waits until all four topic results are available.

## Input

The harness receives the outputs already produced by the call-analysis pipeline:

- Timestamped, speaker-labelled transcript
- Audio measurements
- Voice measurements
- Facial measurements

The transcript is organized into hierarchical sections and extraction chunks before it is sent to the topic agents.

The signal files are not independently chunked and are not processed again.

## Topic Agents

Each agent scans the complete transcript hierarchy for its own topic. It does not scan only keyword matches. This allows it to find indirect statements.

For example, the Authority agent should find both:

```text
Our CFO needs to approve the purchase.
```

and:

```text
I cannot sign this myself. It has to go through Finance.
```

### Budget Agent

Extracts transcript evidence related to:

- Available funds
- Budget allocation
- Price or cost constraints
- Spending limits
- Approval thresholds
- Financial objections

### Authority Agent

Extracts transcript evidence related to:

- Decision-makers
- Approval processes
- Sign-off requirements
- Finance or procurement involvement
- Other stakeholders involved in the decision

### Need Agent

Extracts transcript evidence related to:

- Current problems
- Business pain
- Desired outcomes
- Consequences of not solving the problem
- Reasons for seeking a solution

### Timeline Agent

Extracts transcript evidence related to:

- Desired implementation date
- Deadlines
- Buying windows
- Procurement timing
- Events creating a time constraint

## Agent Output

Each topic agent returns all relevant evidence it finds. Each evidence item must point back to the original transcript.

```text
topic
segment_id
speaker
start
end
exact_quote
source_chunk_id
```

The agent also validates that:

- The quote exists in the original transcript
- The source segment exists
- The speaker is correct
- The timestamp is correct
- The evidence belongs to the assigned BANT topic

This validation concerns extraction accuracy only. It does not evaluate the business meaning of the evidence.

## What Agents Must Not Produce

Topic agents must not produce:

- Qualified or unqualified
- Confirmed or rejected
- Sufficient or insufficient
- Good or bad lead
- Budget accepted
- Authority confirmed
- Need strong enough
- Timeline acceptable

Those decisions depend on each company's own KPIs and remain outside the harness.

## Evidence Assembly

After the four topic agents finish, their results are combined by an assembler.

The assembler is deterministic. It does not use an LLM to decide how to merge the outputs.

It performs the following operations:

1. Collect the four topic results.
2. Remove duplicate references caused by overlapping transcript chunks.
3. Preserve separate statements from different moments.
4. Sort evidence chronologically within each BANT topic.
5. Preserve the original quote, speaker, timestamp, and source IDs.
6. Allow the same transcript segment to appear under more than one topic when appropriate.

For example:

> “Our CFO needs to approve it before the end of the quarter.”

This may correctly appear under both:

- Authority: CFO approval
- Timeline: end of the quarter

The assembler must not force the statement into only one category.

## Signal Linking

Signal linking happens after evidence assembly.

For each validated evidence item, the harness uses its timestamp and source segment to attach the existing measurements:

- Audio measurements for the relevant transcript segment
- Voice measurements for the relevant transcript segment
- The nearest available facial measurement

```text
Validated transcript evidence
            |
            v
      timestamp + segment ID
            |
      +-----+-----+-----+
      |           |     |
      v           v     v
    audio       voice  face
   signals     signals signals
            |
            v
   evidence with measurements
```

The linker does not interpret the measurements. It only associates them with the transcript evidence.

If a measurement is unavailable, the result records it as unavailable. It must not invent a value or replace missing data with zero.

## Harness State

The harness maintains state for:

- Call identifier
- Transcript segments
- Hierarchical sections
- Extraction chunks
- Results from each topic agent
- Validation results
- Assembled evidence
- Linked measurements
- Processing errors and retries

Each extracted item retains:

- Topic agent name
- Section ID
- Chunk ID
- Transcript segment ID
- Attempt number

This makes it possible to retry one failed topic or chunk without rerunning the entire call.

## Failure Handling

The harness distinguishes between valid absence of information and processing failure.

Valid result:

```text
No Timeline evidence found.
```

Processing failure:

```text
The Authority agent failed to process chunk 12.
```

Processing failures should be retried locally. A retry must not change the meaning of the output or add a qualification decision.

If an evidence candidate cannot be grounded in the original transcript, it is excluded from the final artifact and retained in the processing audit information.

## Final Artifact

The final `qualification.json` contains the four assembled evidence collections:

```text
qualification.json
├── budget
│   └── evidence[]
├── authority
│   └── evidence[]
├── need
│   └── evidence[]
└── timeline
    └── evidence[]
```

Each evidence item contains the original transcript information and the linked measurements. The artifact contains information for the company's own qualification process, not a qualification result.

## Boundary of Responsibility

```text
Harness:
  extract information
  validate source grounding
  merge evidence
  link measurements

Company:
  define KPIs
  interpret evidence
  decide whether the lead qualifies
```

The harness is successful when it gives the company the right evidence in a complete, traceable, and consistent format. It is not responsible for deciding what that evidence means for qualification.

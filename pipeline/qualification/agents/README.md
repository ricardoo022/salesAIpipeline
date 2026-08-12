# BANT Topic Agents

This directory will contain the four topic-specific extraction agents:

- Budget
- Authority
- Need
- Timeline

Each agent extracts and validates evidence for one topic. Agents do not decide whether a lead qualifies or whether a BANT criterion is satisfied.

Each agent receives the complete validated chunk hierarchy, not a vector-search
subset. Chunks contain rendered speaker-labeled conversation text and source
segment IDs. Agent results must preserve exact quotes, speakers, timestamps,
topic, chunk ID, and segment ID so they can be grounded and assembled later.

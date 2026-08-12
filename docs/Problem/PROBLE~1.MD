# Problem Statement: CRM & Deal Qualification Data

## The problem

CRM data isn't untrusted because reps are bad at typing. It's untrusted because what gets typed is a human's after-the-fact, self-interested summary of a call — filtered through memory, quota pressure, and whatever that rep personally believes "qualified" means.

That's the same root cause behind both symptoms:

- The data entry burden exists because a human has to reconstruct the call from memory.
- The inconsistency exists because that reconstruction is subjective.

Fix the root cause — stop relying on a human's memory and judgment as the source of the data — and both symptoms go away at once.

## What this means concretely

CRMs already have a place for this data. Contact/account fields (name, title, company) are reliable because they're entered once, mechanically, and rarely disputed. Deal-qualification fields (Budget, Authority, Need, Timeline, Champion, etc.) are a different story: they only exist if a sales org has configured custom fields for their chosen framework (BANT, MEDDIC, MEDDPICC...), and even then they're chronically empty or stale — because filling them out requires a rep to remember and self-report them after the call.

The opportunity isn't inventing new CRM fields. It's populating fields that already exist in most CRM schemas but are unreliable because a human was the only source for them.

## The fix

Extract the qualification fields directly from the call itself — transcript, voice, and facial signal — instead of from a rep's memory:

- Same criteria, applied the same way, every call, every rep (fixes inconsistency).
- Auto-populated instead of manually typed (fixes the data entry burden).
- Evidence-linked (quote + timestamp + confidence read from tone/face), harder to fabricate under quota pressure and easier for sales ops to audit.

This reuses the same call data the pipeline already extracts (transcript + audio features + voice emotion + facial emotion per segment). It requires no new recording, no new data source, and no same-rep/multi-call history — it works on a single call.

## Two separate fixes, not one

"Automatic" and "not relying on non-deterministic human judgment" sound like the same fix. They're not — they're two separate problems stacked on top of each other, solved by two different parts of the system.

**Fix 1 — consistency.** An LLM reading only the transcript and applying the same extraction rule to every call, every rep, every time, is already more consistent than five reps making five subjective judgment calls under quota pressure. This alone fixes the inconsistency-between-reps problem. It does not need voice or face.

**Fix 2 — truthfulness.** Transcript-only automation has a ceiling: it can only be as accurate as what was literally said. If a prospect says "yeah, budget's approved" to be polite, to avoid looking uninformed, or while hedging without realizing it, a transcript-only extractor marks Budget: Confirmed and moves on — consistent, but consistently wrong. This is the same failure mode as a human reading the same transcript.

Fixing this requires the voice/face signal: not "was this said" but "did the way they said it actually support it being true." Same mechanism as the original demo's "No, it looks great" moment, applied to qualification criteria instead of general sentiment.

**Bottom line:** automation alone gets a consistent but naive system. The multimodal layer is what makes it more accurate than just trusting the transcript — it's the reason this is worth building instead of just wiring an LLM to read call transcripts into Salesforce fields.

## Memory vs. judgment — a worked example

These are two separate failure modes, not one causing the other. Same call, one sentence, both failures shown independently:

Prospect says: *"We'd probably need sign-off from our VP of Finance, but I don't think that'll be an issue."*

**Memory failure** — the rep does the call, fills in the CRM four days later, and writes "Authority: confirmed, no blockers" because they genuinely forgot the VP-of-Finance detail happened. Show them the transcript and they'd say "oh right, I forgot about that" and change their answer.

**Judgment failure** — a different rep remembers that sentence perfectly. They still mark "Authority: confirmed" because they read "probably need sign-off but I don't think it's an issue" as strong enough. A third rep, remembering the exact same sentence just as accurately, marks "Authority: not confirmed" because any required sign-off means the contact isn't the real decision-maker to them. Nobody forgot anything — they disagree on what the same remembered fact means.

**The test that tells them apart:** re-show the rep the transcript. If their answer changes, it was memory. If their answer stays the same but a different rep reading the identical transcript reaches a different conclusion, it's judgment.

**Why it doesn't matter which one is "the" problem:** the fix is the same either way. Extract straight from the transcript/signal (kills memory failure — nothing has to be remembered) and apply one fixed rule every time (kills judgment failure — nothing gets interpreted differently call to call). One mechanism solves both, so there's no need to decide which one is bigger before building it.
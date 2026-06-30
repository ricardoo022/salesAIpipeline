# input/

Place the meeting video here before running the pipeline.

## Required file

**`meeting.mp4`** — the B2B sales meeting recording to analyse.

Download the demo video:

```bash
yt-dlp https://youtu.be/N0SF2nZS-S8 -o input/meeting.mp4
```

Or place any `.mp4` recording here with that filename. The pipeline reads only `input/meeting.mp4` — no other filename is supported.

This directory is gitignored. Video files should not be committed.

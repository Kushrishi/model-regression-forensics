# Superseded Exp009 baseline-target draft

**Status:** superseded before any modern attribution ranking was observed.

The content below is preserved only for research provenance. The authoritative
contract is `research/ATTRIBUTION_TARGET.md`.

---

# Superseded baseline-target draft

This document is retained only as a pointer for provenance.

The earlier baseline-target draft used:

- a correct-class-vs-all-classes margin;
- class-balanced target averaging;
- a baseline-to-composite differential;
- mean aggregation over changed slots.

That design was superseded **before any modern attribution ranking was
observed**.

The authoritative Exp009 attribution contract is now:

- `research/ATTRIBUTION_TARGET.md`

The current contract instead freezes:

- the correct-vs-paired-target margin;
- equal weighting of all development-eval examples in the target pair;
- method-specific suspiciousness orientation defined mathematically;
- **sum** aggregation over each opaque debugger-visible candidate change.

Do not implement new attribution work from this file.

Historical content remains recoverable from Git history.

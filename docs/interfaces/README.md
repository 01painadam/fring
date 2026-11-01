# Interface change requests (ICR)

An ICR asks to change a shared interface: the `Backend` trait and types in `fring-model`, the snapshot format, the config schema or the collision specification. The process is in `AGENTS.md`, section "Interfaces".

File name: `ICR-NNN.md`, with the next free three-digit number. Only A0 writes "Accepted" or "Rejected".

## Template

```markdown
# ICR-NNN: <title>
Requested by: <A-n>  Date: <YYYY-MM-DD>
Status: open | Accepted | Rejected

## What
The exact change (types, fields, functions, file format).

## Why
The problem that the current interface causes.

## Roles affected
Which roles must change code, and what.

## Migration
Steps, in order. Is old data or code still usable?

## A0 decision
<date>: Accepted / Rejected, with the reason.
```

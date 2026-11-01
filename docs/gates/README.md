# Gates

A gate ends a phase (`docs/PLAN.md`, section 9). A0 writes `G<n>.md` with the evidence. AJ reads it and writes "Approved", or what must change. No role starts a task of the next phase before AJ writes "Approved".

## Template

```markdown
# Gate G<n>: <phase name>
Prepared by: A0  Date: <YYYY-MM-DD>

## What AJ decides
(from docs/PLAN.md section 9)

## Evidence
| Task | Result (numbers) | Link (PR, report, plot) |
|---|---|---|

## Open issues and risks

## Decision (AJ)
<date>: Approved / Not approved. Notes:
```

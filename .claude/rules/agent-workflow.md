# Agent Workflow Protocol

**Mandatory Procedures for Codex/Gemini Consultations**

## Correct Workflow

### Phase 1: Information Gathering
1. Read relevant files.
2. Verify existing Codex analysis (`.claude/logs/cli-tools.jsonl`).
3. Identify the scope of the problem.

### Phase 2: Codex/Gemini Consultation
4. Launch Codex/Gemini via a sub-agent.
5. Wait for completion using `TaskOutput(block=true, timeout=120000)`.
6. Receive completion notification.
7. Read the files saved by the agent.
8. Verify execution results in CLI logs (`.claude/logs/cli-tools.jsonl`).

### Phase 3: Analysis and Planning
9. Fully understand the Codex/Gemini analysis.
10. Extract key recommendations.
11. Create a prioritized implementation plan.
12. Report to the user.

### Phase 4: Implementation (Codex が実装する)
13. Obtain user approval.
14. **Delegate implementation to Codex** via sub-agent (`gpt-5.3-codex`, `workspace-write`).
15. Claude does NOT write code directly — Codex handles all code and test writing.
16. Claude reviews Codex output and reports to user.

---

## ❌ Prohibited Actions

1. **Claude が直接コードを書くこと**
   - All code implementation and test writing MUST be delegated to Codex.
   - Claude's role is orchestration, planning, review, and user communication only.

2. **Providing your own analysis before the agent completes its task**
   - The Codex analysis must take top priority.

3. **Calling Codex/Gemini directly in parallel**
   - This causes conflicts with the sub-agent.

4. **Giving up when result files cannot be found**
   - Always check the CLI logs.

5. **Changing code abruptly without an implementation plan**
   - A phased plan must always be created first.

---

## Components of the Implementation Plan

The implementation plan must include the following:

1. **Priority of fixes** (Phase 1, 2, 3...)
2. **Scope of impact** (Which files will be modified)
3. **Dependencies** (The order in which fixes should be applied)
4. **Testing methods** (How to verify the results after the fix)
5. **Rollback plan** (How to handle issues if they arise)

---

## Checklist

Confirm the following before implementation:
- [ ] I have fully read the Codex/Gemini analysis.
- [ ] I have understood all key recommendations.
- [ ] I have created a phased implementation plan.
- [ ] I have presented the plan to the user and obtained approval.
- [ ] I am ready to track progress with `TodoWrite`.
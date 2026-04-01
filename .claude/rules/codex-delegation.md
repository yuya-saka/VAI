# Codex Delegation Rule

**Codex CLI is your highly capable supporter.**

## Context Management (CRITICAL)

**コンテキスト消費を意識してCodexを使う。** 大きな出力が予想される場合はサブエージェント経由を推奨。

| 状況 | 推奨方法 |
|------|----------|
| 短い質問・短い回答 | 直接呼び出しOK |
| 詳細な設計相談 | サブエージェント経由 |
| デバッグ分析 | サブエージェント経由 |
| 複数の質問がある | サブエージェント経由 |

```
┌──────────────────────────────────────────────────────────┐
│  Main Claude Code                                        │
│  → 短い質問なら直接呼び出しOK                             │
│  → 大きな出力が予想されるならサブエージェント経由          │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Subagent (general-purpose)                         │ │
│  │  → Calls Codex CLI                                  │ │
│  │  → Processes full response                          │ │
│  │  → Returns key insights only                        │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## About Codex

Codex CLI is an AI with exceptional reasoning and task completion abilities.
Think of it as a trusted senior expert you can always consult.

**When facing difficult decisions → Delegate to subagent → Subagent consults Codex.**

## When to Consult Codex

ALWAYS consult Codex BEFORE:

1. **Design decisions** - How to structure code, which pattern to use
2. **Debugging** - If cause isn't obvious or first fix failed
3. **Implementation planning** - Multi-step tasks, multiple approaches
4. **Trade-off evaluation** - Choosing between options

### Trigger Phrases (User Input)

Consult Codex when user says:

| Japanese | English |
|----------|---------|
| 「どう設計すべき？」「どう実装する？」 | "How should I design/implement?" |
| 「なぜ動かない？」「原因は？」「エラーが出る」 | "Why doesn't this work?" "Error" |
| 「どちらがいい？」「比較して」「トレードオフは？」 | "Which is better?" "Compare" |
| 「〜を作りたい」「〜を実装して」 | "Build X" "Implement X" |
| 「考えて」「分析して」「深く考えて」 | "Think" "Analyze" "Think deeper" |

## When NOT to Consult

Skip Codex for simple, straightforward tasks:

- Simple file edits (typo fixes, small changes)
- Following explicit user instructions
- Standard operations (git commit, running tests)
- Tasks with clear, single solutions
- Reading/searching files

## Quick Check

Ask yourself: "Am I about to make a non-trivial decision?"

- YES → Consult Codex first
- NO → Proceed with execution

## How to Consult (via Subagent)

**IMPORTANT: Use subagent to preserve main context.**

### Recommended: Subagent Pattern

Use Task tool with `subagent_type: "general-purpose"`:

**CRITICAL: Always include explicit instruction to call Codex CLI in the prompt.**

```
Task tool parameters:
- subagent_type: "general-purpose"
- run_in_background: true (for parallel work)
- prompt: |
    CRITICAL: You MUST call Codex CLI. Do NOT provide your own analysis.

    {Task description}

    Step 1: Execute Codex CLI:
    codex exec --model gpt-5.4 --sandbox read-only --full-auto "
    {Question for Codex}
    " 2>/dev/null

    Step 2: ALWAYS save full output to file:
    Use Write tool to save Codex's complete response to:
    .claude/docs/codex/YYYYMMDD-HHMM-{topic}.md

    Format:
    # Codex Analysis: {topic}
    Date: YYYY-MM-DD HH:MM

    ## Question
    {original question}

    ## Codex Response
    {full response from Codex}

    Step 3: Return CONCISE summary to main:
    - Key recommendation
    - Main rationale (2-3 points)
    - File path where saved
```

**Why explicit instruction is required:**
- Without it, subagents may provide their own analysis instead of calling Codex
- This wastes the benefit of Codex's superior reasoning capabilities
- Always start subagent prompts with "CRITICAL: You MUST call Codex CLI"

**File Naming Convention:**
- Format: `YYYYMMDD-HHMM-{topic}.md`
- Example: `20260317-2330-training-collapse.md`
- Location: `.claude/docs/codex/`
- Purpose: Main Claude Code reads these files to implement recommendations

### Direct Call (Only When Necessary)

Only use direct Bash call when:
- Quick, simple question (< 1 paragraph response expected)
- Subagent overhead not justified

```bash
# Only for simple queries
codex exec --model gpt-5.4 --sandbox read-only --full-auto "Brief question" 2>/dev/null
```

### Sandbox Modes & Model Selection

| Mode | Model | Sandbox | Use Case |
|------|-------|---------|----------|
| Analysis | `gpt-5.4` | `read-only` | Design review, debugging analysis, trade-offs |
| Implementation | `gpt-5.3-codex` | `workspace-write` | Implement, fix, refactor (subagent recommended) |

**Language protocol:**
1. Ask Codex in **English**
2. Subagent receives response in **English**
3. Subagent summarizes and returns to main
4. Main reports to user in **Japanese**

## Main Claude Code Workflow (After Subagent Returns)

**CRITICAL: Main must read Codex output file and implement recommendations.**

When subagent returns with summary:

1. **Read the saved file**:
   ```
   Read tool: .claude/docs/codex/{filename}
   ```

2. **Analyze Codex's recommendations**:
   - Review full analysis (not just summary)
   - Identify specific code changes needed
   - Check for risks/concerns mentioned

3. **Implement or report**:
   - If implementation: Use Edit/Write tools to apply fixes
   - If user decision needed: Present Codex's analysis in Japanese
   - If research only: Summarize findings for user

4. **Mark todo as completed**:
   - Update TodoWrite with implementation status

**Example:**
```
Subagent returns: "Saved to .claude/docs/codex/20260317-2330-fix-nan.md"

Main Claude Code:
1. Read(".claude/docs/codex/20260317-2330-fix-nan.md")
2. Identify fixes from Codex analysis
3. Edit(line_losses.py) to apply recommended guards
4. Report to user in Japanese: "Codexの推奨に基づいて修正しました"
```

## Why Subagent Pattern?

- **Context preservation**: Main orchestrator stays lightweight
- **Full analysis**: Subagent can process entire Codex response
- **Concise handoff**: Main only receives actionable summary
- **Parallel work**: Background subagents enable concurrent tasks
- **Persistent records**: All Codex analyses saved in `.claude/docs/codex/`

**Don't hesitate to delegate. Subagents + Codex = efficient collaboration.**

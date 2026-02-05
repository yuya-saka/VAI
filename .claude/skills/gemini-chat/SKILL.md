---
name: gemini-chat
description: |
  Switch to Gemini CLI interactive chat mode when Claude Code hits rate limits.
  Use this skill when Claude Code returns rate limit errors and you want to continue working.
metadata:
  short-description: Switch to Gemini CLI when Claude hits rate limits
---

# Gemini Chat — Fallback for Claude Rate Limits

**Use this skill when Claude Code hits usage/rate limits.**

## When to Use

- Claude Code returns rate limit errors (429, quota exceeded)
- You receive "usage limit" messages
- You want to continue working but Claude is unavailable

## What This Does

1. Notifies you about the switch
2. Launches Gemini CLI in interactive mode
3. You can continue your conversation with Gemini

## Usage

```
/gemini-chat
```

## How It Works

This skill will:
1. Display a message about switching to Gemini
2. Execute `gemini` command to start interactive chat
3. Gemini CLI will open in your terminal

## Context Preservation

Currently, this is a simple handoff. To preserve context:
- Manually summarize what you were working on
- Or check `.claude/logs/` for recent conversation history

## Returning to Claude

When Claude Code is available again:
- Exit Gemini CLI (Ctrl+D or type "exit")
- Restart Claude Code session

## Tips

- Gemini has 1M token context - good for large codebases
- Use `--include-directories .` for codebase access
- Gemini excels at research and analysis
- For implementation, consider waiting for Claude Code to become available again

---

# SKILL PROMPT

Claude Codeのレート制限により、Gemini CLIに切り替えます。

以下のコマンドでGemini CLIインタラクティブモードを起動してください：

```bash
gemini
```

Geminiで作業を継続できます。主な機能：
- 1Mトークンのコンテキストウィンドウ
- リポジトリ全体の分析が可能
- Google検索によるグラウンディング
- マルチモーダル対応（PDF/動画/音声）

コードベースにアクセスする場合：
```bash
gemini --include-directories .
```

Claude Codeが再び利用可能になったら、Geminiを終了（Ctrl+D）してClaude Codeセッションを再開してください。

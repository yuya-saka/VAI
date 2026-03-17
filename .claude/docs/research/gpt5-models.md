# GPT-5 Models Available in Codex CLI (March 2026)

**Research Date:** 2026-03-17

## Summary

GPT-5.4 **is available** in Codex CLI as of March 2026. It is OpenAI's flagship frontier model combining advanced reasoning, coding, and agentic workflows.

---

## Available Models

### Recommended Models (Current)

#### 1. **gpt-5.4** ⭐ NEWEST
- **Description:** Flagship frontier model for professional work
- **Features:**
  - Combines recent advances in reasoning, coding, and agentic workflows
  - First general-purpose model with native computer-use capabilities
  - Experimental support for 1M context window
  - Released: Early March 2026
- **Usage:** `codex -m gpt-5.4` or `codex exec --model gpt-5.4`
- **Availability:** Codex CLI, Codex app, IDE extension, Codex Cloud

#### 2. **gpt-5.3-codex**
- **Description:** Industry-leading coding model for complex software engineering
- **Best For:** Sophisticated development tasks requiring deep code understanding
- **Usage:** `codex -m gpt-5.3-codex`
- **Status:** Generally available (as of Feb 2026)

#### 3. **gpt-5.3-codex-spark** 🔬
- **Description:** Research preview optimized for near-instant, real-time coding iteration
- **Best For:** Rapid prototyping, real-time coding assistance
- **Usage:** `codex -m gpt-5.3-codex-spark`
- **Availability:** ChatGPT Pro subscribers only

---

### Older/Alternative Models

#### **gpt-5.2-codex**
- **Status:** Superseded by GPT-5.3-Codex
- **Description:** Advanced engineering model (previous generation)
- **Note:** Still functional but GPT-5.3-Codex recommended for new work

#### **gpt-5.2**
- **Status:** Succeeded by GPT-5.4
- **Description:** Previous general-purpose model

#### **gpt-5.1-codex-max**
- **Best For:** Long-horizon agentic tasks

#### **gpt-5.1**
- **Best For:** Cross-domain coding and agentic work

#### **gpt-5.1-codex**
- **Status:** Replaced by newer versions

#### **gpt-5-codex**
- **Status:** Deprecated (original tuned version)

---

## Key Differences Between Versions

### GPT-5.4 vs GPT-5.3-Codex vs GPT-5.2-Codex

| Feature | gpt-5.4 | gpt-5.3-codex | gpt-5.2-codex |
|---------|---------|---------------|---------------|
| **Release Date** | March 2026 | Feb 2026 | Earlier 2025 |
| **Primary Focus** | General professional work + coding | Pure coding excellence | Engineering tasks |
| **Reasoning** | Enhanced | Strong | Good |
| **Agentic Workflows** | Native support | Strong | Basic |
| **Computer Use** | Native capabilities | Limited | No |
| **Context Window** | 1M (experimental) | Standard | Standard |
| **Best Use Case** | Full-stack development + reasoning | Complex code engineering | Legacy projects |

### Model Selection Guide

```
Choose gpt-5.4 when:
- You need the latest capabilities
- Task requires reasoning + coding
- Want agentic workflows
- Need computer-use features

Choose gpt-5.3-codex when:
- Pure coding task (complex software engineering)
- Don't need extra reasoning overhead
- Specialized code generation/refactoring

Choose gpt-5.3-codex-spark when:
- Real-time iteration needed
- Prototyping/experimentation
- Have ChatGPT Pro subscription

Choose gpt-5.2-codex when:
- Maintaining legacy compatibility
- Specific requirement for older model
```

---

## How to Use in Codex CLI

### Method 1: Command-line flag
```bash
# One-time usage
codex exec --model gpt-5.4 --full-auto "Your task here"
codex -m gpt-5.4 "Interactive session"
```

### Method 2: Config file (default model)
Edit `~/.codex/config.toml`:
```toml
model = "gpt-5.4"
```

### Method 3: IDE selector
Use model picker in Codex IDE extension

---

## Update Instructions

If GPT-5.4 is not available:
1. Update Codex CLI: Check for latest version
2. Update IDE extension if using
3. Update Codex app if using desktop version

---

## Sources

- [Introducing GPT-5.4 | OpenAI](https://openai.com/index/introducing-gpt-5-4/)
- [Codex Models Documentation](https://developers.openai.com/codex/models/)
- [Introducing GPT-5.2-Codex | OpenAI](https://openai.com/index/introducing-gpt-5-2-codex/)
- [Introducing GPT-5.3-Codex | OpenAI](https://openai.com/index/introducing-gpt-5-3-codex/)
- [GPT-5.3-Codex GitHub Copilot Announcement](https://github.blog/changelog/2026-02-09-gpt-5-3-codex-is-now-generally-available-for-github-copilot/)
- [Codex Changelog](https://developers.openai.com/codex/changelog)

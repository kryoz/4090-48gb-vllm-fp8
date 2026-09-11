"""
Builds a calibration dataset for llm-compressor from agent traffic logs.

Source: JSONL session files from the pi coding agent (`~/.pi/agent/sessions/`),
recursively across all project directories. Subdirectories of a session folder
contain SUBAGENT sessions (separate real conversations with their own context)
— they are included. Files in a different format (e.g. `subagent-artifacts/
*_transcript.jsonl`) are skipped: a file must start with a `session` header
to be treated as a pi session.

pi session format (vs the old OMP format this script was adapted from):
- The FIRST record is a `session` header (id/cwd/version). There is NO
  `session_init` event — pi does NOT persist the system prompt in the
  session file. The oh-my-pi extension DOES emit `session_init` with
  `systemPrompt`; such an event still splits a file into separate samples,
  but the systemPrompt is deliberately DROPPED so that all samples are
  homogeneous (dialogue only, no system message).
- Records form a TREE via `id`/`parentId` (branching/rewinding creates
  multiple children for one parent). The context the model actually saw in
  its last forward pass is the CHAIN from the leaf (last record in the file)
  back to the root — not the raw file order. Branched files (~12% of the
  corpus) would be corrupted if read linearly.
- Extra message roles: `toolResult` (maps to "tool"), `developer`,
  `fileMention`, `bashExecution` (all mapped to "user" — same as pi's own
  convertToLlm in dist/core/messages.js).
- `custom_message` entries participate in LLM context as user messages
  (extension injections like wiki-recall context) — included.
- `compaction` entries RESET the context: the model afterwards sees only
  the compaction summary + messages from `firstKeptEntryId` onward. We
  mirror this: a compaction starts a NEW sample with the summary as the
  first user message, and pre-kept entries are skipped.
- Pure metadata entries (model_change, thinking_level_change, custom,
  label, title*, session_info, mode_change, ...) are ignored.

Dialogue boundaries: a sample is one session file's leaf chain, further
split only by `session_init` (omp) and `compaction` events. A single pi
session is a long multi-turn chain and the model sees the ENTIRE
accumulated history in one forward pass — do NOT split at each user
message; that would destroy exactly the long context the stratification
into buckets is meant to preserve.

Known limitation: the (often large) system prompt is absent from ALL
samples — plain pi sessions do not persist it, and it is dropped from omp
sessions for homogeneity — so estimated lengths are biased downward
relative to production. Dialogue content dominates for medium/long
buckets.

Pipeline: parse -> dedupe (hash of ENTIRE dialogue) -> sanitize secrets/PII
-> stratify by context length (short/medium/long) -> save JSONL with a
"messages" field in chat template format.
"""

import json
import re
import hashlib
import random
from pathlib import Path
from collections import defaultdict

# --- 1. Data source ---------------------------------------------------
RAW_LOG_DIR = Path.home() / ".pi" / "agent" / "sessions"

# Same wrapping pi uses when it puts a compaction/branch summary back into context
COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"
BRANCH_SUMMARY_PREFIX = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
BRANCH_SUMMARY_SUFFIX = "\n</summary>"

SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),                 # sk-... style API keys
    re.compile(r"(?i)(password|passwd|secret)\s*[:=]\s*\S+"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+"),  # email
    re.compile(r"\b\d{16}\b"),                          # looks like a card number
]


def sanitize(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def messages_token_estimate(messages: list) -> int:
    # Rough estimate without a tokenizer — 1 token ~= 3.5-4 characters for a mix of RU/code
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    return int(total_chars / 3.7)


def dedupe_key(messages: list) -> str:
    # Hash the ENTIRE dialogue content. The system prompt (AGENTS.md/skills/rules)
    # is the same for all sessions within a project — hashing only the beginning
    # would collapse different sessions into a single record.
    full_text = "".join(m.get("content", "") or "" for m in messages)
    return hashlib.sha256(full_text.encode()).hexdigest()


def _collect_jsonl_files(root: Path):
    """Recursively collect all *.jsonl files from subdirectories."""
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    for p in sorted(root.rglob("*.jsonl")):
        yield p


def _extract_text_from_blocks(content, include_tool_calls: bool = True) -> str:
    """
    Extract text from multi-part pi message content.

    include_tool_calls: if True, tool calls are serialized as short text
    markers instead of being discarded entirely — real agent traffic is
    structurally defined by tool calls, and removing this information
    completely would mean calibrating on data that does not match what the
    model actually sees.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(block.get("text", "") or "")
            elif block_type == "thinking":
                # Reasoning tokens — you have --reasoning-parser qwen3 enabled
                # in production, so the model actually generates and processes
                # these tokens as part of the context. Keep a short fragment
                # instead of discarding them completely.
                thinking_text = block.get("thinking", "") or block.get("text", "") or ""
                parts.append(str(thinking_text)[:300])
            elif block_type == "toolCall" and include_tool_calls:
                name = block.get("name", "unknown_tool")
                args = str(block.get("arguments", ""))[:200]
                parts.append(f"[tool_call: {name}({args})]")
            elif block_type == "tool_result":
                tool_text = block.get("content", "")
                if isinstance(tool_text, list):
                    tool_text = " ".join(
                        b.get("text", "") for b in tool_text if isinstance(b, dict)
                    )
                parts.append(str(tool_text)[:500])
        return " ".join(parts).strip()
    return str(content)


def _message_to_chat(msg: dict):
    """
    Convert one pi `message` record's message to a chat-template message
    (or None if it carries no usable text). Role mapping mirrors pi's own
    convertToLlm (dist/core/messages.js): bashExecution/custom/summaries ->
    user, toolResult -> tool.
    """
    role = msg.get("role")
    if role in ("user", "assistant"):
        text = _extract_text_from_blocks(msg.get("content"))
        return {"role": role, "content": text} if text.strip() else None

    if role == "toolResult":
        text = _extract_text_from_blocks(msg.get("content"))
        return {"role": "tool", "content": text} if text.strip() else None

    if role == "bashExecution":
        parts = [f"Ran `{msg.get('command', '')}`"]
        if msg.get("output"):
            parts.append(f"```\n{msg['output']}\n```")
        else:
            parts.append("(no output)")
        text = "\n".join(parts)
        return {"role": "user", "content": text} if text else None

    if role == "fileMention":
        files = msg.get("files") or []
        text = "\n\n".join(
            f"{f.get('path', '')}\n{f.get('content', '')}"
            for f in files if isinstance(f, dict)
        )
        return {"role": "user", "content": text} if text else None

    if role == "developer":
        # System-reminder-style injections delivered as their own turn
        text = _extract_text_from_blocks(msg.get("content"))
        return {"role": "user", "content": text} if text.strip() else None

    return None


def _leaf_chain(entries: list) -> list:
    """
    Reconstruct the branch the model actually saw: walk from the LAST record
    (the current leaf) back to the root via parentId, then reverse.

    Falls back to raw file order if the records lack parentId links (older
    / foreign formats). The header (`type: session`) is excluded.
    """
    body = [e for e in entries if e.get("type") != "session"]
    if not body:
        return []
    if not all("parentId" in e and "id" in e for e in body):
        return body
    by_id = {e["id"]: e for e in body}
    chain = []
    cur = body[-1]
    seen = set()
    while cur is not None and cur["id"] not in seen:
        seen.add(cur["id"])
        chain.append(cur)
        cur = by_id.get(cur.get("parentId")) if cur.get("parentId") else None
    chain.reverse()
    return chain


def load_pi_sessions(root: Path):
    """
    Adapter for pi JSONL sessions.

    Yields a stream of INDIVIDUAL samples (each as a list of messages).
    Boundaries: one file = one session (leaf chain), split further by
    `session_init` (omp sessions, which persist the system prompt) and
    `compaction` (context reset — the new sample starts with the summary).
    """
    for jsonl_file in _collect_jsonl_files(root):
        entries = list(_load_jsonl_lines(jsonl_file))
        if not entries or entries[0].get("type") != "session":
            continue  # not a pi session file (e.g. subagent artifacts)

        chain = _leaf_chain(entries)
        if not chain:
            continue

        current_system = None
        current_msgs = []
        skip_until_id = None  # after compaction: skip entries before firstKeptEntryId

        def flush():
            nonlocal current_msgs
            if not current_msgs:
                return None
            msgs = []
            if current_system:
                msgs.append({"role": "system", "content": current_system})
            msgs.extend(current_msgs)
            current_msgs = []
            return msgs

        for record in chain:
            if skip_until_id:
                if record.get("id") == skip_until_id:
                    skip_until_id = None
                else:
                    continue

            rec_type = record.get("type")

            if rec_type == "session_init":
                # omp extension: a new session block inside the same file —
                # close the previous one as a separate sample. The persisted
                # systemPrompt is deliberately DROPPED: plain pi sessions do
                # not persist their system prompt at all, so keeping it for
                # omp sessions only would make the dataset inhomogeneous.
                session = flush()
                if session:
                    yield session
                current_system = None
                continue

            if rec_type == "compaction":
                session = flush()
                if session:
                    yield session
                summary = record.get("summary", "") or ""
                current_msgs = [{
                    "role": "user",
                    "content": COMPACTION_SUMMARY_PREFIX + summary + COMPACTION_SUMMARY_SUFFIX,
                }]
                skip_until_id = record.get("firstKeptEntryId")
                continue

            if rec_type == "branch_summary":
                summary = record.get("summary", "") or ""
                current_msgs.append({
                    "role": "user",
                    "content": BRANCH_SUMMARY_PREFIX + summary + BRANCH_SUMMARY_SUFFIX,
                })
                continue

            if rec_type == "custom_message":
                # Extension content injected into LLM context (rendered as a
                # user message by pi's convertToLlm)
                text = _extract_text_from_blocks(record.get("content"))
                if text.strip():
                    current_msgs.append({"role": "user", "content": text})
                continue

            if rec_type == "message":
                chat_msg = _message_to_chat(record.get("message", {}))
                if chat_msg:
                    current_msgs.append(chat_msg)
                continue
            # everything else (model_change, custom, label, title, ...) — skip

        session = flush()
        if session:
            yield session


def _load_jsonl_lines(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_dataset(
    raw_path: Path,
    output_path: Path,
    target_samples: int = 400,
    max_seq_chars: int = 400_000,  # soft sanity limit, NOT a hard MAX_SEQUENCE_LENGTH filter —
                                    # the tokenizer with truncation=True in calibrate.py will
                                    # truncate anything beyond the actual limit anyway
):
    buckets = defaultdict(list)
    seen_hashes = set()

    # Diagnose the actual length distribution before any stratification,
    # so that buckets are chosen based on real data rather than blindly
    all_token_estimates = []

    for messages in load_pi_sessions(raw_path):
        key = dedupe_key(messages)
        if key in seen_hashes:
            continue
        seen_hashes.add(key)

        for m in messages:
            if isinstance(m.get("content"), str):
                m["content"] = sanitize(m["content"])

        n_tokens_est = messages_token_estimate(messages)
        all_token_estimates.append(n_tokens_est)

        if n_tokens_est < 50:
            continue  # fragments that are too short provide little calibration value

        total_chars = sum(len(m.get("content", "") or "") for m in messages)
        if total_chars > max_seq_chars:
            continue  # only protects against extremely anomalous/corrupted records

        if n_tokens_est < 1000:
            bucket = "short"
        elif n_tokens_est < 8000:
            bucket = "medium"
        else:
            bucket = "long"

        buckets[bucket].append({"messages": messages})

    if all_token_estimates:
        all_token_estimates.sort()
        n = len(all_token_estimates)
        print(f"Total sessions after deduplication: {n}")
        print(
            "Length distribution (estimated tokens): "
            f"p10={all_token_estimates[int(n*0.1)]}, "
            f"p50={all_token_estimates[int(n*0.5)]}, "
            f"p90={all_token_estimates[int(n*0.9)]}, "
            f"max={all_token_estimates[-1]}"
        )

    print("Available by bucket:", {k: len(v) for k, v in buckets.items()})

    quota = {"short": 0.2, "medium": 0.35, "long": 0.45}
    final_samples = []
    for bucket_name, frac in quota.items():
        available = buckets[bucket_name]
        random.shuffle(available)
        n_take = min(len(available), int(target_samples * frac))
        final_samples.extend(available[:n_take])

    random.shuffle(final_samples)
    final_samples = final_samples[:target_samples]

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in final_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Total samples: {len(final_samples)} -> {output_path}")


if __name__ == "__main__":
    build_dataset(
        raw_path=RAW_LOG_DIR,
        output_path=Path("calibration_agentic_samples.jsonl"),
        target_samples=400,
    )

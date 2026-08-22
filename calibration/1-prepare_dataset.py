"""
Builds a calibration dataset for llm-compressor from agent traffic logs.

Source: JSONL session files from omp.sh Agent (`~/.omp/agent/sessions/`),
recursively across all subdirectories.

What the script does:
1. Recursively collects all JSONL files from the OMP session directory
2. Parses session events, extracting user/assistant/toolResult messages
3. Dialogue boundaries are determined by the session_init event, NOT by each
   user message — this is important: a single OMP session can be a long
   multi-turn chain (including service continuation messages such as
   "<system-notice>Continue.</system-notice>"), and in production the model
   sees the ENTIRE accumulated history at once in a single forward pass,
   rather than one message at a time. Splitting a session into chunks at each
   user message would destroy exactly the long context that stratification
   into buckets is intended to preserve.
4. Deduplicates by hashing the ENTIRE dialogue content (not just the beginning —
   otherwise different sessions with the same overall system prompt would
   collapse into a single record)
5. Sanitizes the data: removes potential secrets/PII
6. Stratifies samples by context length — short / medium / long sessions
7. Saves the result as JSONL with a "messages" field in chat template format
"""

import json
import re
import hashlib
import random
from pathlib import Path
from collections import defaultdict

# --- 1. Data source ---------------------------------------------------
RAW_LOG_DIR = Path.home() / ".omp" / "agent" / "sessions"

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
    Extract text from multi-part OMP message content.

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
                args = str(block.get("arguments", block.get("input", "")))[:200]
                parts.append(f"[tool_call: {name}({args})]")
            elif block_type == "tool_result":
                tool_text = block.get("content", "")
                if isinstance(tool_text, list):
                    tool_text = " ".join(
                        b.get("text", "") for b in tool_text if isinstance(b, dict)
                    )
                parts.append(str(tool_text)[:500])
        return " ".join(parts)
    return str(content)


def load_omp_sessions(root: Path):
    """
    Adapter for OMP JSONL sessions.

    Yields a stream of INDIVIDUAL sessions (each as a list of messages), where
    session boundaries are determined by the session_init event rather than
    role-based heuristics.

    A single file may contain multiple consecutive session_init events
    (resume/restart) — each such block becomes a separate sample.
    """
    for jsonl_file in _collect_jsonl_files(root):
        current_system = None
        current_msgs = []

        def flush():
            if not current_msgs:
                return None
            msgs = []
            if current_system:
                msgs.append({"role": "system", "content": current_system})
            msgs.extend(current_msgs)
            return msgs

        for record in _load_jsonl_lines(jsonl_file):
            rec_type = record.get("type")

            if rec_type == "session_init":
                # A new session within the same file — close the previous one as a separate sample
                session = flush()
                if session:
                    yield session
                current_system = record.get("systemPrompt", "") or None
                current_msgs = []
                continue

            if rec_type == "message":
                msg = record.get("message", {})
                role = msg.get("role")
                content = msg.get("content")

                if role == "user":
                    text = _extract_text_from_blocks(content)
                    if text:
                        current_msgs.append({"role": "user", "content": text})
                elif role == "assistant":
                    text = _extract_text_from_blocks(content)
                    if text:
                        current_msgs.append({"role": "assistant", "content": text})
                elif role == "toolResult":
                    text = _extract_text_from_blocks(content)
                    if text:
                        current_msgs.append({"role": "tool", "content": text})

        # Last session in the file
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

    for messages in load_omp_sessions(raw_path):
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

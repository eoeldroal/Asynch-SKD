import argparse
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

from transformers import AutoTokenizer


STUDENT_FINISH = "student_generate_finish"
STUDENT_DELTA = "student_generate_delta"
TEACHER_START = "teacher_verify_start"
TEACHER_FINISH = "teacher_verify_finish"
TEACHER_ERROR = "teacher_verify_error"
FINAL_RESULT = "final_result"


class Colors:
    def __init__(self, enabled):
        self.enabled = enabled
        self.reset = "\033[0m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""
        self.bold = "\033[1m" if enabled else ""
        self.strike = "\033[9m" if enabled else ""
        self.red = "\033[31m" if enabled else ""
        self.green = "\033[32m" if enabled else ""
        self.yellow = "\033[33m" if enabled else ""
        self.blue = "\033[34m" if enabled else ""
        self.magenta = "\033[35m" if enabled else ""
        self.cyan = "\033[36m" if enabled else ""
        self.gray = "\033[90m" if enabled else ""

    def wrap(self, color, text):
        if not self.enabled:
            return text
        return f"{color}{text}{self.reset}"

    def style(self, text, *styles):
        if not self.enabled:
            return text
        return "".join(styles) + text + self.reset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay or tail Async SKD probe JSONL logs with readable streaming diagnostics."
    )
    parser.add_argument("--probe-log", required=True, help="Path to the probe JSONL log.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path or Hugging Face id.")
    parser.add_argument("--verify-top-k", type=int, default=5)
    parser.add_argument("--follow", action="store_true", help="Continue waiting for appended JSONL records.")
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--force-color", action="store_true", help="Emit ANSI colors even when stdout is not a TTY.")
    parser.add_argument("--max-token-width", type=int, default=24)
    parser.add_argument("--show-top-k", type=int, default=3)
    parser.add_argument(
        "--view",
        choices=("sequence", "chat", "compact"),
        default="sequence",
        help="sequence renders one committed stream; chat renders chunk blocks; compact prints one diagnostic line per event.",
    )
    parser.add_argument(
        "--token-delay",
        type=float,
        default=0.0,
        help="Optional seconds to sleep after each rendered token in chat view.",
    )
    parser.add_argument("--max-final-chars", type=int, default=420)
    return parser.parse_args()


def payload(record):
    value = record.get("payload")
    return value if isinstance(value, dict) else {}


def event_kind(record):
    kind = record.get("kind")
    if kind:
        return kind
    return payload(record).get("kind")


def request_id(record):
    return record.get("request_id") or payload(record).get("request_id")


def short_request_id(value):
    if value is None:
        return "-"
    text = str(value)
    return text if len(text) <= 10 else text[-10:]


def compact_text(text, limit=120):
    if text is None:
        return ""
    text = str(text).replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def trunc(text, limit):
    text = compact_text(text, limit=max(limit, 1))
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def as_int_list(value):
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            pass
    return out


def teacher_rows(value):
    if not isinstance(value, list):
        return []
    rows = []
    for row in value:
        rows.append(as_int_list(row) if isinstance(row, list) else [])
    return rows


def matrix_shape(rows):
    if not rows:
        return 0, 0
    return len(rows), max((len(row) for row in rows), default=0)


def eos_ids(tokenizer):
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(token_id) for token_id in eos_token_id}
    return {int(eos_token_id)}


def special_ids(tokenizer):
    ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids.update(eos_ids(tokenizer))
    return {int(token_id) for token_id in ids}


def is_eos(token_id, tokenizer):
    return token_id is not None and int(token_id) in eos_ids(tokenizer)


def is_special(token_id, tokenizer):
    return token_id is not None and int(token_id) in special_ids(tokenizer)


def decode_ids(tokenizer, token_ids, skip_special_tokens):
    if not token_ids:
        return ""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    except Exception as exc:
        return f"<decode_error:{type(exc).__name__}>"


def token_preview(tokenizer, token_id, max_width):
    if token_id is None:
        return "None:''"
    decoded = decode_ids(tokenizer, [token_id], skip_special_tokens=False)
    return f"{token_id}:{trunc(repr(decoded), max_width)}"


def top_tokens_preview(tokenizer, row, show_top_k, max_width):
    if not row:
        return "[]"
    items = [token_preview(tokenizer, token_id, max_width) for token_id in row[:show_top_k]]
    return "[" + ", ".join(items) + "]"


def print_line(colors, color, text):
    print(colors.wrap(color, text), flush=True)


def print_piece(colors, color, text, *, end="", flush=True):
    print(colors.wrap(color, text), end=end, flush=flush)


def visible_text(text):
    return str(text).replace("\r", "\\r")


class StreamProbeViewer:
    def __init__(
        self,
        tokenizer,
        colors,
        verify_top_k,
        show_top_k,
        max_token_width,
        view,
        token_delay,
        max_final_chars,
    ):
        self.tokenizer = tokenizer
        self.colors = colors
        self.verify_top_k = verify_top_k
        self.show_top_k = show_top_k
        self.max_token_width = max_token_width
        self.view = view
        self.token_delay = max(0.0, token_delay)
        self.max_final_chars = max_final_chars
        self.pending_students = defaultdict(deque)
        self.started_reqs = set()
        self.streamed_reqs = set()
        self.open_stream_reqs = set()
        self.sequence_started = False

    def process_line(self, line, line_no):
        stripped = line.strip()
        if not stripped:
            return
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            print_line(self.colors, self.colors.red, f"BAD_JSON line={line_no} error={exc}")
            return
        if not isinstance(record, dict):
            print_line(self.colors, self.colors.red, f"BAD_JSON line={line_no} error=record is not an object")
            return
        self.process_record(record)

    def process_record(self, record):
        kind = event_kind(record)
        if kind == STUDENT_DELTA:
            self.handle_student_delta(record)
        elif kind == STUDENT_FINISH:
            self.handle_student_finish(record)
        elif kind == TEACHER_START:
            self.handle_teacher_start(record)
        elif kind == TEACHER_FINISH:
            self.handle_teacher_finish(record)
        elif kind == TEACHER_ERROR:
            self.handle_teacher_error(record)
        elif kind == FINAL_RESULT:
            self.handle_final_result(record)
        elif kind in {
            "probe_start",
            "probe_finish",
            "student_generate_start",
            "boundary_result",
            "mixed_current_start",
            "mixed_current_partial",
            "mixed_current_completed_without_partial",
            "mixed_current_finish",
        }:
            self.handle_status(record, kind)

    def handle_student_finish(self, record):
        req = request_id(record)
        data = payload(record)
        token_ids = as_int_list(data.get("token_ids"))
        self.pending_students[req].append(record)
        if self.view == "sequence":
            return
        if self.view == "chat":
            if req in self.streamed_reqs:
                if req in self.open_stream_reqs:
                    print("", flush=True)
                    self.open_stream_reqs.discard(req)
                return
            self.render_student_chunk_chat(req, token_ids, data)
            return
        decoded_raw = decode_ids(self.tokenizer, token_ids, skip_special_tokens=False)
        decoded_clean = decode_ids(self.tokenizer, token_ids, skip_special_tokens=True)
        clean_preview = compact_text(decoded_clean)
        raw_preview = compact_text(decoded_raw)
        text_part = f"text='{clean_preview}'"
        if raw_preview != clean_preview:
            text_part += f" raw='{raw_preview}'"
        stop_reason = data.get("stop_reason")
        print_line(
            self.colors,
            self.colors.cyan,
            (
                f"STUDENT req={short_request_id(req)} chunk_len={len(token_ids)} "
                f"stop_reason={stop_reason!r} {text_part}"
            ),
        )

    def handle_student_delta(self, record):
        if self.view not in {"chat", "sequence"}:
            return
        req = request_id(record)
        data = payload(record)
        token_ids = as_int_list(data.get("token_ids"))
        if self.view == "sequence":
            return
        if req not in self.started_reqs:
            print_line(self.colors, self.colors.gray, f"\n╭─ student req={short_request_id(req)} streaming")
            self.started_reqs.add(req)
        if req not in self.open_stream_reqs:
            print_piece(self.colors, self.colors.cyan, "│ assistant ")
            self.open_stream_reqs.add(req)
        self.streamed_reqs.add(req)
        for token_id in token_ids:
            text = decode_ids(self.tokenizer, [token_id], skip_special_tokens=False)
            print_piece(self.colors, self.colors.cyan, visible_text(text))
            if self.token_delay > 0:
                time.sleep(self.token_delay)

    def render_student_chunk_chat(self, req, token_ids, data):
        stop_reason = data.get("stop_reason")
        if req not in self.started_reqs:
            print_line(
                self.colors,
                self.colors.gray,
                (
                    f"\n╭─ student req={short_request_id(req)} "
                    f"chunk={len(token_ids)} stop={stop_reason!r}"
                ),
            )
            self.started_reqs.add(req)
        print_piece(self.colors, self.colors.cyan, "│ assistant ")
        for token_id in token_ids:
            text = decode_ids(self.tokenizer, [token_id], skip_special_tokens=False)
            print_piece(self.colors, self.colors.cyan, visible_text(text))
            if self.token_delay > 0:
                time.sleep(self.token_delay)
        print("", flush=True)

    def handle_teacher_start(self, record):
        req = request_id(record)
        data = payload(record)
        if self.view == "sequence":
            return
        if self.view == "chat":
            if req in self.open_stream_reqs:
                print("", flush=True)
                self.open_stream_reqs.discard(req)
            print_line(
                self.colors,
                self.colors.gray,
                (
                    f"│ teacher  verifying suffix={data.get('expected_suffix_len')} "
                    f"seq={data.get('sequence_len')} start={data.get('logprob_start_len')}"
                ),
            )
            return
        print_line(
            self.colors,
            self.colors.dim,
            (
                f"TEACHER_START req={short_request_id(req)} sequence_len={data.get('sequence_len')} "
                f"logprob_start_len={data.get('logprob_start_len')} "
                f"expected_suffix_len={data.get('expected_suffix_len')}"
            ),
        )

    def handle_teacher_finish(self, record):
        req = request_id(record)
        data = payload(record)
        rows = teacher_rows(data.get("teacher_ids"))
        rows_count = data.get("rows")
        width = data.get("width")
        if rows_count is None or width is None:
            rows_count, width = matrix_shape(rows)
        if self.view == "compact":
            print_line(
                self.colors,
                self.colors.magenta,
                f"TEACHER req={short_request_id(req)} rows={rows_count} width={width}",
            )

        if not self.pending_students[req]:
            print_line(self.colors, self.colors.red, f"ROW_MISMATCH req={short_request_id(req)} reason=no_pending_student")
            return

        student_record = self.pending_students[req].popleft()
        student_ids = as_int_list(payload(student_record).get("token_ids"))
        if len(rows) != len(student_ids):
            print_line(
                self.colors,
                self.colors.red,
                (
                    f"ROW_MISMATCH req={short_request_id(req)} "
                    f"student_len={len(student_ids)} teacher_rows={len(rows)}"
                ),
            )

        first_reject_pos = None
        for idx, token_id in enumerate(student_ids):
            row = rows[idx] if idx < len(rows) else []
            if token_id not in row[: self.verify_top_k]:
                first_reject_pos = idx
                break

        if first_reject_pos is None:
            if self.view == "sequence":
                self.render_sequence_accept_all(student_ids)
                return
            if self.view == "chat":
                self.render_accept_all_chat(student_ids, rows_count, width)
                return
            print_line(
                self.colors,
                self.colors.green,
                f"ACCEPT_ALL req={short_request_id(req)} accept_count={len(student_ids)} first_reject_pos=None",
            )
            return

        row = rows[first_reject_pos] if first_reject_pos < len(rows) else []
        student_token = student_ids[first_reject_pos]
        replacement = row[0] if row else None
        replacement_is_eos = is_eos(replacement, self.tokenizer)
        replacement_is_special = is_special(replacement, self.tokenizer)
        label = "REJECT"
        color = self.colors.yellow
        if replacement_is_eos:
            label = "REPLACE_EOS"
            color = self.colors.red
        elif replacement_is_special:
            label = "REPLACE_SPECIAL"
            color = self.colors.red
        top1 = row[0] if row else None
        if self.view == "sequence":
            self.render_sequence_reject(
                student_ids=student_ids,
                first_reject_pos=first_reject_pos,
                row=row,
                student_token=student_token,
                replacement=replacement,
                replacement_is_eos=replacement_is_eos,
                replacement_is_special=replacement_is_special,
            )
            return
        if self.view == "chat":
            self.render_reject_chat(
                student_ids=student_ids,
                first_reject_pos=first_reject_pos,
                row=row,
                student_token=student_token,
                replacement=replacement,
                replacement_is_eos=replacement_is_eos,
                replacement_is_special=replacement_is_special,
                rows_count=rows_count,
                width=width,
            )
            return
        print_line(
            self.colors,
            color,
            (
                f"{label} req={short_request_id(req)} accept_count={first_reject_pos} "
                f"first_reject_pos={first_reject_pos} "
                f"student={token_preview(self.tokenizer, student_token, self.max_token_width)} "
                f"teacher_top1={token_preview(self.tokenizer, top1, self.max_token_width)} "
                f"replacement_id={replacement} replacement_is_eos={replacement_is_eos} "
                f"replacement_is_special={replacement_is_special} "
                f"top={top_tokens_preview(self.tokenizer, row, self.show_top_k, self.max_token_width)}"
            ),
        )

    def ensure_sequence_started(self):
        if not self.sequence_started:
            print_line(self.colors, self.colors.blue, "\nCOMMITTED SEQUENCE")
            self.sequence_started = True

    def render_sequence_tokens(self, token_ids, color=""):
        for token_id in token_ids:
            text = decode_ids(self.tokenizer, [token_id], skip_special_tokens=False)
            print_piece(self.colors, color, visible_text(text), flush=True)
            if self.token_delay > 0:
                time.sleep(self.token_delay)

    def render_sequence_accept_all(self, student_ids):
        self.ensure_sequence_started()
        self.render_sequence_tokens(student_ids)

    def render_sequence_reject(
        self,
        *,
        student_ids,
        first_reject_pos,
        row,
        student_token,
        replacement,
        replacement_is_eos,
        replacement_is_special,
    ):
        self.ensure_sequence_started()
        label_color = self.colors.red if replacement_is_eos or replacement_is_special else self.colors.yellow
        self.render_sequence_tokens(student_ids[:first_reject_pos])
        if replacement is not None:
            self.render_sequence_tokens([replacement], self.colors.bold + label_color if self.colors.enabled else "")

    def render_accept_all_chat(self, student_ids, rows_count, width):
        committed = decode_ids(self.tokenizer, student_ids, skip_special_tokens=False)
        print_line(
            self.colors,
            self.colors.green,
            f"│ verify   ✓ accept all {len(student_ids)}/{len(student_ids)} rows={rows_count}x{width}",
        )
        print_piece(self.colors, self.colors.green, "│ commit   ")
        print_piece(self.colors, self.colors.green, visible_text(committed))
        print("", flush=True)
        print_line(self.colors, self.colors.gray, "╰─")

    def render_reject_chat(
        self,
        *,
        student_ids,
        first_reject_pos,
        row,
        student_token,
        replacement,
        replacement_is_eos,
        replacement_is_special,
        rows_count,
        width,
    ):
        accepted_ids = student_ids[:first_reject_pos]
        dropped_ids = student_ids[first_reject_pos:]
        accepted_text = decode_ids(self.tokenizer, accepted_ids, skip_special_tokens=False)
        dropped_text = decode_ids(self.tokenizer, dropped_ids, skip_special_tokens=False)
        replacement_text = decode_ids(self.tokenizer, [replacement], skip_special_tokens=False) if replacement is not None else ""
        top1 = row[0] if row else None
        label = "replace"
        label_color = self.colors.yellow
        if replacement_is_eos or replacement_is_special:
            label = "replace-special"
            label_color = self.colors.red

        print_line(
            self.colors,
            label_color,
            (
                f"│ verify   ✗ reject @{first_reject_pos} "
                f"accepted={first_reject_pos}/{len(student_ids)} rows={rows_count}x{width}"
            ),
        )
        print_line(
            self.colors,
            self.colors.yellow,
            (
                f"│ token    student={token_preview(self.tokenizer, student_token, self.max_token_width)} "
                f"teacher_top1={token_preview(self.tokenizer, top1, self.max_token_width)} "
                f"top={top_tokens_preview(self.tokenizer, row, self.show_top_k, self.max_token_width)}"
            ),
        )
        print_piece(self.colors, self.colors.green, "│ commit   ")
        if accepted_text:
            print_piece(self.colors, self.colors.green, visible_text(accepted_text))
        print(self.colors.style(visible_text(replacement_text), self.colors.bold, label_color), flush=True)
        print_piece(self.colors, self.colors.red, "│ drop     ")
        print(self.colors.style(visible_text(dropped_text), self.colors.dim, self.colors.strike, self.colors.red), flush=True)
        special_flags = []
        if replacement_is_eos:
            special_flags.append("EOS")
        if replacement_is_special:
            special_flags.append("SPECIAL")
        suffix = f" [{' '.join(special_flags)}]" if special_flags else ""
        print_line(
            self.colors,
            label_color,
            f"│ {label}  {token_preview(self.tokenizer, replacement, self.max_token_width)}{suffix}",
        )
        print_line(self.colors, self.colors.gray, "╰─")

    def handle_teacher_error(self, record):
        req = request_id(record)
        error = payload(record).get("error")
        print_line(self.colors, self.colors.red, f"TEACHER_ERROR req={short_request_id(req)} error={error!r}")

    def handle_final_result(self, record):
        data = payload(record)
        response_ids = as_int_list(data.get("response_ids"))
        decoded = data.get("decoded_skip_special")
        if decoded is None:
            decoded = decode_ids(self.tokenizer, response_ids, skip_special_tokens=True)
        decoded = str(decoded)
        stripped = decoded.strip()
        if not stripped:
            label = "EMPTY"
            color = self.colors.red
        elif len(stripped) < 16:
            label = "SHORT"
            color = self.colors.yellow
        else:
            label = "FINAL"
            color = self.colors.green
        response_len = len(response_ids) if response_ids else len(decoded)
        if self.view == "sequence":
            print_line(
                self.colors,
                color,
                f"\n◆ {label} sample={data.get('sample_id')} mode={data.get('mode')} len={response_len}",
            )
            return
        if self.view == "chat":
            preview = decoded if len(decoded) <= self.max_final_chars else decoded[: self.max_final_chars - 3] + "..."
            print_line(
                self.colors,
                color,
                f"\n◆ {label} sample={data.get('sample_id')} mode={data.get('mode')} len={response_len}",
            )
            print_line(self.colors, color, visible_text(preview))
            return
        print_line(
            self.colors,
            color,
            (
                f"{label} sample_id={data.get('sample_id')} mode={data.get('mode')} "
                f"response_len={response_len} text='{compact_text(decoded)}'"
            ),
        )

    def handle_status(self, record, kind):
        data = payload(record)
        req = short_request_id(request_id(record))
        if kind == "probe_start":
            config_parts = []
            for key in (
                "max_prompt",
                "max_response",
                "chunk_size",
                "max_chunks",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
                "student_stream",
            ):
                if key in data:
                    config_parts.append(f"{key}={data.get(key)}")
            config_text = ""
            if config_parts:
                config_text = "\n" + " ".join(config_parts)
            text = (
                f"PROBE_START mode={data.get('mode')} prompts={data.get('prompt_count')} "
                f"student={data.get('student_model')} teacher={data.get('teacher_model')}"
            )
            if config_parts:
                text += " " + " ".join(config_parts)
            if self.view in {"chat", "sequence"}:
                text = (
                    f"SKD LIVE PROBE mode={data.get('mode')} prompts={data.get('prompt_count')}\n"
                    f"student={data.get('student_model')}\n"
                    f"teacher={data.get('teacher_model')}"
                    f"{config_text}"
                )
            print_line(self.colors, self.colors.blue, text)
        elif kind == "probe_finish":
            text = f"PROBE_FINISH mode={data.get('mode')}"
            if self.view in {"chat", "sequence"}:
                text = f"\nSKD LIVE PROBE FINISHED mode={data.get('mode')}"
            print_line(self.colors, self.colors.blue, text)
        elif kind == "student_generate_start":
            if self.view in {"chat", "sequence"}:
                return
            print_line(
                self.colors,
                self.colors.dim,
                (
                    f"STUDENT_START req={req} prompt_len={data.get('prompt_len')} "
                    f"has_images={data.get('has_images')} has_videos={data.get('has_videos')}"
                ),
            )
        elif kind == "boundary_result":
            print_line(
                self.colors,
                self.colors.blue,
                f"BOUNDARY req={req} sample_id={data.get('sample_id')} result_type={data.get('result_type')}",
            )
        elif kind == "mixed_current_start":
            if self.view in {"chat", "sequence"}:
                print_line(
                    self.colors,
                    self.colors.blue,
                    (
                        f"\nMIXED carryover={data.get('carryover_sample_id')} "
                        f"fresh={data.get('fresh_sample_ids')}"
                    ),
                )
                return
            print_line(
                self.colors,
                self.colors.blue,
                (
                    f"MIXED_START carryover={data.get('carryover_sample_id')} "
                    f"fresh={data.get('fresh_sample_ids')}"
                ),
            )
        elif kind == "mixed_current_partial":
            if self.view in {"chat", "sequence"}:
                print_line(self.colors, self.colors.blue, f"\nPARTIAL sample={data.get('sample_id')} req={req}")
                return
            print_line(
                self.colors,
                self.colors.blue,
                f"MIXED_PARTIAL req={req} sample_id={data.get('sample_id')}",
            )
        elif kind == "mixed_current_completed_without_partial":
            print_line(
                self.colors,
                self.colors.blue,
                f"MIXED_COMPLETED_WITHOUT_PARTIAL sample_id={data.get('sample_id')}",
            )
        elif kind == "mixed_current_finish":
            text = "MIXED_FINISH"
            if self.view == "chat":
                text = "\nMIXED FINISH"
            print_line(self.colors, self.colors.blue, text)


def stream_lines(path, follow, poll_interval):
    while follow and not path.exists():
        time.sleep(poll_interval)
    with path.open("r", encoding="utf-8") as handle:
        line_no = 0
        while True:
            pos = handle.tell()
            line = handle.readline()
            if line:
                if follow and not line.endswith("\n"):
                    handle.seek(pos)
                    time.sleep(poll_interval)
                    continue
                line_no += 1
                yield line_no, line
                continue
            if not follow:
                break
            time.sleep(poll_interval)


def main():
    args = parse_args()
    use_color = (not args.no_color) and (args.force_color or sys.stdout.isatty())
    colors = Colors(use_color)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    viewer = StreamProbeViewer(
        tokenizer=tokenizer,
        colors=colors,
        verify_top_k=args.verify_top_k,
        show_top_k=args.show_top_k,
        max_token_width=args.max_token_width,
        view=args.view,
        token_delay=args.token_delay,
        max_final_chars=args.max_final_chars,
    )
    try:
        for line_no, line in stream_lines(Path(args.probe_log), args.follow, args.poll_interval):
            viewer.process_line(line, line_no)
    except KeyboardInterrupt:
        print_line(colors, colors.dim, "INTERRUPTED")


if __name__ == "__main__":
    main()

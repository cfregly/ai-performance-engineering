"""OOF-042: a wrapper's time_regex is the only timing pattern the parser uses.

parse_kernel_time_ms used to fall back to the last time-like value in stdout
when a custom regex failed to compile, did not match, or captured something
that was not a number. That value can belong to a kernel the wrapper does not
declare. The parser now raises ValueError instead.

Because a mismatch now fails the run, one test audits every wrapper under
code/ that passes time_regex. It renders the stdout printf calls of the
wrapper's CUDA source and checks that exactly one printed line matches, that
the regex captures the whole printed number, and that the number is a
millisecond variable. These checks read source text, so a printf inside a
loop or behind a runtime branch still needs the B200 run to confirm.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from core.benchmark.timing_parser import parse_kernel_time_ms

CODE_ROOT = Path(__file__).resolve().parents[1]
TIME_MS_REGEX = r"TIME_MS:\s*([0-9.]+)"


# ---------------------------------------------------------------------------
# Helpers that read printf calls from CUDA sources. test_audit_fix_o13_w2011.py
# imports them to build realistic binary output.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrintfCall:
    """One printf call that writes to stdout, with its format unescaped."""

    format: str
    arguments: tuple[str, ...]


_PRINTF_START = re.compile(r"(?<![\w.>:])((?:std::)?f?printf)\s*\(")
_STRING_LITERALS = re.compile(r'\s*(?:"(?:[^"\\]|\\.)*"\s*)+')
_C_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'"}
_CONVERSION = re.compile(
    r"%(?P<flags>[-+ #0]*)(?P<width>\d+|\*)?(?:\.(?P<precision>\d+|\*))?"
    r"(?P<length>hh|h|ll|l|z|j|t|L)?(?P<conversion>[a-zA-Z%])"
)


def strip_c_comments(source: str) -> str:
    """Remove // and /* */ comments, leaving string and character literals intact."""
    pieces: list[str] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char in "\"'":
            end = index + 1
            while end < len(source) and source[end] != char:
                end += 2 if source[end] == "\\" else 1
            pieces.append(source[index : end + 1])
            index = end + 1
        elif source.startswith("//", index):
            newline = source.find("\n", index)
            index = len(source) if newline < 0 else newline
        elif source.startswith("/*", index):
            close = source.find("*/", index + 2)
            assert close >= 0, "unterminated block comment"
            pieces.append(" ")
            index = close + 2
        else:
            pieces.append(char)
            index += 1
    return "".join(pieces)


def _call_arguments(source: str, open_paren: int) -> list[str]:
    """Split the argument list that starts at source[open_paren] at top-level commas."""
    arguments: list[str] = []
    depth = 0
    start = open_paren + 1
    index = open_paren
    while index < len(source):
        char = source[index]
        if char in "\"'":
            end = index + 1
            while source[end] != char:
                end += 2 if source[end] == "\\" else 1
            index = end
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                arguments.append(source[start:index].strip())
                return arguments
        elif char == "," and depth == 1:
            arguments.append(source[start:index].strip())
            start = index + 1
        index += 1
    raise AssertionError(f"unterminated call at offset {open_paren}")


def _unescape_c(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        escaped = match.group(1)
        if escaped not in _C_ESCAPES:
            raise ValueError(f"unsupported C escape \\{escaped}")
        return _C_ESCAPES[escaped]

    return re.sub(r"\\(.)", replace, text)


def stdout_printf_calls(source: str) -> list[PrintfCall]:
    """Return the printf calls in a C++ source that write to stdout, in order.

    fprintf(stderr, ...) calls are skipped. So are calls whose format is not
    made only of string literals, because they cannot be rendered here.
    """
    code = strip_c_comments(source)
    calls: list[PrintfCall] = []
    for match in _PRINTF_START.finditer(code):
        arguments = _call_arguments(code, match.end() - 1)
        if match.group(1).endswith("fprintf"):
            if arguments[0] != "stdout":
                continue
            arguments = arguments[1:]
        if not _STRING_LITERALS.fullmatch(arguments[0]):
            continue
        literal = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', arguments[0]))
        calls.append(PrintfCall(_unescape_c(literal), tuple(arguments[1:])))
    return calls


def render_printf(call: PrintfCall, value_for: Callable[[int, str], object]) -> str:
    """Render a printf call. value_for(argument_index, conversion) supplies each value."""
    used = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal used
        conversion = match.group("conversion")
        if conversion == "%":
            return "%"
        if "*" in (match.group("width") or "") + (match.group("precision") or ""):
            raise ValueError(f"* width or precision is not supported in {call.format!r}")
        spec = "%" + match.group("flags") + (match.group("width") or "")
        if match.group("precision") is not None:
            spec += "." + match.group("precision")
        value = value_for(used, conversion)
        used += 1
        if conversion in "diu":
            return (spec + "d") % int(value)
        if conversion in "fFeEgG":
            return (spec + conversion) % float(value)
        if conversion == "s":
            return (spec + "s") % str(value)
        raise ValueError(f"unsupported printf conversion %{conversion} in {call.format!r}")

    rendered = _CONVERSION.sub(replace, call.format)
    if used != len(call.arguments):
        raise ValueError(
            f"{call.format!r} has {used} conversions but {len(call.arguments)} arguments"
        )
    return rendered


# ---------------------------------------------------------------------------
# Parser contract
# ---------------------------------------------------------------------------


def test_custom_regex_that_does_not_match_raises_instead_of_returning_the_last_time() -> None:
    stdout = "Pipeline copy: 0.071 ms\nTMA copy: 0.029 ms\n"
    # The default patterns pick the last printed time.
    assert parse_kernel_time_ms(stdout) == pytest.approx(0.029)

    with pytest.raises(ValueError, match="did not match the binary output"):
        parse_kernel_time_ms(stdout, TIME_MS_REGEX)


def test_custom_regex_that_does_not_compile_raises() -> None:
    with pytest.raises(ValueError, match="does not compile"):
        parse_kernel_time_ms("TIME_MS: 1.0\n", r"TIME_MS:\s*([0-9.]+")


@pytest.mark.parametrize(
    ("regex", "groups"),
    [("", 0), (r"TIME_MS:\s*[0-9.]+", 0), (r"(TIME_MS):\s*([0-9.]+)", 2)],
)
def test_custom_regex_needs_exactly_one_capture_group(regex: str, groups: int) -> None:
    with pytest.raises(ValueError, match=rf"exactly one capture group .* has {groups}\."):
        parse_kernel_time_ms("TIME_MS: 1.0\nKernel: 2.5 ms\n", regex)


@pytest.mark.parametrize(
    ("stdout", "regex", "captured"),
    [
        ("TIME_MS: .\n", TIME_MS_REGEX, "."),
        ("TIME_MS: 1.2.3\n", TIME_MS_REGEX, "1.2.3"),
        ("TIME_MS: fast\n", r"TIME_MS:\s*(\S+)", "fast"),
    ],
)
def test_custom_regex_capture_that_is_not_a_number_raises(
    stdout: str, regex: str, captured: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(f"captured {captured!r}, which is not a number")):
        parse_kernel_time_ms(stdout + "Kernel: 2.5 ms\n", regex)


@pytest.mark.parametrize("printed", ["nan", "inf", "1e999"])
def test_custom_regex_capture_that_is_not_finite_raises(printed: str) -> None:
    with pytest.raises(ValueError, match="which is not a finite number"):
        parse_kernel_time_ms(f"TIME_MS: {printed}\nKernel: 2.5 ms\n", r"TIME_MS:\s*(\S+)")


def test_custom_regex_group_that_did_not_take_part_raises() -> None:
    with pytest.raises(ValueError, match="did not take part in the match"):
        parse_kernel_time_ms(
            "TIME_MS: n/a\nKernel: 2.5 ms\n", r"TIME_MS:\s*(?:([0-9.]+)|n/a)"
        )


def test_custom_regex_returns_its_first_match() -> None:
    stdout = "Kernel: 9.0 ms\nTIME_MS: 0.029123\nTIME_MS: 5.0\n"

    assert parse_kernel_time_ms(stdout, TIME_MS_REGEX) == 0.029123


# ---------------------------------------------------------------------------
# Every time_regex wrapper against its binary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeRegexWrapper:
    path: Path
    binary_name: str
    time_regex: str


def _time_regex_wrappers() -> list[TimeRegexWrapper]:
    wrappers: list[TimeRegexWrapper] = []
    for path in sorted(CODE_ROOT.rglob("*.py")):
        relative = path.relative_to(CODE_ROOT)
        if relative.parts[0] == "tests" or "__pycache__" in relative.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "time_regex" not in text:
            continue
        for node in ast.walk(ast.parse(text, filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            regex_node = keywords.get("time_regex")
            if regex_node is None:
                continue
            if isinstance(regex_node, ast.Constant) and regex_node.value is None:
                continue  # None selects the default patterns.
            location = f"{relative}:{node.lineno}"
            assert isinstance(regex_node, ast.Constant) and isinstance(regex_node.value, str), (
                f"{location}: time_regex must be a string literal so this audit can check it"
            )
            binary_node = keywords.get("binary_name")
            assert isinstance(binary_node, ast.Constant) and isinstance(binary_node.value, str), (
                f"{location}: time_regex needs a literal binary_name in the same call"
            )
            wrappers.append(TimeRegexWrapper(relative, binary_node.value, regex_node.value))
    return wrappers


def _audit_value(index: int, conversion: str) -> object:
    # A distinct value per argument shows which argument the regex captured.
    if conversion in "fFeEgG":
        return index + 0.123456789
    if conversion in "diu":
        return index
    return f"argument{index}"


_NUMBER_TOKEN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _audit_problem(label: str, source_path: Path, time_regex: str) -> str | None:
    """Return why time_regex does not fit the binary built from source_path, or None."""
    if not source_path.is_file():
        return f"{label}: missing CUDA source {source_path}"
    calls = stdout_printf_calls(source_path.read_text(encoding="utf-8"))
    rendered = [render_printf(call, _audit_value) for call in calls]
    matching = [index for index, text in enumerate(rendered) if re.search(time_regex, text)]
    if len(matching) != 1:
        formats = [calls[index].format for index in matching]
        return (
            f"{label}: {time_regex!r} matches {len(matching)} stdout printf calls in "
            f"{source_path.name}, expected exactly 1: {formats}"
        )
    call = calls[matching[0]]
    line = rendered[matching[0]]
    captured = re.search(time_regex, line).group(1)
    if captured not in _NUMBER_TOKEN.findall(line):
        return f"{label}: {time_regex!r} captures {captured!r}, only part of a number in {line!r}"
    expression = call.arguments[int(float(captured))]
    if not re.search(r"ms\b", expression):
        return (
            f"{label}: {time_regex!r} captures argument {expression!r} of {call.format!r}, "
            "which does not name a millisecond value"
        )
    parsed = parse_kernel_time_ms("".join(rendered), time_regex)
    if parsed != float(captured):
        return f"{label}: parsed {parsed} from the full output, expected {captured}"
    return None


def test_every_time_regex_wrapper_matches_exactly_one_time_line_of_its_binary() -> None:
    wrappers = _time_regex_wrappers()
    audited = {str(wrapper.path) for wrapper in wrappers}
    for expected in (
        "ch07/baseline_tma_copy.py",
        "ch07/optimized_tma_copy.py",
        "ch08/baseline_occupancy_tuning.py",
        "ch12/optimized_dynamic_parallelism_device.py",
    ):
        assert expected in audited, f"{expected} was not audited. Found: {sorted(audited)}"

    problems = [
        problem
        for wrapper in wrappers
        if (
            problem := _audit_problem(
                str(wrapper.path),
                CODE_ROOT / wrapper.path.parent / f"{wrapper.binary_name}.cu",
                wrapper.time_regex,
            )
        )
    ]

    assert not problems, "\n".join(problems)


def test_audit_rejects_regexes_that_miss_or_truncate_the_printed_time(tmp_path: Path) -> None:
    # Positive controls for the audit itself.
    source = tmp_path / "probe.cu"
    source.write_text(
        "int main() {\n"
        '    // std::printf("TIME_MS: %.6f\\n", avg_ms);\n'
        '    std::printf("Probe kernel: %.3f ms\\n", avg_ms);\n'
        '    std::printf("TIME_MS: %.6e\\n", avg_ms);\n'
        '    std::fprintf(stderr, "Elapsed_ms: %.6f\\n", avg_ms);\n'
        '    std::printf("checksum=%.6f\\n", checksum);\n'
        "    return 0;\n"
        "}\n",
        encoding="utf-8",
    )

    def problem(regex: str) -> str | None:
        return _audit_problem("probe", source, regex)

    # [0-9.]+ stops before the exponent that %.6e prints.
    assert "only part of a number" in problem(TIME_MS_REGEX)
    # Comments and stderr lines are not binary output.
    assert "matches 0 stdout printf calls" in problem(r"Elapsed_ms:\s*([0-9.]+)")
    assert "matches 2 stdout printf calls" in problem(r":\s*([0-9.]+)")
    assert "does not name a millisecond value" in problem(r"checksum=([0-9.]+)")
    assert problem(r"TIME_MS:\s*([0-9.eE+-]+)") is None

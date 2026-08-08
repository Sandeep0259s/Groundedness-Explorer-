import pandas as pd
import pytest

from src.rag.structured_qa import (
    StructuredQAError,
    _extract_expression,
    _run_sandboxed,
    _safe_eval,
    answer_structured_question,
)


@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "Product": ["Widget A", "Widget B", "Gadget X"],
        "Category": ["Hardware", "Hardware", "Electronics"],
        "Price": [19.99, 29.99, 49.99],
        "UnitsSold": [120, 80, 45],
    })


def test_safe_eval_computes_aggregation(sales_df):
    assert _safe_eval("df['Price'].sum()", sales_df) == pytest.approx(99.97)


def test_safe_eval_computes_filtered_aggregation(sales_df):
    result = _safe_eval("df[df['Category'] == 'Hardware']['UnitsSold'].sum()", sales_df)
    assert result == 200


@pytest.mark.parametrize("malicious", [
    "__import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "exec('import os')",
    "globals()",
])
def test_safe_eval_rejects_dangerous_code(sales_df, malicious):
    with pytest.raises(StructuredQAError):
        _safe_eval(malicious, sales_df)


def test_safe_eval_rejects_empty_code(sales_df):
    with pytest.raises(StructuredQAError):
        _safe_eval("", sales_df)


def test_extract_expression_strips_markdown_fence():
    raw = "```python\ndf['Price'].sum()\n```"
    assert _extract_expression(raw) == "df['Price'].sum()"


def test_extract_expression_keeps_only_first_line():
    raw = "df['Price'].sum()\nprint('extra')"
    assert _extract_expression(raw) == "df['Price'].sum()"


def test_answer_structured_question_returns_none_on_bad_file(tmp_path):
    bad_path = tmp_path / "not_a_real.csv"
    bad_path.write_bytes(b"\x00\x01\x02 not valid csv data at all {{{")
    # Even if this somehow parses as a single-column CSV, the point is it
    # must never raise out of answer_structured_question — always None or a string.
    result = answer_structured_question(bad_path, "what is the total?")
    assert result is None or isinstance(result, str)


def test_run_sandboxed_computes_correctly(sales_df):
    assert _run_sandboxed("df['Price'].sum()", sales_df) == pytest.approx(99.97)


def test_run_sandboxed_still_rejects_dangerous_code(sales_df):
    with pytest.raises(StructuredQAError):
        _run_sandboxed("__import__('os').system('echo pwned')", sales_df)


def test_run_sandboxed_times_out(sales_df):
    # A real computation always takes longer than 0.001s to even spawn and
    # import pandas in the child process — this forces the timeout path
    # deterministically without needing a genuinely hung expression (a
    # single eval() expression can't contain a while loop anyway).
    with pytest.raises(StructuredQAError, match="timed out"):
        _run_sandboxed("df['Price'].sum()", sales_df, timeout=0.001)


def test_structured_qa_end_to_end(tmp_path, ollama_available):
    if not ollama_available:
        pytest.skip("Ollama is not running — skipping tests that need real LLM generation")

    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "Product,Category,Price,UnitsSold\n"
        "Widget A,Hardware,19.99,120\n"
        "Widget B,Hardware,29.99,80\n"
        "Gadget X,Electronics,49.99,45\n"
    )

    answer = answer_structured_question(csv_path, "What is the total price of all products?")
    assert answer is not None
    assert "99.97" in answer

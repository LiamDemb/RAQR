"""Oracle batch skip logic: empty Graph answers (e.g. NO_CONTEXT) still count as completed."""

from raqr.generation.batch_orchestrator import _oracle_row_has_merged_strategies


def test_empty_graph_counts_as_merged():
    row = {
        "question_id": "q1",
        "pred_dense": "some answer",
        "pred_graph": "",
    }
    assert _oracle_row_has_merged_strategies(row) is True


def test_empty_dense_and_graph_count_as_merged():
    row = {
        "question_id": "q2",
        "pred_dense": "",
        "pred_graph": "",
    }
    assert _oracle_row_has_merged_strategies(row) is True


def test_both_nonempty_still_merged():
    row = {
        "question_id": "q3",
        "pred_dense": "a",
        "pred_graph": "b",
    }
    assert _oracle_row_has_merged_strategies(row) is True


def test_missing_graph_key_not_merged():
    row = {
        "question_id": "q4",
        "pred_dense": "only dense",
    }
    assert _oracle_row_has_merged_strategies(row) is False


def test_no_question_id_not_merged():
    row = {"pred_dense": "a", "pred_graph": ""}
    assert _oracle_row_has_merged_strategies(row) is False


def test_json_null_graph_counts_as_merged():
    row = {
        "question_id": "q5",
        "pred_dense": "answer",
        "pred_graph": None,
    }
    assert _oracle_row_has_merged_strategies(row) is True

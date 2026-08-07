import pytest

from src.rag import labels


def test_default_labels_created():
    labels.ensure_default_labels()
    names = {label["name"] for label in labels.list_labels()}
    assert labels.DEFAULT_LABEL in names
    assert labels.SESSION_LABEL in names


def test_session_label_is_ephemeral():
    labels.ensure_default_labels()
    assert labels.SESSION_LABEL in labels.ephemeral_label_names()


def test_create_and_delete_custom_label():
    labels.create_label("resume")
    assert labels.label_exists("resume")

    labels.delete_label("resume")
    assert not labels.label_exists("resume")


def test_general_label_cannot_be_deleted():
    labels.ensure_default_labels()
    with pytest.raises(ValueError):
        labels.delete_label(labels.DEFAULT_LABEL)


@pytest.mark.parametrize("bad_name", ["", "   ", "has spaces", "has/slash", "has.dot"])
def test_invalid_label_names_rejected(bad_name):
    with pytest.raises(ValueError):
        labels.create_label(bad_name)


def test_clear_label_contents_empties_folder_but_keeps_label():
    labels.create_label("scratch")
    scratch_dir = labels.label_dir("scratch")
    (scratch_dir / "file.txt").write_text("temporary")
    assert any(scratch_dir.iterdir())

    labels.clear_label_contents("scratch")
    assert labels.label_exists("scratch")
    assert not any(scratch_dir.iterdir())

    labels.delete_label("scratch")

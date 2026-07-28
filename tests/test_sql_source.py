import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
pytestmark = pytest.mark.sql

from xwalk.sources.sql import sql_source  # noqa: E402


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = sqlalchemy.create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                "CREATE TABLE compounds (id TEXT, label TEXT, synonyms TEXT, tax INTEGER)"
            )
        )
        conn.execute(
            sqlalchemy.text("INSERT INTO compounds VALUES (:i, :l, :s, :t)"),
            [
                {"i": "T1", "l": "glucose", "s": "dextrose|grape sugar", "t": 9606},
                {"i": "T2", "l": "fructose", "s": "fruit sugar", "t": 9606},
                {"i": "T3", "l": "sucrose", "s": "", "t": 10090},
            ],
        )
    return url


def test_yields_one_record_per_row(db):
    records = list(sql_source(db, "SELECT * FROM compounds", id_column="id"))
    assert [r.id for r in records] == ["T1", "T2", "T3"]


def test_drops_the_id_column_from_fields(db):
    record = next(iter(sql_source(db, "SELECT * FROM compounds", id_column="id")))
    assert "id" not in record.fields and record.fields["label"] == "glucose"


def test_splits_multivalue_columns(db):
    record = next(
        iter(
            sql_source(
                db, "SELECT * FROM compounds", id_column="id", multivalue_columns=["synonyms"]
            )
        )
    )
    assert record.fields["synonyms"] == ["dextrose", "grape sugar"]


def test_a_blank_multivalue_cell_becomes_an_empty_list(db):
    records = list(
        sql_source(db, "SELECT * FROM compounds", id_column="id", multivalue_columns=["synonyms"])
    )
    assert records[2].fields["synonyms"] == []


def test_bound_parameters_are_supported(db):
    records = list(
        sql_source(
            db, "SELECT * FROM compounds WHERE tax = :tax", id_column="id", params={"tax": 10090}
        )
    )
    assert [r.id for r in records] == ["T3"]


def test_a_missing_id_column_is_a_clear_error(db):
    with pytest.raises(ValueError, match="id_column"):
        list(sql_source(db, "SELECT label FROM compounds", id_column="id"))


def test_a_null_id_is_rejected(db):
    engine = sqlalchemy.create_engine(db)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("INSERT INTO compounds VALUES (NULL, 'x', '', 1)"))
    with pytest.raises(ValueError, match="blank"):
        list(sql_source(db, "SELECT * FROM compounds", id_column="id"))


def test_a_numeric_id_is_coerced_to_string(db):
    records = list(sql_source(db, "SELECT tax AS id, label FROM compounds", id_column="id"))
    assert records[0].id == "9606"


def test_the_source_is_lazy(db):
    import types

    assert isinstance(
        sql_source(db, "SELECT * FROM compounds", id_column="id"), types.GeneratorType
    )


def test_the_connection_is_closed_when_the_source_is_exhausted(db):
    """A generator that leaks a warehouse connection per call is worse than useless."""
    records = sql_source(db, "SELECT * FROM compounds", id_column="id")
    list(records)
    engine = sqlalchemy.create_engine(db)
    with engine.connect() as conn:  # would block or error on an exhausted pool
        assert conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM compounds")).scalar() == 3

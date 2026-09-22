"""The four demo queries: their SQL, model, inputs, and how they are drawn.

Every query runs on Quail Server. The SQL is the BigQuery dialect the
docs use. Selectivity hints on the BIO-4 query are the quail-bench
estimates for its predicates; the IMDB hints are the ones the
repository's IMDB demo uses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

DEVICE = "h100-sxm"
QWEN3_4B = "qwen3-4b-fp8"
RERANKER = "qwen3-reranker-0.6b-bf16"
GEMMA = "diffusion-gemma-26b-a4b-fp8"

# one H100 per model; the order is the order the page lists them
MODELS = (QWEN3_4B, RERANKER, GEMMA)

# the reaction pairing prompt of BIO-4, asked by both joins
REACTION_PROMPT = (
    "Does the medical report in DOCUMENT {0} describe the reaction in "
    "DOCUMENT {1} as something the patient experienced?"
)
SERIOUS_PROMPT = (
    "Judge strictly from the report above whether it describes a serious "
    "or life-threatening adverse event.\\n\\n{0}\\n\\nInstruction: answer "
    "TRUE if the report describes a serious or life-threatening adverse "
    "event, FALSE otherwise."
)
NEUROLOGICAL_PROMPT = (
    "Is this reaction neurological, affecting the nervous system? {0}"
)
CARDIOVASCULAR_PROMPT = (
    "Is this reaction cardiovascular, affecting the heart or blood vessels? {0}"
)

# quail-bench selectivity estimates for the BIO-4 predicates, at the
# benchmark's labeled sample: 319 of 500 serious reports, 505 and 394 of
# 1,127 terms, 19,144 of 563,500 report-term pairs
SERIOUS_SELECTIVITY = round(319 / 500, 4)
NEUROLOGICAL_SELECTIVITY = round(505 / 1127, 4)
CARDIOVASCULAR_SELECTIVITY = round(394 / 1127, 4)
REACTION_SELECTIVITY = round(19144 / 563500, 4)

IMDB_SENTIMENT_SQL = """\
SELECT r.review_id
FROM reviews AS r
WHERE AI.SCORE(
    PROMPT('Did the reviewer enjoy the movie? Would they recommend it?\\n\\n{0}',
           r.review)
) >= 0.5
"""

IMDB_ENDING_SQL = """\
SELECT r.review_id
FROM reviews AS r
WHERE AI.IF(
    PROMPT('Does this review discuss the ending of the movie?\\n\\n{0}',
           r.review),
    {'selectivity': 0.25}
)
AND AI.IF(
    PROMPT('Does the reviewer recommend watching the movie?\\n\\n{0}',
           r.review),
    {'selectivity': 0.5}
)
"""

BIO4_SQL = f"""\
SELECT r.id, n.id, c.id
FROM reports AS r
JOIN terms AS n
  ON AI.IF(
    PROMPT('{REACTION_PROMPT}', r.report, n.term),
    {{'selectivity': {REACTION_SELECTIVITY}, 'anchor': 'r'}}
  )
JOIN terms AS c
  ON AI.IF(
    PROMPT('{REACTION_PROMPT}', r.report, c.term),
    {{'selectivity': {REACTION_SELECTIVITY}, 'anchor': 'r'}}
  )
WHERE AI.IF(
    PROMPT('{SERIOUS_PROMPT}', r.report),
    {{'selectivity': {SERIOUS_SELECTIVITY}}}
)
AND AI.IF(
    PROMPT('{NEUROLOGICAL_PROMPT}', n.term),
    {{'selectivity': {NEUROLOGICAL_SELECTIVITY}}}
)
AND AI.IF(
    PROMPT('{CARDIOVASCULAR_PROMPT}', c.term),
    {{'selectivity': {CARDIOVASCULAR_SELECTIVITY}}}
)
"""

COMPACTION_PROMPT = (
    "Using the compaction state in DOCUMENT {0}, evaluate whether the "
    "retention statement in DOCUMENT {1} is true."
)

COMPACTION_SQL = f"""\
SELECT c.id, q.id, q.tool_call_id, q.kind
FROM conversations AS c
JOIN tool_questions AS q
  ON c.id = q.conversation_id
 AND AI.IF(
    PROMPT('{COMPACTION_PROMPT}', c.state, q.statement),
    {{'anchor': 'c'}}
 )
"""


@dataclass(frozen=True)
class Table:
    """One input table of a demo, as the server sees it."""

    name: str          # the name the SQL uses
    group: str         # the data group whose Arrow file holds it
    id_col: str
    text_col: str      # the document column the prompts read
    columns: tuple[str, ...]


@dataclass(frozen=True)
class Demo:
    """One playground query."""

    key: str
    title: str
    group: str         # the data group: "imdb", "bio", or "compaction"
    model: str
    sql: str
    tables: tuple[Table, ...]
    view: str          # how the page draws it: filter, score, join, compaction
    note: str          # one line under the SQL
    timeout_s: float = 1800.0
    dialect: str = "bq"
    hints: dict = field(default_factory=dict)

    def table(self, name: str) -> Table:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(name)

    def public(self) -> dict:
        """The fields the page needs, as JSON."""
        data = asdict(self)
        data["tables"] = [asdict(table) for table in self.tables]
        return data


REVIEWS = Table("reviews", "imdb", "review_id", "review",
                ("review_id", "review"))
REPORTS = Table("reports", "bio", "id", "report", ("id", "report"))
TERMS = Table("terms", "bio", "id", "term", ("id", "term"))
CONVERSATIONS = Table("conversations", "compaction", "id", "state",
                      ("id", "state"))
TOOL_QUESTIONS = Table(
    "tool_questions", "compaction", "id", "statement",
    ("id", "conversation_id", "key", "tool_call_id", "kind", "statement"))

DEMOS = (
    Demo(
        key="imdb-sentiment",
        title="IMDB · sentiment",
        group="imdb",
        model=RERANKER,
        sql=IMDB_SENTIMENT_SQL,
        tables=(REVIEWS,),
        view="score",
        note=("One reranker score per review. The question comes before "
              "the review, so its KV is computed once and read for every "
              "later review. A score of 0.5 or more counts as yes."),
    ),
    Demo(
        key="imdb-ending",
        title="IMDB · ending + recommends",
        group="imdb",
        model=QWEN3_4B,
        sql=IMDB_ENDING_SQL,
        tables=(REVIEWS,),
        view="filter",
        note=("Two questions on each review. The second is asked only "
              "when the first passes, and it reads the review from KV."),
        hints={"stages": ["discusses the ending", "recommends the movie"]},
    ),
    Demo(
        key="bio-4",
        title="BIO-4 · serious reports with two reactions",
        group="bio",
        model=QWEN3_4B,
        sql=BIO4_SQL,
        tables=(REPORTS, TERMS),
        view="join",
        note=("The report is the anchor of both joins. The corpus is the "
              "quail-bench BIO tables at scale factor 0.1: more reports "
              "than one H100 keeps in KV, so anchors are evicted and "
              "computed again. Selectivity hints are the benchmark "
              "estimates."),
        hints={"filters": {"r": "serious adverse event",
                           "n": "neurological reaction",
                           "c": "cardiovascular reaction"},
               "joins": {"n": "neurological", "c": "cardiovascular"}},
    ),
    Demo(
        key="compaction",
        title="Agent trace compaction",
        group="compaction",
        model=GEMMA,
        sql=COMPACTION_SQL,
        tables=(CONVERSATIONS, TOOL_QUESTIONS),
        view="compaction",
        note=("One retention question per tool call and per tool result, "
              "answered against the compaction state of its conversation. "
              "The state is the anchor, so every question of a "
              "conversation reads it from KV."),
    ),
)

DEMOS_BY_KEY = {demo.key: demo for demo in DEMOS}
GROUPS = tuple(dict.fromkeys(demo.group for demo in DEMOS))


def demo(key: str) -> Demo:
    try:
        return DEMOS_BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown demo {key!r}; one of {list(DEMOS_BY_KEY)}") \
            from None


def tables_of(group: str) -> dict[str, Table]:
    """Every table of a data group, by name, across the demos."""
    tables = {}
    for item in DEMOS:
        for table in item.tables:
            if table.group == group:
                tables.setdefault(table.name, table)
    return tables

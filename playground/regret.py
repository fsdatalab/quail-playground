"""KV regret and requested input tokens of a finished query, on the CPU.

The numbers come from quail-bench: ``quail_b.minimum.token_metrics``
counts a run's requested input tokens, the fewest input tokens the same
requests need with unlimited KV, and the regret (fresh tokens the
engine reported minus that minimum). The prompt pieces around each
document come from ``quail.bench.quailb.prompt_pieces``, read off the
compiled query. This module only describes the demo query to
quail-bench and hands it the saved answer tables with row indices
turned into ids. Nothing is tracked while the query runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from quail.bench.quailb import prompt_pieces
from quail_b.minimum import DocumentTokens, token_metrics
from quail_b.scoring import RunOutput
from quail_b.substrait import _Filter, _Join, _PlanInfo, _Relation


def filter_id(alias: str, position: int) -> str:
    return f"filter:{alias}:{position}"


def join_id(position: int) -> str:
    return f"join:{position}"


class _OperatorIds:
    """What ``prompt_pieces`` reads off a benchmark plan: operator ids."""

    filter_id = staticmethod(filter_id)
    join_id = staticmethod(join_id)


@dataclass(frozen=True)
class QueryDescription:
    """A demo query described the way quail-bench describes a benchmark query.

    Attributes:
        info: The relations and operators, keyed by the ids above.
        pieces: The prompt token ids around each document.
        anchors: Join written position -> the anchor alias.

    """

    info: _PlanInfo
    pieces: dict
    anchors: dict

    @property
    def _info(self) -> _PlanInfo:
        # token_metrics reads the plan info from spec._info
        return self.info


def describe_query(query, anchors: dict, tokenizer=None) -> QueryDescription:
    """Describe a compiled local query to quail-bench.

    Args:
        query: A ``Query`` from ``Session.sql``; its prompts carry the
            token ids of their fixed text.
        anchors: Join written position -> the anchor alias the engine
            chose, from the saved join answer tables.
        tokenizer: Callable(text) -> token ids for the reranker layout,
            which a compiled AI.SCORE prompt does not tokenize.

    """
    from quail.logical.nodes import model_call

    operators = query.logical.operators()
    relations = {}
    for prompt in operators.prompts:
        for argument in prompt.args:
            relations.setdefault(argument.alias, _Relation(
                argument.alias, argument.provider, argument.column))
    filters = [_Filter(filter_id(alias, position), alias,
                       predicate.prompt.template)
               for alias, predicates in operators.filters.items()
               for position, predicate in enumerate(predicates)]
    joins = [_Join(join_id(position), tuple(dict.fromkeys(
                 argument.alias for argument in join.prompt.args)),
                   join.prompt.template)
             for position, join in enumerate(operators.joins)]
    info = _PlanInfo(relations=tuple(relations.values()),
                     operators=(*filters, *joins), select=())
    pieces = prompt_pieces(query, _OperatorIds(), anchors)
    scored = [(alias, position, predicate)
              for alias, predicates in operators.filters.items()
              for position, predicate in enumerate(predicates)
              if model_call(predicate.expression).kind == "score"]
    if scored:
        pieces = _reranker_pieces(pieces, scored, tokenizer)
    return QueryDescription(info=info, pieces=pieces, anchors=dict(anchors))


def _reranker_pieces(pieces: dict, scored: list, tokenizer) -> dict:
    """Fill in the reranker's fixed text, which the compiled prompt leaves out.

    The reranker holds the document in its own field of a fixed layout:
    the text before the field is the preamble of every scored row, and
    the text after it is the tail.
    """
    from quail.logical.nodes import model_call
    from quail.planner.reranker import _query_template
    from quail.reranker import render_qwen3_reranker_input

    if tokenizer is None:
        raise ValueError("a tokenizer is needed for AI.SCORE prompt pieces")
    tails = {}
    preamble = None
    for alias, position, predicate in scored:
        rendered = render_qwen3_reranker_input(
            _query_template(model_call(predicate.expression).prompt),
            "{document}")
        before, after = rendered.split("{document}")
        preamble = preamble or list(tokenizer(before))
        tails[filter_id(alias, position)] = list(tokenizer(after))
    return {
        **pieces,
        "preamble": pieces["preamble"] or preamble,
        "filters": [{**item, "tail": tails.get(item["id"], item["tail"])}
                    for item in pieces["filters"]],
    }


def _ids(table: pa.Table, id_col: str) -> pa.Array:
    return pc.cast(table.column(id_col), pa.string()).combine_chunks()


def run_output(description: QueryDescription, report: dict,
               filter_tables: dict, join_tables: dict,
               tables: dict, id_cols: dict) -> RunOutput:
    """Turn the server's saved answer tables into a quail-bench run output.

    Args:
        description: The query description.
        report: The saved execution report; ``fresh_tokens`` is read.
        filter_tables: (alias, written position) -> the saved filter
            answer table, whose alias column holds row indices.
        join_tables: written position -> the saved join answer table.
        tables: Table name -> the input table.
        id_cols: Table name -> its id column.

    """
    ids = {relation.alias: _ids(tables[relation.table], id_cols[relation.table])
           for relation in description.info.relations}

    def id_column(alias, indices):
        return pc.take(ids[alias], pc.cast(indices, pa.int64()))

    filter_answers = {}
    for (alias, position), table in filter_tables.items():
        answer = (table.column("answer") if "answer" in table.column_names
                  else pa.array([True] * table.num_rows, pa.bool_()))
        filter_answers[filter_id(alias, position)] = pa.table({
            alias: id_column(alias, table.column(alias)), "answer": answer})
    join_answers = {}
    for position, table in join_tables.items():
        aliases = next(join.relations for join in description.info.joins
                       if join.id == join_id(position))
        join_answers[join_id(position)] = pa.table({
            **{alias: id_column(alias, table.column(alias)) for alias in aliases},
            "answer": table.column("answer")})
    return RunOutput(filter_answers, join_answers, rows=None,
                     runtime_s=report.get("wall_s"),
                     measurements={"fresh_tokens": report.get("fresh_tokens")},
                     prompt_pieces=description.pieces)


def corpus_rows(tables: dict, id_cols: dict) -> dict:
    """The input tables with each id column renamed to ``id`` for quail-bench."""
    rows = {}
    for name, table in tables.items():
        id_col = id_cols[name]
        if id_col != "id":
            table = table.rename_columns(
                ["id" if column == id_col else column
                 for column in table.column_names])
        rows[name] = table
    return rows


def token_numbers(description: QueryDescription, output: RunOutput,
                  tables: dict, id_cols: dict, tokenizer=None,
                  documents: DocumentTokens | None = None) -> dict:
    """quail-bench's input, fresh, minimum, and regret token counts.

    Args:
        description: The query description.
        output: The run output from ``run_output``.
        tables: Table name -> the input table.
        id_cols: Table name -> its id column.
        tokenizer: Callable(list of texts) -> token id lists; the
            model's Hugging Face tokenizer when omitted.
        documents: A reusable ``DocumentTokens`` store. When supplied,
            documents already tokenized by an earlier query are reused.
    """
    names = {relation.table for relation in description.info.relations}
    rows = corpus_rows({name: tables[name] for name in names}, id_cols)
    stores = None
    if documents is not None:
        stores = {description.pieces["tokenizer"]: documents}
    elif tokenizer is not None:
        stores = {description.pieces["tokenizer"]:
                  DocumentTokens(rows, tokenizer)}
    try:
        return token_metrics(description, output, rows, stores)
    except ValueError as error:
        # fresh tokens below the minimum: the engine's count and the
        # benchmark's definition disagree, so no regret is reported
        if "below the minimum" not in str(error):
            raise
        return {"input_tokens": None, "fresh_tokens": output.measurements.get(
            "fresh_tokens"), "minimum_tokens": None, "regret_tokens": None,
            "note": str(error)}


def metrics(report: dict, numbers: dict, *, gpus: int,
            usd_per_hour: float) -> dict:
    """Combine the report with the token counts into the page's numbers.

    Throughput divides requested input tokens by the query's wall time.
    Tokens read from KV are requested minus fresh. The engine's cached
    token counter only counts the reranker's and vLLM's prefix hits and
    stays 0 for Quail's own filter and join KV reuse, so it stands in for
    the requested count only before the slower minimum calculation
    finishes, and only when it is above 0. Cost is wall time in hours
    times the GPU count and hourly price; model startup is excluded, as
    the wall time excludes it.
    """
    wall_s = float(report["wall_s"])
    requested = numbers.get("input_tokens")
    fresh = report.get("fresh_tokens")
    cached = report.get("cached_tokens")
    if (not isinstance(requested, int) and isinstance(fresh, int)
            and isinstance(cached, int) and cached > 0):
        requested = fresh + cached
    kv_read = (requested - fresh if isinstance(requested, int)
               and isinstance(fresh, int) else None)
    return {
        "wall_s": wall_s,
        "boot_s": report.get("boot_s"),
        "fresh_tokens": fresh,
        "cached_tokens": cached,
        "kv_read_tokens": kv_read,
        "input_tokens": requested,
        "minimum_tokens": numbers.get("minimum_tokens"),
        "regret_tokens": numbers.get("regret_tokens"),
        "note": numbers.get("note"),
        "tokens_per_second": (requested / wall_s
                              if requested and wall_s > 0 else None),
        "gpu_cost_usd": wall_s / 3600 * gpus * usd_per_hour,
        "gpus": gpus,
        "usd_per_hour": usd_per_hour,
    }

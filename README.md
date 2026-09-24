# Quail Playground

Quail Playground is a live demo of
[Quail](https://github.com/fsdatalab/quail), an AI query engine. Choose a
query, click **Run**, and watch the results arrive as Quail runs the query on
Modal.

The page reports query time, input tokens per second, fresh input tokens,
tokens read from KV, KV regret, GPU cost, and result count.

## Run the playground

You need:

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- A [Modal](https://modal.com/) account with H100 access
- A Modal secret named `quail-service-token`

The secret must contain:

- `QUAIL_SERVER_TOKEN`, which protects the model servers
- `HF_TOKEN`, with access to DiffusionGemma on Hugging Face

Install the dependencies and connect the Modal CLI:

```bash
uv sync
uv run modal setup
```

You can skip `modal setup` if the CLI is already connected to your account.

Deploy the playground:

```bash
uv run modal deploy playground/modal_app.py 2>&1 | tee deploy.log
```

The first deploy downloads and prepares the demo data. It can take several
minutes. When the deploy finishes, Modal prints the playground page URL. Open
that URL in your browser.

Warm one container per model before a demo:

```bash
uv run modal run playground/modal_app.py::warm 2>&1 | tee warm.log
```

The command starts slot zero for each model. The other three slots start
only when queries are assigned to them. Every slot keeps its inputs and
query database on the `quail-results` volume, so a later start restores its
data. A container scales down after 15 minutes without a request, so run
the command again before a scheduled demo.

## Demo queries

| Demo | Query | Data | Model |
| --- | --- | --- | --- |
| Agent trace compaction | Decide which tool calls and results to keep in a shorter agent trace | 100 OpenHands trajectories and 6,554 questions | `diffusion-gemma-26b-a4b-fp8` |
| IMDB strong feelings | Score how strongly each reviewer feels about the movie (0.1 or higher passes) | 10,000 IMDB reviews | `qwen3-reranker-0.6b-bf16` |
| IMDB ending and recommendation | Find reviews that discuss the ending and recommend the movie | 10,000 IMDB reviews | `qwen3-4b-fp8` |
| BIO | Find serious reports with both a neurological and a cardiovascular reaction | 500 reports and 1,127 reaction terms | `qwen3-4b-fp8` |

## How it runs

- The playground page runs in a CPU container on Modal.
- Each model has four Quail Server slots, each with a separate H100 container
  limit of one. At most four containers can run for one model.
- The page assigns each new query to a slot and sends its later status,
  results, and metrics requests to that same slot. Each slot stores its own
  query state and results.
- Quail streams status updates and answer batches back to the page while the
  query runs.
- The page prepares its token cache while the query runs, then uses the final
  report and saved answers to calculate the final metrics.
- The bearer token stays in the CPU service. It is not sent to the browser.

The deployment uses the Modal app `quail-playground`. It stores model files
in `quail-hf-cache`, compiled GPU kernels in `quail-kernel-cache`, and
query results in `quail-results`.

## Test changes

The tests use small tables and fake servers. They do not need a GPU.

```bash
uv run ruff check playground tests tools
uv run pytest -q
```

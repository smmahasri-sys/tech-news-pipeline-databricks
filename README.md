# Tech News Pipeline for Databricks

Distributed PySpark implementation of the technology-news and ARR pipeline. Transformations use Spark SQL expressions and broadcast joins for small reference data; source articles stay distributed and modeled outputs are written as Delta tables.

## Pipeline behavior

- Parses revenue values, ranges, currencies, and missing/undisclosed values using the existing fixed conversion rates.
- Normalizes ISO, US slash-separated, EU hyphen-separated, and common month-name dates.
- Standardizes article categories and resolves company aliases against the supplied metadata.
- Preserves raw revenue, source file, a distributed source-row locator, parse statuses, and deterministic article and ARR IDs.
- Builds `dim_company`, `fct_article`, `fct_arr_observation`, `ai_articles_enriched`, `unmatched_companies`, and `invalid_dates` as Delta tables.
- Keeps missing ARR out of the ARR fact and applies the documented AI dataset filters for 2022-2024 and ARR greater than $50 million.

## Databricks setup

Use a Databricks Runtime with Spark 3.5 or later and Unity Catalog/Delta enabled. Upload the script as a Databricks Python task or install it as a workspace file. Store the CSV and metadata JSON in a Unity Catalog Volume or another Spark-readable cloud location.

Example task parameters:

```text
--news /Volumes/main/tech_news/raw/tech_news.csv
--metadata /Volumes/main/tech_news/raw/company_metadata.json
--catalog main
--schema tech_news
```

The task creates the schema if needed and overwrites the six named Delta tables on each run. For production backfills, change the write strategy to `MERGE` by `article_key` and `observation_id` rather than overwriting.

## Local execution and tests

Install the development requirements in a Python environment with Java available for local Spark:

```bash
python -m pip install -r requirements-dev.txt
python src/pipeline.py --news ./tech_news.csv --metadata ./company_metadata.json --catalog main --schema tech_news
pytest -q
```

On Databricks, run the script as a Python task with the parameters above; the cluster supplies Spark and Delta.

## Scale and lineage notes

`source_file` records the input path's filename. `source_row_number` is generated with Spark's distributed `monotonically_increasing_id`; it is a unique locator for a run, not a physical CSV line number, and may change if input partitioning changes. If stable source offsets are required, provide them in the source data before ingestion.

The original local script materializes a dense all-pairs article similarity matrix, which has quadratic time and memory cost and is not appropriate for large datasets. This ETL writes the article text and IDs needed for a separate Databricks Vector Search index. Configure an embedding model endpoint and Vector Search index for semantic retrieval rather than computing every article-to-article pair in the batch job.

## Table grains

- `dim_company`: one row per metadata company.
- `fct_article`: one row per source article, including parse and company-resolution lineage.
- `fct_arr_observation`: one row per article with valid parsed ARR.
- `ai_articles_enriched`: qualifying AI articles with valid ARR, from 2022 through 2024, above $50 million.
- `unmatched_companies`: source companies without a metadata match.
- `invalid_dates`: articles with missing or invalid publication dates.# tech-news-pipeline-databricks
PySpark-based data engineering pipeline for technology news, optimized for Databricks execution with ARR observations, enrichment, and semantic search.

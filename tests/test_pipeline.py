import json
import sys
from pathlib import Path

import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pipeline import build_ai_articles, build_tables, clean_and_enrich_articles


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("tech-news-pipeline-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_distributed_transforms_and_fact_filters(spark, tmp_path):
    news_path = tmp_path / "news.csv"
    metadata_path = tmp_path / "company_metadata.json"
    news_path.write_text(
        "article_id,company_name,published_date,revenue,category,title,summary,url,author,word_count\n"
        'ART001,AWS,02/23/2023,"$10M - $20M",AI/ML,Range article,Summary,https://example.com/1,Author,100\n'
        "ART002,UnknownCo,23-08-2023,£100M,Artificial Intelligence,Large ARR,Summary,https://example.com/2,Author,200\n"
        "ART003,AWS,2023-12-13T00:00:00Z,Not disclosed,Cloud,Missing ARR,Summary,https://example.com/3,Author,300\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "Amazon Web Services": {
                    "founded_year": 2006,
                    "headquarters": "Seattle, WA",
                    "employee_count": 100000,
                    "industry": "Cloud Computing",
                    "is_public": False,
                    "stock_ticker": None,
                }
            }
        ),
        encoding="utf-8",
    )

    enriched, companies = clean_and_enrich_articles(spark, str(news_path), str(metadata_path))
    rows = {row.article_id: row for row in enriched.collect()}

    assert rows["ART001"].arr_usd == 15_000_000
    assert rows["ART001"].published_date_normalized.isoformat() == "2023-02-23"
    assert rows["ART001"].company_id == "Amazon Web Services"
    assert rows["ART001"].company_resolution_status == "alias"
    assert rows["ART002"].arr_usd == 127_000_000
    assert rows["ART002"].published_date_normalized.isoformat() == "2023-08-23"
    assert rows["ART002"].company_id is None
    assert rows["ART003"].arr_usd is None
    assert rows["ART003"].revenue_parse_status == "missing"

    _, _, arr_observations = build_tables(enriched, companies)
    assert {row.article_id for row in arr_observations.collect()} == {"ART001", "ART002"}
    assert {row.article_id for row in build_ai_articles(enriched).collect()} == {"ART002"}
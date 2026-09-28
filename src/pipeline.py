from __future__ import annotations

import argparse
import json
import re
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


CURRENCY_RATES = {"USD": 1.0, "EUR": 1.10, "GBP": 1.27, "JPY": 1 / 150}
MISSING_REVENUE = ("", "N/A", "NA", "NULL", "NONE", "NOT DISCLOSED", "NOT DISCLOSED.")

CATEGORY_MAP = {
    "AI & ML": "AI_ML",
    "AI/ML": "AI_ML",
    "Artificial Intelligence": "AI_ML",
    "Machine Learning": "AI_ML",
    "Cloud": "CLOUD",
    "Cloud Computing": "CLOUD",
    "Cloud Services": "CLOUD",
    "Data Analytics": "DATA_ANALYTICS",
    "Analytics": "DATA_ANALYTICS",
    "Big Data": "BIG_DATA",
    "Financial Technology": "FINTECH",
    "FinTech": "FINTECH",
    "Finance": "FINTECH",
    "Cybersecurity": "CYBERSECURITY",
    "Security": "CYBERSECURITY",
    "InfoSec": "CYBERSECURITY",
    "SaaS": "SAAS",
    "Software": "SOFTWARE",
    "Enterprise Software": "ENTERPRISE_SOFTWARE",
}

COMPANY_ALIASES = {
    "AWS": "Amazon Web Services",
    "Amazon Web Services (AWS)": "Amazon Web Services",
    "DeepMind": "Google DeepMind",
    "Google Deepmind": "Google DeepMind",
    "Meta AI Research": "Meta AI",
    "Facebook AI Research": "Meta AI",
    "Nvidia": "NVIDIA",
    "NVIDIA Corporation": "NVIDIA",
    "Open AI": "OpenAI",
    "OpenAI Inc.": "OpenAI",
    "Databricks Inc.": "Databricks",
    "Snowflake Inc.": "Snowflake",
    "CloudFlare": "Cloudflare",
    "Data Robot": "DataRobot",
    "Mongo DB": "MongoDB",
    "Palantir Technologies": "Palantir",
    "Stripe Inc.": "Stripe",
    "The Boring Company / SpaceX": "SpaceX",
    "Microsoft Azure": "Microsoft",
    "Azure": "Microsoft",
}

METADATA_SCHEMA = StructType(
    [
        StructField("company_id", StringType(), False),
        StructField("company_name", StringType(), False),
        StructField("founded_year", IntegerType(), True),
        StructField("headquarters", StringType(), True),
        StructField("employee_count", LongType(), True),
        StructField("industry", StringType(), True),
        StructField("is_public", BooleanType(), True),
        StructField("stock_ticker", StringType(), True),
    ]
)

ALIAS_SCHEMA = StructType(
    [
        StructField("alias_name", StringType(), False),
        StructField("canonical_name", StringType(), False),
        StructField("resolution_status", StringType(), False),
    ]
)


def _read_metadata(spark: SparkSession, metadata_path: str) -> tuple[DataFrame, DataFrame]:
    # Metadata is a small dimension file; read it through Spark so DBFS and cloud
    # storage URIs work without collecting source articles on the driver.
    metadata_text = "\n".join(row["value"] for row in spark.read.text(metadata_path).collect())
    metadata: dict[str, dict[str, Any]] = json.loads(metadata_text)

    company_rows = [
        {
            "company_id": name,
            "company_name": name,
            "founded_year": values.get("founded_year"),
            "headquarters": values.get("headquarters"),
            "employee_count": values.get("employee_count"),
            "industry": values.get("industry"),
            "is_public": values.get("is_public"),
            "stock_ticker": values.get("stock_ticker"),
        }
        for name, values in metadata.items()
    ]
    companies = spark.createDataFrame(company_rows, METADATA_SCHEMA)

    alias_rows = {
        name: (name, "exact")
        for name in metadata
    }
    for alias, canonical in COMPANY_ALIASES.items():
        if canonical in metadata and alias not in alias_rows:
            alias_rows[alias] = (canonical, "alias")

    aliases = spark.createDataFrame(
        [
            (alias, canonical, status)
            for alias, (canonical, status) in alias_rows.items()
        ],
        ALIAS_SCHEMA,
    )
    return companies, aliases


def _parse_revenue(df: DataFrame) -> DataFrame:
    raw = F.trim(F.coalesce(F.col("revenue").cast("string"), F.lit("")))
    upper = F.upper(raw)
    cleaned = F.trim(
        F.regexp_replace(
            F.regexp_replace(raw, ",", ""),
            r"(?i)USD|EUR|GBP|JPY|[$€£¥]",
            "",
        )
    )
    first_number = F.regexp_extract(cleaned, r"(\d+(?:\.\d+)?)", 1)
    second_number = F.regexp_extract(
        cleaned,
        r"\d+(?:\.\d+)?[^0-9]*-\s*(\d+(?:\.\d+)?)",
        1,
    )
    is_range = second_number != ""
    amount = F.when(
        is_range,
        (first_number.cast("double") + second_number.cast("double")) / F.lit(2.0),
    ).otherwise(first_number.cast("double"))

    currency_rate = (
        F.when(upper.rlike(r"EUR|€"), F.lit(CURRENCY_RATES["EUR"]))
        .when(upper.rlike(r"GBP|£"), F.lit(CURRENCY_RATES["GBP"]))
        .when(upper.rlike(r"JPY|¥"), F.lit(CURRENCY_RATES["JPY"]))
        .otherwise(F.lit(CURRENCY_RATES["USD"]))
    )
    lower = F.lower(cleaned)
    multiplier = (
        F.when(lower.contains("trillion") | lower.endswith("t"), F.lit(1_000_000_000_000))
        .when(lower.contains("billion") | lower.endswith("b"), F.lit(1_000_000_000))
        .when(lower.contains("million") | lower.endswith("m"), F.lit(1_000_000))
        .when(lower.contains("thousand") | lower.endswith("k"), F.lit(1_000))
        .otherwise(F.lit(1))
    )

    missing = upper.isin(*MISSING_REVENUE)
    invalid = first_number == ""
    return (
        df.withColumn("raw_revenue", F.col("revenue"))
        .withColumn(
            "arr_usd",
            F.when(~missing & ~invalid, F.bround(amount * multiplier * currency_rate, 0).cast("long")),
        )
        .withColumn(
            "revenue_parse_status",
            F.when(missing, F.lit("missing")).when(invalid, F.lit("invalid")).otherwise(F.lit("parsed")),
        )
        .withColumn(
            "revenue_parse_reason",
            F.when(missing, F.lit("missing_or_undisclosed"))
            .when(invalid, F.lit("no_numeric_amount"))
            .when(is_range, F.lit("range_midpoint"))
            .otherwise(F.lit("single_value")),
        )
    )


def _normalize_dates(df: DataFrame) -> DataFrame:
    raw = F.trim(F.coalesce(F.col("published_date").cast("string"), F.lit("")))
    parsed_formats = [
        F.try_to_timestamp(raw, F.lit(pattern)).cast("date")
        for pattern in (
            "M/d/yyyy",
            "d-M-yyyy",
            "d MMM yyyy",
            "d MMMM yyyy",
            "MMMM d, yyyy",
            "MMM d, yyyy",
        )
    ]
    iso_date = F.try_to_timestamp(F.substring(raw, 1, 10), F.lit("yyyy-MM-dd")).cast("date")
    normalized = F.coalesce(iso_date, *parsed_formats)
    status = (
        F.when(raw == "", F.lit("missing"))
        .when(normalized.isNotNull(), F.lit("parsed"))
        .otherwise(F.lit("invalid"))
    )
    return (
        df.withColumn("published_date_normalized", normalized)
        .withColumn("date_parse_status", status)
        .withColumn("year", F.year("published_date_normalized"))
        .withColumn(
            "quarter",
            F.when(F.col("published_date_normalized").isNotNull(), F.concat(F.lit("Q"), F.quarter("published_date_normalized"))),
        )
        .withColumn("month", F.month("published_date_normalized"))
    )


def _standardize_category(df: DataFrame, spark: SparkSession) -> DataFrame:
    mapping = spark.createDataFrame(
        [(source, standardized) for source, standardized in CATEGORY_MAP.items()],
        ["category_source", "category_value"],
    )
    joined = df.join(
        F.broadcast(mapping),
        F.trim(F.col("category")) == F.col("category_source"),
        "left",
    )
    return (
        joined.withColumn("category_parse_status", F.when(F.col("category_value").isNull(), "unmapped").otherwise("mapped"))
        .withColumn("category_standardized", F.coalesce(F.col("category_value"), F.lit("OTHER")))
        .drop("category_source", "category_value")
    )


def _enrich_companies(df: DataFrame, companies: DataFrame, aliases: DataFrame) -> DataFrame:
    resolved = df.join(
        F.broadcast(aliases),
        F.trim(F.col("company_name")) == F.col("alias_name"),
        "left",
    ).withColumnRenamed("canonical_name", "company_id")

    dimension = companies.select(
        "company_id",
        "industry",
        "founded_year",
        "headquarters",
        "employee_count",
        "is_public",
        "stock_ticker",
    ).alias("company_dimension")
    resolved = resolved.join(
        F.broadcast(dimension),
        resolved["company_id"] == dimension["company_id"],
        "left",
    ).drop(dimension["company_id"])

    return (
        resolved.withColumn("company_match_flag", F.col("company_id").isNotNull())
        .withColumn(
            "company_resolution_status",
            F.coalesce(F.col("resolution_status"), F.lit("unmatched")),
        )
        .withColumn(
            "company_age",
            F.when(
                F.col("year").isNotNull() & F.col("founded_year").isNotNull(),
                F.col("year") - F.col("founded_year"),
            ),
        )
        .withColumn(
            "company_size_category",
            F.when(F.col("employee_count").isNull(), "UNKNOWN")
            .when(F.col("employee_count") < 10_000, "SMALL")
            .when(F.col("employee_count") <= 30_000, "MEDIUM")
            .otherwise("LARGE"),
        )
        .drop("alias_name", "resolution_status")
    )


def clean_and_enrich_articles(
    spark: SparkSession,
    news_path: str,
    metadata_path: str,
) -> tuple[DataFrame, DataFrame]:
    news = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("mode", "PERMISSIVE")
        .csv(news_path)
    )
    required = {"article_id", "company_name", "published_date", "revenue", "category", "title", "summary"}
    missing_columns = sorted(required.difference(news.columns))
    if missing_columns:
        raise ValueError(f"News CSV is missing required columns: {', '.join(missing_columns)}")

    companies, aliases = _read_metadata(spark, metadata_path)
    news = (
        news.withColumn("source_file", F.element_at(F.split(F.input_file_name(), "/"), -1))
        .withColumn("source_row_number", F.monotonically_increasing_id() + F.lit(2))
        .withColumn("article_key", F.concat(F.lit("ART_"), F.substring(F.sha2("article_id", 256), 1, 16)))
    )
    enriched = _parse_revenue(news)
    enriched = _normalize_dates(enriched)
    enriched = _standardize_category(enriched, spark)
    enriched = _enrich_companies(enriched, companies, aliases)
    return enriched, companies


def build_tables(enriched: DataFrame, companies: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    dim_company = companies.withColumn(
        "company_size_category",
        F.when(F.col("employee_count").isNull(), "UNKNOWN")
        .when(F.col("employee_count") < 10_000, "SMALL")
        .when(F.col("employee_count") <= 30_000, "MEDIUM")
        .otherwise("LARGE"),
    )

    article_columns = [
        "article_key",
        "article_id",
        "source_row_number",
        "source_file",
        "title",
        "company_id",
        "company_name",
        "published_date_normalized",
        "year",
        "quarter",
        "month",
        "category",
        "category_standardized",
        "category_parse_status",
        "summary",
        "url",
        "author",
        "word_count",
        "raw_revenue",
        "arr_usd",
        "revenue_parse_status",
        "revenue_parse_reason",
        "date_parse_status",
        "company_resolution_status",
        "company_match_flag",
        "industry",
        "founded_year",
        "headquarters",
        "employee_count",
        "is_public",
        "stock_ticker",
        "company_age",
        "company_size_category",
    ]
    fct_article = enriched.select(*[name for name in article_columns if name in enriched.columns])

    fct_arr = (
        enriched.where(F.col("arr_usd").isNotNull())
        .withColumn("observation_id", F.concat(F.lit("ARR_"), F.substring(F.sha2("article_id", 256), 1, 16)))
        .select(
            "observation_id",
            "article_id",
            "article_key",
            "company_id",
            "company_name",
            "published_date_normalized",
            "year",
            "quarter",
            "month",
            "arr_usd",
            "raw_revenue",
            "revenue_parse_status",
            "source_file",
            "source_row_number",
        )
        .dropDuplicates(["observation_id"])
    )
    return dim_company, fct_article, fct_arr


def build_ai_articles(enriched: DataFrame) -> DataFrame:
    ai_filter = (F.col("category_standardized") == "AI_ML") | (F.col("industry") == "AI/ML")
    ai = enriched.where(
        ai_filter
        & F.col("year").between(2022, 2024)
        & F.col("arr_usd").isNotNull()
        & (F.col("arr_usd") > 50_000_000)
    )
    columns = [
        "article_id",
        "title",
        "company_name",
        "published_date_normalized",
        "category_standardized",
        "arr_usd",
        "summary",
        "url",
        "industry",
        "founded_year",
        "headquarters",
        "employee_count",
        "is_public",
        "stock_ticker",
        "company_age",
        "company_size_category",
    ]
    return ai.select(*columns).withColumnRenamed("published_date_normalized", "published_date").withColumnRenamed(
        "category_standardized", "category"
    )


def _validate_identifier(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _write_delta(df: DataFrame, table_name: str, catalog: str, schema: str) -> None:
    full_name = f"{catalog}.{schema}.{table_name}"
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_name)


def run_pipeline(
    spark: SparkSession,
    news_path: str,
    metadata_path: str,
    catalog: str,
    schema: str,
) -> dict[str, DataFrame]:
    catalog = _validate_identifier(catalog, "catalog")
    schema = _validate_identifier(schema, "schema")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    enriched, companies = clean_and_enrich_articles(spark, news_path, metadata_path)
    dim_company, fct_article, fct_arr = build_tables(enriched, companies)
    ai_articles = build_ai_articles(enriched)
    unmatched = enriched.where(~F.col("company_match_flag")).select(
        "article_id", "source_row_number", "company_name", "company_resolution_status"
    )
    invalid_dates = enriched.where(F.col("date_parse_status") != "parsed").select(
        "article_id", "source_row_number", "published_date", "date_parse_status"
    )

    tables = {
        "dim_company": dim_company,
        "fct_article": fct_article,
        "fct_arr_observation": fct_arr,
        "ai_articles_enriched": ai_articles,
        "unmatched_companies": unmatched,
        "invalid_dates": invalid_dates,
    }
    for name, table in tables.items():
        _write_delta(table, name, catalog, schema)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the technology news Delta tables in Databricks.")
    parser.add_argument("--news", required=True, help="CSV path, such as a Unity Catalog Volume path")
    parser.add_argument("--metadata", required=True, help="Company metadata JSON path")
    parser.add_argument("--catalog", default="main")
    parser.add_argument("--schema", default="tech_news")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("tech-news-data-pipeline").getOrCreate()
    run_pipeline(spark, args.news, args.metadata, args.catalog, args.schema)


if __name__ == "__main__":
    main()
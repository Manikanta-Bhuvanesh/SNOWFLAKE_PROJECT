import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.types import StructType, StructField, StringType, FloatType, TimestampType
import toml
from datetime import datetime


def get_snowpark_session():
    """Create and return a Snowpark session."""
    config = toml.load("config/config.toml")["snowflake"]
    return Session.builder.configs({
        "account": config["account"],
        "user": config["user"],
        "password": config["password"],
        "role": config["role"],
        "warehouse": config["warehouse"],
        "database": config["database"],
        "schema": config["schema"]
    }).create()


def upsert_to_snowflake(session: Session, df: pd.DataFrame, 
                             table_name: str = "STOCKS_DB.STOCKS_INFO.STOCKS_DATA") -> None:
    """
    Efficiently upsert large DataFrame to Snowflake using batch processing.
    
    Args:
        session: Snowpark session
        df: DataFrame to upsert
        table_name: Target table name
        batch_size: Number of records per batch
    """
    if df.empty:
        return

    # Create staging table name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_table = f"TEMP_STOCKS_STAGING_{timestamp}"
    
    try:
        # Define schema
        schema = StructType([
            StructField("SYMBOL", StringType()),
            StructField("TICKER_TIME", TimestampType()),
            StructField("CLOSE", FloatType()),
            StructField("OPEN", FloatType()),
            StructField("HIGH", FloatType()),
            StructField("LOW", FloatType()),
            StructField("VOLUME", FloatType()),
            StructField("LAST_UPDATED_AT", TimestampType())
        ])

        data_list = [
        (row.SYMBOL, row.TICKER_TIME, row.CLOSE, row.OPEN, row.HIGH, row.LOW, row.VOLUME, row.LAST_UPDATED_AT)
        for row in df.itertuples(index=False)
        ]

        # Create Snowpark DataFrame
        sp_df = session.create_dataframe(data_list, schema=schema)

        # Overwrite temp table
        sp_df.write.mode("overwrite").save_as_table(staging_table, table_type="transient")
        print('Staging Data got pushed to snowflake.')

        # Perform single MERGE operation
        merge_sql = f"""
        MERGE INTO {table_name} AS target
        USING {staging_table} AS source
        ON target.SYMBOL = source.SYMBOL AND target.TICKER_TIME = source.TICKER_TIME
        WHEN MATCHED AND (
            target.CLOSE != source.CLOSE OR
            target.OPEN != source.OPEN OR
            target.HIGH != source.HIGH OR
            target.LOW != source.LOW OR
            target.VOLUME != source.VOLUME
        ) THEN UPDATE SET
            target.CLOSE = source.CLOSE,
            target.OPEN = source.OPEN,
            target.HIGH = source.HIGH,
            target.LOW = source.LOW,
            target.VOLUME = source.VOLUME,
            target.LAST_UPDATED_AT = source.LAST_UPDATED_AT
        WHEN NOT MATCHED THEN INSERT (
            SYMBOL, TICKER_TIME, CLOSE, OPEN, HIGH, LOW, VOLUME, LAST_UPDATED_AT
        ) VALUES (
            source.SYMBOL, source.TICKER_TIME, source.CLOSE, source.OPEN,
            source.HIGH, source.LOW, source.VOLUME, source.LAST_UPDATED_AT
        )
        """
        
        session.sql(merge_sql).collect()

    except Exception as e:
        raise
    finally:
        # Clean up staging table
        try:
            session.sql(f"DROP TABLE IF EXISTS {staging_table}").collect()
        except Exception as e:
            pass
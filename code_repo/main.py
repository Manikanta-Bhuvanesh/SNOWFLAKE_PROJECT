from code_repo.fetch_data import fetch_stocks_parallel
from code_repo.load_with_snowpark import get_snowpark_session, upsert_to_snowflake
import pandas as pd


def main():
    try:
        # Read stock symbols
        stocks = pd.read_csv('source/STOCKS.csv')['Symbol'].values.tolist()
        
        # Fetch all stock data in parallel
        combined_df = fetch_stocks_parallel(stocks,full_load=False)

        if combined_df.empty:
            return
        print('All stocks data is Fetched.')
        # Connect to Snowflake
        session = get_snowpark_session()
        
        try:
            upsert_to_snowflake(session, combined_df)
            print('Data is Upserted To snowflake.')
        finally:
            session.close()
        
    except Exception as e:
        raise


if __name__ == "__main__":
    main()
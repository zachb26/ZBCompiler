/$$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$ /$$   /$$  /$$$$$$  /$$$$$$$$  /$$$$$$  /$$   /$$
| $$__  $$| $$__  $$ /$$__  $$|_____ $$ |_  $$_/| $$$ | $$ /$$__  $$|__  $$__/ /$$__  $$| $$$ | $$
| $$  \ $$| $$  \ $$| $$  \ $$     /$$/   | $$  | $$$$| $$| $$  \__/   | $$   | $$  \ $$| $$$$| $$
| $$$$$$$ | $$$$$$$/| $$$$$$$$    /$$/    | $$  | $$ $$ $$| $$ /$$$$   | $$   | $$  | $$| $$ $$ $$
| $$__  $$| $$__  $$| $$__  $$   /$$/     | $$  | $$  $$$$| $$|_  $$   | $$   | $$  | $$| $$  $$$$
| $$  \ $$| $$  \ $$| $$  | $$  /$$/      | $$  | $$\  $$$| $$  \ $$   | $$   | $$  | $$| $$\  $$$
| $$$$$$$/| $$  | $$| $$  | $$ /$$$$$$$$ /$$$$$$| $$ \  $$|  $$$$$$/   | $$   |  $$$$$$/| $$ \  $$
|_______/ |__/  |__/|__/  |__/|________/|______/|__/  \__/ \______/    |__/    \______/ |__/  \__/
Compiler
============

Starter universe:
Use [sp500_tickers.txt](c:/brazingtoncompiler/sp500_tickers.txt) with the bulk seeder to populate the shared database from a current S&P 500 constituent list.

Example:
.\.venv\Scripts\python seed_universe.py --tickers-file .\sp500_tickers.txt --preset Balanced --skip-fresh-hours 24

SQLite -> Postgres migration:
Copy your saved analyses out of the local SQLite file and into Postgres before switching the app over.

Example:
.\.venv\Scripts\python migrate_sqlite_to_postgres.py --sqlite-path .\stocks_data.db --postgres-url "postgresql://USER:PASSWORD@HOST:5432/DBNAME"

Scheduled refresh:
The app no longer refreshes stale saved analyses on page load by default. Run the refresh worker separately on a schedule instead.

Example:
.\.venv\Scripts\python refresh_saved_analyses.py --database-url "postgresql://USER:PASSWORD@HOST:5432/DBNAME" --preset Balanced --stale-after-hours 12

Optional app-load refresh:
If you still want the old behavior temporarily, set STOCK_ENGINE_RUN_STARTUP_REFRESH=1 before launching Streamlit.

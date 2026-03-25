Timezone handling matters enormously for time-based strategies. The zoneinfo module handles EST/EDT (daylight saving) transitions correctly — hardcoding -5 UTC would give wrong tags for half the year
Half-open intervals [start, end) are the correct way to partition time — a bar cannot belong to two windows simultaneously
The ATR normalisation trick is essential for comparing activity across instruments: a 30-point NQ range and a 30-pip EURUSD range mean completely different things, but a 0.2× ATR range is comparable
Session high/low capture rates are the most useful metric — a window that prints the daily extreme 30%+ of the time is worth watching, regardless of direction


Tech stack

yfinance — 1-minute and 5-minute intraday bars
pandas + numpy — time tagging, ATR, aggregation
zoneinfo — correct New York EST/EDT timezone handling
matplotlib — 4-panel dark chart with candlestick + macro overlays
pytest — 25+ unit tests

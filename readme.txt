/$$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$ /$$   /$$  /$$$$$$  /$$$$$$$$  /$$$$$$  /$$   /$$
| $$__  $$| $$__  $$ /$$__  $$|_____ $$ |_  $$_/| $$$ | $$ /$$__  $$|__  $$__/ /$$__  $$| $$$ | $$
| $$  \ $$| $$  \ $$| $$  \ $$     /$$/   | $$  | $$$$| $$| $$  \__/   | $$   | $$  \ $$| $$$$| $$
| $$$$$$$ | $$$$$$$/| $$$$$$$$    /$$/    | $$  | $$ $$ $$| $$ /$$$$   | $$   | $$  | $$| $$ $$ $$
| $$__  $$| $$__  $$| $$__  $$   /$$/     | $$  | $$  $$$$| $$|_  $$   | $$   | $$  | $$| $$  $$$$
| $$  \ $$| $$  \ $$| $$  | $$  /$$/      | $$  | $$\  $$$| $$  \ $$   | $$   | $$  | $$| $$\  $$$
| $$$$$$$/| $$  | $$| $$  | $$ /$$$$$$$$ /$$$$$$| $$ \  $$|  $$$$$$/   | $$   |  $$$$$$/| $$ \  $$
|_______/ |__/  |__/|__/  |__/|________/|______/|__/  \__/ \______/    |__/    \______/ |__/  \__/
Compiler 0.1 
============
How to use:
1. Install latest version of python and make sure path is added in windows (and restart pc if necessary)
2. Open cmd as admin and install required dependencies by pasting the command below into the terminal 

python -m pip install yfinance pandas numpy xlsxwriter openpyxl

3. Once dependencies are done installing, paste the command below into the terminal and enjoy!

python quantitative_analyst.py

=============
How it works:
=============

The Scoring System (0 to 5 Points)
Score Range Verdict
+3 or more  STRONG BUY (High conviction)
+1 to +2    BUY (Good signals)
0           HOLD (Mixed or neutral signals)
-1 to -2    SELL (Weak signals)
-3 or less  STRONG SELL (High downside risk)

The Tests and Point Assignments
The script runs four main checks, pulling data from the calculated technical indicators and the stock's fundamentals (basic financial data).

1. Relative Strength Index (RSI)
The RSI measures the speed and change of price movements. It ranges from 0 to 100.
-Oversold (Buy Signal): If RSI is less than 30, it suggests the stock has been excessively sold and may be due for a rebound.
 -Action: Score +2
-Overbought (Sell Signal): If RSI is greater than 70, it suggests the stock has been excessively bought and may be due for a pullback.
 -Action: Score -2

2. MACD (Moving Average Convergence Divergence)
The MACD is a momentum indicator showing the relationship between two moving averages. A "bullish crossover" is a strong buy signal.
-Bullish Crossover: If the MACD Line is higher than the Signal Line (meaning momentum is accelerating upward).
 -Action: Score +1

3. Long-Term Trend (SMA 200)
This checks the current price against the 200-day Simple Moving Average (SMA), which is a universally accepted proxy for the stock's long-term trend.
-Up-Trend: If the Current Close Price is greater than the SMA_200.
 -Action: Score +1
-Down-Trend: If the Current Close Price is less than the SMA_200.
 -Action: Score -1

4. Fundamental Check (P/E Ratio)
This incorporates a basic fundamental test: the Price-to-Earnings (P/E) ratio, which shows how much investors are willing to pay per dollar of earnings.
-Undervalued Signal: If the P/E Ratio is available and is less than 15. (This is a simplified, arbitrary rule for the model, as P/E standards vary widely by industry.)
 -Action: Score +1

====================
Example Walkthrough:
====================

Let's say a stock is being analyzed:
RSI = 28: (Less than 30) +2 points
MACD Line > Signal Line: (Bullish Crossover) +1 point
Price < SMA 200: (In a long-term down-trend) -1 point
P/E Ratio = 12: (Less than 15)  +1 point
Sum: 2+1-1+1=3

Since the final score is +3, the system assigns a verdict of STRONG BUY.

=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!
Disclaimer: This was developed using ai, there may be errors inherent in the math/software that i'm unaware of yet
=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!

What i plan on adding:
-Something to verify the info given
-Making this actually compile lol
-Hidden Gems list
-Outputs to one spreadsheet and the date is very apparent when reading financial info
-More valuation methods and metrics (auto or semi-operated DCF, WACC?)
-Just making this an .exe that has settings, better GUI
-Refresh button?
-Better looking sheet
-Built in assupmtions/options to compile spreadsheets off of (automatically lists different tickers based on market sector, different types of stocks based growth and value categories)
-Just adding like a batch file or something in the folder that instantly does step #2


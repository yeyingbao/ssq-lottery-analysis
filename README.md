# SSQ Lottery Analysis

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This project analyzes historical data from the Double Color Ball (SSQ) lottery（双色球彩票） in China, spanning from 2008 to 2023 (2354 periods). The goal is to statistically examine patterns, win rates, ball frequencies, and potential anomalies (e.g., prize pool effects) to detect signs of manipulation or randomness. Inspired by online discussions (e.g., on Zhihu), it tests hypotheses like the "High Prize Pool Clearance Theory" using rigorous statistical methods.

The analysis is implemented in Python and includes:
- Time series overviews of prize pools and sales.
- Win rate comparisons (actual vs. theoretical).
- Ball frequency analysis (red and blue balls) with chi-square tests.
- Effects of prize pools and sales on win rates (t-tests, Mann-Whitney U, KS tests, ANOVA).
- Multiple testing corrections (Bonferroni, FDR, Holm).
- Visualizations (charts saved as PNGs).

A companion WeChat public account article (in Chinese) summarizes the findings in a narrative style, with charts and conclusions. See [公众号推文ver2.0.pdf](公众号推文ver2.0.pdf) for the full post.

**Disclaimer**: This is for entertainment and educational purposes only. It does not provide lottery advice, and statistical significance does not imply manipulation. Lottery outcomes should be random; any patterns may arise from player behavior or other factors.

## Data Source

- Data: `ssq_data_clean_adj.csv` (cleaned SSQ lottery results from 2008-2023).
- Source: Scraped from the official China Welfare Lottery website[](https://www.cwl.gov.cn/). Note: Early data (pre-2008) was excluded due to missing values and inflation adjustments.
- Additionally, since the official website indicates that the jackpot amount for each Double Color Ball draw actually represents the remaining balance from that draw, in the `ssq_data_clean_adj.csv` file, all remaining jackpot amounts from the current draw are shifted to the next draw for analysis.Of course, I also provided the original unadjusted file, `ssq_data_clean.csv`.

## Features

- **Statistical Tests**: Chi-square for frequencies and win rates, t-tests/Mann-Whitney/KS for group comparisons, Runs Test for randomness, ANOVA for multi-group analysis, and power/effect size calculations.
- **Visualizations**: Time series, histograms, scatter plots, box/violin plots, and deviation charts.
- **Hypotheses Tested**:
  - Is the overall win rate close to theoretical (1/17,721,088)?
  - Are ball numbers uniformly distributed?
  - Does high prize pool correlate with higher win rates? (Debunking "manipulation" claims.)
  - Sales vs. prize pool correlations.
- **Outputs**: Console logs, saved PNG charts (e.g., `Analysis_1_First_Prize_Win_Rate.png`), and a CSV of multiple testing results.

## Installation

1. Clone the repository:

git clone https://github.com/yeyingbao/ssq-lottery-analysis.git

cd ssq-lottery-analysis

2. Install dependencies (Python 3.10+ required):

pip install -r requirements.txt

## Usage
Run the script with the provided CSV data:

python ssq_analysis.py

- The script loads `ssq_data_clean_adj.csv` by default.
- It performs all analyses sequentially, printing results to the console.
- Charts are saved in the current directory (e.g., `Analysis_0_Time_Series_Overview.png`).
- A summary report is printed at the end, highlighting key findings and limitations.

Customization:
- Edit `csv_file` in `__main__` to use a different data file.
- Extend the `SSQAnalyzer` class for new analyses.

Example Output Snippet:
Data loaded successfully! Total periods: 2354
Date range: 2008-01-01 to 2023-07-18
...
Analysis 1: First Prize Win Rate Analysis
Theoretical win rate: 0.0000000564
Actual average win rate: 0.0000000566
Chi-square p-value: 0.702 (No significant difference)


## Results and Insights

From the analysis:
- Win rates match theoretical expectations (p > 0.05 in most tests).
- Ball frequencies are mostly uniform, consistent with randomness.
- High prize pools correlate with higher sales, explaining apparent "anomalies" in winner counts—no strong evidence of manipulation.
- See the PDF article for a narrative walkthrough with charts.

Full results are in the console output and saved charts. For multiple testing corrections, check `Multiple_Testing_Results.csv`.

## Contributing

Contributions are welcome! Fork the repo and submit a pull request. Ideas:
- Add more tests (e.g., autocorrelation for temporal dependencies).
- Update data to include recent periods.
- Improve visualizations or add interactive plots (e.g., with Plotly).

Please follow standard Python conventions and add tests if possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Zhihu discussions on lottery fairness.
- Built with open-source libraries: Pandas, SciPy, Matplotlib, Seaborn, Statsmodels.

If you find this useful, star the repo or share your findings!



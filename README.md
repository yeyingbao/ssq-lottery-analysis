# SSQ Lottery Analysis

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This project analyzes historical data from the Double Color Ball (SSQ) lottery in China, spanning from 2008 to 2023 (2354 periods). The goal is to statistically examine patterns, win rates, ball frequencies, and potential anomalies (e.g., prize pool effects) to detect signs of manipulation or randomness. Inspired by online discussions (e.g., on Zhihu), it tests hypotheses like the "High Prize Pool Clearance Theory" using rigorous statistical methods.

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

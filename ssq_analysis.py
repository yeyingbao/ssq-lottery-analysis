"""
Double Color Ball (SSQ) Lottery Data Statistical Analysis
Objective: Detect potential signs of manipulation through statistical methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, ks_2samp
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import tt_ind_solve_power
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class SSQAnalyzer:
    def __init__(self, csv_path):
        """Initialize analyzer with multiple testing correction capabilities"""
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.results = {}
        self.p_values = []  # Store all p-values for multiple testing correction
        self.p_value_labels = []  # Store labels for each p-value
        self.alpha = 0.05  # Significance level
        print(f"Data loaded successfully! Total periods: {len(self.df)}")
        print(f"Date range: {self.df['开奖日期'].iloc[-1]} to {self.df['开奖日期'].iloc[0]}")
        
    def analysis_0_time_series_overview(self):
        """Analysis 0: Prize Pool and Sales Time Series Overview"""
        print("\n" + "="*60)
        print("Analysis 0: Prize Pool and Sales Time Series Overview")
        print("="*60)
        
        # Calculate total bets per period (needed for later analyses)
        self.df['Total Bets'] = self.df['销售额'] / 2
        
        # Basic statistics
        print(f"\nPrize Pool Statistics:")
        print(f"  Minimum: ¥{self.df['奖池（元）'].min():,.0f}")
        print(f"  Maximum: ¥{self.df['奖池（元）'].max():,.0f}")
        print(f"  Mean: ¥{self.df['奖池（元）'].mean():,.0f}")
        print(f"  Median: ¥{self.df['奖池（元）'].median():,.0f}")
        print(f"  Std deviation: ¥{self.df['奖池（元）'].std():,.0f}")
        
        print(f"\nSales Amount Statistics:")
        print(f"  Minimum: ¥{self.df['销售额'].min():,.0f}")
        print(f"  Maximum: ¥{self.df['销售额'].max():,.0f}")
        print(f"  Mean: ¥{self.df['销售额'].mean():,.0f}")
        print(f"  Median: ¥{self.df['销售额'].median():,.0f}")
        print(f"  Std deviation: ¥{self.df['销售额'].std():,.0f}")
        
        # Find top 10 periods with highest first prize winners
        top10_winners = self.df.nlargest(10, '一等奖注数')[['开奖日期', '一等奖注数', '奖池（元）', '销售额']]
        print(f"\nTop 10 Periods with Most First Prize Winners:")
        for idx, row in top10_winners.iterrows():
            print(f"  {row['开奖日期']}: {row['一等奖注数']:.0f} winners, "
                  f"Pool: ¥{row['奖池（元）']:,.0f}, Sales: ¥{row['销售额']:,.0f}")
        
        # Store top 10 indices for later use
        self.top10_winner_indices = top10_winners.index.tolist()
        
        # Visualization
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        
        # Subplot 1: Prize pool over time with top winner periods marked
        ax1 = axes[0]
        ax1.plot(range(len(self.df)), self.df['奖池（元）'] / 1e8, 
                color='blue', linewidth=1.5, alpha=0.7, label='Prize Pool')
        
        # Mark top 10 winner periods
        for idx in self.top10_winner_indices:
            pos = self.df.index.get_loc(idx)
            ax1.scatter(pos, self.df.loc[idx, '奖池（元）'] / 1e8, 
                       color='red', s=150, zorder=5, alpha=0.8)
            ax1.annotate(f"{self.df.loc[idx, '一等奖注数']:.0f} winners",
                        xy=(pos, self.df.loc[idx, '奖池（元）'] / 1e8),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax1.set_xlabel('Period Number')
        ax1.set_ylabel('Prize Pool (100 Million Yuan)')
        ax1.set_title('Prize Pool Over Time (Red dots = Top 10 highest winner periods)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Sales over time
        ax2 = axes[1]
        ax2.plot(range(len(self.df)), self.df['销售额'] / 1e8, 
                color='green', linewidth=1.5, alpha=0.7, label='Sales Amount')
        ax2.set_xlabel('Period Number')
        ax2.set_ylabel('Sales Amount (100 Million Yuan)')
        ax2.set_title('Sales Amount Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: First prize winner count over time
        ax3 = axes[2]
        ax3.bar(range(len(self.df)), self.df['一等奖注数'], 
               color='purple', alpha=0.6, width=1.0)
        # Highlight top 10 periods
        for idx in self.top10_winner_indices:
            pos = self.df.index.get_loc(idx)
            ax3.bar(pos, self.df.loc[idx, '一等奖注数'], 
                   color='red', alpha=0.8, width=1.0)
        ax3.set_xlabel('Period Number')
        ax3.set_ylabel('First Prize Winner Count')
        ax3.set_title('First Prize Winner Count Over Time (Red bars = Top 10 periods)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('Analysis_0_Time_Series_Overview.png', dpi=300, bbox_inches='tight')
        print("\nChart saved: Analysis_0_Time_Series_Overview.png")
        
        self.results['Time Series Overview'] = {
            'Mean prize pool': self.df['奖池（元）'].mean(),
            'Mean sales': self.df['销售额'].mean(),
            'Top 10 winner periods count': len(self.top10_winner_indices)
        }
        
    def analysis_1_first_prize_rate(self):
        """Analysis 1: First Prize Win Rate - Actual vs Theoretical"""
        print("\n" + "="*60)
        print("Analysis 1: First Prize Win Rate Analysis")
        print("="*60)
        
        # Calculate total bets per period (sales / 2 yuan) if not already done
        if 'Total Bets' not in self.df.columns:
            self.df['Total Bets'] = self.df['销售额'] / 2
        
        # Calculate actual win rate per period
        self.df['Actual Win Rate'] = self.df['一等奖注数'] / self.df['Total Bets']
        
        # Theoretical win rate: C(33,6) * C(16,1) = 17721088
        theoretical_prob = 1 / 17721088
        
        # Calculate average actual win rate
        avg_real_prob = self.df['Actual Win Rate'].mean()
        
        # Calculate expected first prize count
        self.df['Expected First Prize'] = self.df['Total Bets'] * theoretical_prob
        
        print(f"Theoretical win rate: {theoretical_prob:.10f} (1/{17721088:,})")
        print(f"Actual average win rate: {avg_real_prob:.10f}")
        print(f"Actual/Theoretical ratio: {avg_real_prob/theoretical_prob:.4f}")
        print(f"\nActual win rate statistics:")
        print(f"  Min: {self.df['Actual Win Rate'].min():.10f}")
        print(f"  Max: {self.df['Actual Win Rate'].max():.10f}")
        print(f"  Std: {self.df['Actual Win Rate'].std():.10f}")
        
        # Chi-square test: observed vs expected
        observed = self.df['一等奖注数'].sum()
        expected = self.df['Expected First Prize'].sum()
        
        chi2_stat = ((observed - expected) ** 2) / expected
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        # Store p-value for multiple testing correction
        self.p_values.append(p_value)
        self.p_value_labels.append("Analysis 1: Chi-square test (First prize rate)")
        
        print(f"\nChi-square test results:")
        print(f"  Observed total first prizes: {observed:.0f}")
        print(f"  Expected total first prizes: {expected:.0f}")
        print(f"  Chi-square statistic: {chi2_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Conclusion: {'Significant difference (p<0.05)' if p_value < 0.05 else 'No significant difference (p>=0.05)'}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Win rate time series
        ax1 = axes[0, 0]
        ax1.plot(range(len(self.df)), self.df['Actual Win Rate'], alpha=0.6, label='Actual Win Rate')
        ax1.axhline(y=theoretical_prob, color='r', linestyle='--', label='Theoretical Win Rate')
        ax1.axhline(y=avg_real_prob, color='g', linestyle='--', label='Average Actual Rate')
        ax1.set_xlabel('Period Number')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('First Prize Win Rate Time Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Win rate distribution histogram
        ax2 = axes[0, 1]
        ax2.hist(self.df['Actual Win Rate'], bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(x=theoretical_prob, color='r', linestyle='--', linewidth=2, label='Theoretical')
        ax2.axvline(x=avg_real_prob, color='g', linestyle='--', linewidth=2, label='Actual Average')
        ax2.set_xlabel('Win Rate')
        ax2.set_ylabel('Frequency')
        ax2.set_title('First Prize Win Rate Distribution')
        ax2.legend()
        
        # Subplot 3: Observed vs Expected
        ax3 = axes[1, 0]
        ax3.scatter(self.df['Expected First Prize'], self.df['一等奖注数'], alpha=0.5)
        max_val = max(self.df['Expected First Prize'].max(), self.df['一等奖注数'].max())
        ax3.plot([0, max_val], [0, max_val], 'r--', label='y=x')
        ax3.set_xlabel('Expected First Prize Count')
        ax3.set_ylabel('Actual First Prize Count')
        
        # Add p-value to title
        sig_star = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        ax3.set_title(f'Actual vs Expected First Prize Count\nChi-square p={p_value:.6f} {sig_star}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: First prize count distribution
        ax4 = axes[1, 1]
        bins = range(0, int(self.df['一等奖注数'].max()) + 2)
        ax4.hist(self.df['一等奖注数'], bins=bins, alpha=0.7, edgecolor='black')
        ax4.axvline(x=self.df['一等奖注数'].mean(), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.df["一等奖注数"].mean():.2f}')
        ax4.set_xlabel('First Prize Count')
        ax4.set_ylabel('Frequency')
        ax4.set_title('First Prize Count Distribution')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('Analysis_1_First_Prize_Win_Rate.png', dpi=300, bbox_inches='tight')
        print("\nChart saved: Analysis_1_First_Prize_Win_Rate.png")
        
        self.results['First Prize Analysis'] = {
            'Theoretical win rate': theoretical_prob,
            'Actual average win rate': avg_real_prob,
            'p-value': p_value
        }
        
    def analysis_2_number_frequency(self):
        """Analysis 2: Red & Blue Ball Number Frequency Analysis + Chi-square Test"""
        print("\n" + "="*60)
        print("Analysis 2: Ball Number Frequency Analysis")
        print("="*60)
        
        # Red ball analysis (1-33)
        red_balls = []
        for i in range(1, 7):
            red_balls.extend(self.df[f'红球{i}'].tolist())
        
        red_freq = pd.Series(red_balls).value_counts().sort_index()
        total_draws = len(self.df) * 6
        red_theoretical_prob = 6 / 33
        red_expected_freq = total_draws / 33
        
        print(f"\nRed Ball Analysis (1-33):")
        print(f"  Total draws: {total_draws}")
        print(f"  Theoretical probability: {red_theoretical_prob:.6f}")
        print(f"  Expected frequency per number: {red_expected_freq:.2f}")
        print(f"  Actual frequency range: {red_freq.min()} - {red_freq.max()}")
        print(f"  Mean frequency: {red_freq.mean():.2f}")
        print(f"  Std deviation: {red_freq.std():.2f}")
        
        # Red ball chi-square test
        observed_red = [red_freq.get(i, 0) for i in range(1, 34)]
        expected_red = [red_expected_freq] * 33
        chi2_red, p_red = stats.chisquare(observed_red, expected_red)
        
        # Store p-value
        self.p_values.append(p_red)
        self.p_value_labels.append("Analysis 2: Chi-square test (Red ball frequency)")
        
        print(f"\nRed Ball Chi-square Test:")
        print(f"  Chi-square statistic: {chi2_red:.4f}")
        print(f"  Degrees of freedom: 32")
        print(f"  p-value: {p_red:.6f}")
        print(f"  Conclusion: {'Significant deviation from uniform' if p_red < 0.05 else 'No significant deviation from uniform'}")
        
        # Blue ball analysis (1-16)
        blue_freq = self.df['蓝球'].value_counts().sort_index()
        total_blue_draws = len(self.df)
        blue_theoretical_prob = 1 / 16
        blue_expected_freq = total_blue_draws / 16
        
        print(f"\nBlue Ball Analysis (1-16):")
        print(f"  Total draws: {total_blue_draws}")
        print(f"  Theoretical probability: {blue_theoretical_prob:.6f}")
        print(f"  Expected frequency per number: {blue_expected_freq:.2f}")
        print(f"  Actual frequency range: {blue_freq.min()} - {blue_freq.max()}")
        print(f"  Mean frequency: {blue_freq.mean():.2f}")
        print(f"  Std deviation: {blue_freq.std():.2f}")
        
        # Blue ball chi-square test
        observed_blue = [blue_freq.get(i, 0) for i in range(1, 17)]
        expected_blue = [blue_expected_freq] * 16
        chi2_blue, p_blue = stats.chisquare(observed_blue, expected_blue)
        
        # Store p-value
        self.p_values.append(p_blue)
        self.p_value_labels.append("Analysis 2: Chi-square test (Blue ball frequency)")
        
        print(f"\nBlue Ball Chi-square Test:")
        print(f"  Chi-square statistic: {chi2_blue:.4f}")
        print(f"  Degrees of freedom: 15")
        print(f"  p-value: {p_blue:.6f}")
        print(f"  Conclusion: {'Significant deviation from uniform' if p_blue < 0.05 else 'No significant deviation from uniform'}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Red ball frequency chart
        ax1 = axes[0, 0]
        bars1 = ax1.bar(red_freq.index, red_freq.values, alpha=0.7, edgecolor='black', color='red')
        ax1.axhline(y=red_expected_freq, color='blue', linestyle='--', linewidth=2, label='Expected')
        ax1.set_xlabel('Red Ball Number')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Red Ball Number Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Red ball deviation chart
        ax2 = axes[0, 1]
        red_deviation = [(red_freq.get(i, 0) - red_expected_freq) / np.sqrt(red_expected_freq) 
                         for i in range(1, 34)]
        colors = ['red' if x < 0 else 'green' for x in red_deviation]
        ax2.bar(range(1, 34), red_deviation, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='±2σ')
        ax2.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Red Ball Number')
        ax2.set_ylabel('Standardized Deviation')
        
        # Add p-value to title
        sig_red = '***' if p_red < 0.001 else '**' if p_red < 0.01 else '*' if p_red < 0.05 else 'ns'
        ax2.set_title(f'Red Ball Frequency Standardized Deviation\nChi-square p={p_red:.6f} {sig_red}')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Blue ball frequency chart
        ax3 = axes[1, 0]
        bars3 = ax3.bar(blue_freq.index, blue_freq.values, alpha=0.7, edgecolor='black', color='blue')
        ax3.axhline(y=blue_expected_freq, color='red', linestyle='--', linewidth=2, label='Expected')
        ax3.set_xlabel('Blue Ball Number')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Blue Ball Number Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Blue ball deviation chart
        ax4 = axes[1, 1]
        blue_deviation = [(blue_freq.get(i, 0) - blue_expected_freq) / np.sqrt(blue_expected_freq) 
                          for i in range(1, 17)]
        colors = ['red' if x < 0 else 'blue' for x in blue_deviation]
        ax4.bar(range(1, 17), blue_deviation, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='±2σ')
        ax4.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Blue Ball Number')
        ax4.set_ylabel('Standardized Deviation')
        
        # Add p-value to title
        sig_blue = '***' if p_blue < 0.001 else '**' if p_blue < 0.01 else '*' if p_blue < 0.05 else 'ns'
        ax4.set_title(f'Blue Ball Frequency Standardized Deviation\nChi-square p={p_blue:.6f} {sig_blue}')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('Analysis_2_Number_Frequency.png', dpi=300, bbox_inches='tight')
        print("\nChart saved: Analysis_2_Number_Frequency.png")
        
        self.results['Number Frequency Analysis'] = {
            'Red ball chi-square p-value': p_red,
            'Blue ball chi-square p-value': p_blue
        }
        
    def analysis_3_prize_pool_effect(self):
        """Analysis 3: Prize Pool Effect on Win Rate"""
        print("\n" + "="*60)
        print("Analysis 3: Prize Pool Impact on First Prize Win Rate")
        print("="*60)
        
        # Calculate quantiles
        q25 = self.df['奖池（元）'].quantile(0.25)
        q75 = self.df['奖池（元）'].quantile(0.75)
        
        # Group data
        low_pool = self.df[self.df['奖池（元）'] <= q25].copy()
        high_pool = self.df[self.df['奖池（元）'] >= q75].copy()
        
        # Calculate win rates
        low_pool_rate = (low_pool['一等奖注数'] / low_pool['Total Bets']).dropna()
        high_pool_rate = (high_pool['一等奖注数'] / high_pool['Total Bets']).dropna()
        
        print(f"\nPrize Pool Group Statistics:")
        print(f"  Low Pool Group (≤25th percentile): ¥{q25:,.0f}")
        print(f"    Sample size: {len(low_pool)}")
        print(f"    Mean win rate: {low_pool_rate.mean():.10f}")
        print(f"    Std deviation: {low_pool_rate.std():.10f}")
        print(f"    Mean first prize count: {low_pool['一等奖注数'].mean():.2f}")
        
        print(f"\n  High Pool Group (≥75th percentile): ¥{q75:,.0f}")
        print(f"    Sample size: {len(high_pool)}")
        print(f"    Mean win rate: {high_pool_rate.mean():.10f}")
        print(f"    Std deviation: {high_pool_rate.std():.10f}")
        print(f"    Mean first prize count: {high_pool['一等奖注数'].mean():.2f}")
        
        print(f"\n  Win rate difference: {(high_pool_rate.mean() - low_pool_rate.mean()):.10f}")
        print(f"  Ratio (High/Low): {high_pool_rate.mean() / low_pool_rate.mean():.4f}")
        
        # t-test
        t_stat, t_pvalue = ttest_ind(high_pool_rate, low_pool_rate, equal_var=False)
        
        # Store p-value
        self.p_values.append(t_pvalue)
        self.p_value_labels.append("Analysis 3: t-test (Prize pool effect)")
        
        # Power analysis
        effect_size = (high_pool_rate.mean() - low_pool_rate.mean()) / np.sqrt((high_pool_rate.std()**2 + low_pool_rate.std()**2) / 2)
        try:
            power = tt_ind_solve_power(effect_size=effect_size, nobs1=len(high_pool_rate), 
                                      alpha=self.alpha, ratio=len(low_pool_rate)/len(high_pool_rate), alternative='two-sided')
        except:
            power = np.nan
        
        print(f"\nt-test (Welch's):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {t_pvalue:.6f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        print(f"  Statistical power: {power:.4f}" if not np.isnan(power) else "  Statistical power: N/A")
        print(f"  Conclusion: {'Significant difference' if t_pvalue < 0.05 else 'No significant difference'}")
        
        # Mann-Whitney U test
        u_stat, u_pvalue = mannwhitneyu(high_pool_rate, low_pool_rate, alternative='two-sided')
        
        # Store p-value
        self.p_values.append(u_pvalue)
        self.p_value_labels.append("Analysis 3: Mann-Whitney U test (Prize pool effect)")
        
        print(f"\nMann-Whitney U Test:")
        print(f"  U-statistic: {u_stat:.4f}")
        print(f"  p-value: {u_pvalue:.6f}")
        print(f"  Conclusion: {'Significant difference' if u_pvalue < 0.05 else 'No significant difference'}")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(high_pool_rate, low_pool_rate)
        
        # Store p-value
        self.p_values.append(ks_pvalue)
        self.p_value_labels.append("Analysis 3: KS test (Prize pool effect)")
        
        print(f"\nKolmogorov-Smirnov Test:")
        print(f"  KS-statistic: {ks_stat:.4f}")
        print(f"  p-value: {ks_pvalue:.6f}")
        print(f"  Conclusion: {'Distributions differ significantly' if ks_pvalue < 0.05 else 'No significant difference'}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Prize pool vs win rate scatter
        ax1 = axes[0, 0]
        ax1.scatter(self.df['奖池（元）'] / 1e8, self.df['Actual Win Rate'], alpha=0.5, s=20)
        ax1.axvline(x=q25/1e8, color='r', linestyle='--', label=f'25th: {q25/1e8:.2f}B')
        ax1.axvline(x=q75/1e8, color='g', linestyle='--', label=f'75th: {q75/1e8:.2f}B')
        ax1.set_xlabel('Prize Pool Amount (100 Million Yuan)')
        ax1.set_ylabel('First Prize Win Rate')
        ax1.set_title('Prize Pool vs Win Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Win rate distribution comparison
        ax2 = axes[0, 1]
        ax2.hist(low_pool_rate, bins=30, alpha=0.6, label='Low Pool', color='blue', edgecolor='black')
        ax2.hist(high_pool_rate, bins=30, alpha=0.6, label='High Pool', color='red', edgecolor='black')
        ax2.set_xlabel('Win Rate')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Win Rate Distribution Comparison')
        ax2.legend()
        
        # Subplot 3: Box plot
        ax3 = axes[1, 0]
        box_data = [low_pool_rate, high_pool_rate]
        bp = ax3.boxplot(box_data, labels=['Low Pool\n(≤25%)', 'High Pool\n(≥75%)'], 
                         patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax3.set_ylabel('Win Rate')
        
        # Add p-value to title
        sig_t = '***' if t_pvalue < 0.001 else '**' if t_pvalue < 0.01 else '*' if t_pvalue < 0.05 else 'ns'
        ax3.set_title(f'Win Rate by Prize Pool Group\nt-test p={t_pvalue:.6f} {sig_t}')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Subplot 4: First prize count comparison
        ax4 = axes[1, 1]
        ax4.hist(low_pool['一等奖注数'], bins=range(0, 25), alpha=0.6, 
                label='Low Pool', color='blue', edgecolor='black')
        ax4.hist(high_pool['一等奖注数'], bins=range(0, 25), alpha=0.6, 
                label='High Pool', color='red', edgecolor='black')
        ax4.set_xlabel('First Prize Count')
        ax4.set_ylabel('Frequency')
        ax4.set_title('First Prize Count Distribution')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('Analysis_3_Prize_Pool_Effect.png', dpi=300, bbox_inches='tight')
        print("\nChart saved: Analysis_3_Prize_Pool_Effect.png")
        
        self.results['Prize Pool Effect'] = {
            't-test p-value': t_pvalue,
            'Mann-Whitney p-value': u_pvalue,
            'KS test p-value': ks_pvalue
        }
        
    def analysis_4_sales_effect(self):
        """Analysis 4: Sales Amount Effect on Win Rate"""
        print("\n" + "="*60)
        print("Analysis 4: Sales Amount Impact on Win Rate")
        print("="*60)
        
        # Calculate quantiles
        q25 = self.df['销售额'].quantile(0.25)
        q75 = self.df['销售额'].quantile(0.75)
        
        # Group data
        low_sales = self.df[self.df['销售额'] <= q25].copy()
        high_sales = self.df[self.df['销售额'] >= q75].copy()
        
        # Calculate win rates
        low_sales_rate = (low_sales['一等奖注数'] / low_sales['Total Bets']).dropna()
        high_sales_rate = (high_sales['一等奖注数'] / high_sales['Total Bets']).dropna()
        
        print(f"\nSales Amount Group Statistics:")
        print(f"  Low Sales Group (≤25th percentile): ¥{q25:,.0f}")
        print(f"    Sample size: {len(low_sales)}")
        print(f"    Mean win rate: {low_sales_rate.mean():.10f}")
        print(f"    Std deviation: {low_sales_rate.std():.10f}")
        print(f"    Mean first prize count: {low_sales['一等奖注数'].mean():.2f}")
        
        print(f"\n  High Sales Group (≥75th percentile): ¥{q75:,.0f}")
        print(f"    Sample size: {len(high_sales)}")
        print(f"    Mean win rate: {high_sales_rate.mean():.10f}")
        print(f"    Std deviation: {high_sales_rate.std():.10f}")
        print(f"    Mean first prize count: {high_sales['一等奖注数'].mean():.2f}")
        
        print(f"\n  Win rate difference: {(high_sales_rate.mean() - low_sales_rate.mean()):.10f}")
        print(f"  Ratio (High/Low): {high_sales_rate.mean() / low_sales_rate.mean():.4f}")
        
        # Correlation analysis
        corr_pearson, p_pearson = stats.pearsonr(self.df['销售额'], self.df['Actual Win Rate'])
        corr_spearman, p_spearman = stats.spearmanr(self.df['销售额'], self.df['Actual Win Rate'])
        
        print(f"\nCorrelation Analysis:")
        print(f"  Pearson correlation: {corr_pearson:.6f}, p-value: {p_pearson:.6f}")
        print(f"  Spearman correlation: {corr_spearman:.6f}, p-value: {p_spearman:.6f}")
        
        # t-test
        t_stat, t_pvalue = ttest_ind(high_sales_rate, low_sales_rate, equal_var=False)
        
        # Store p-value
        self.p_values.append(t_pvalue)
        self.p_value_labels.append("Analysis 4: t-test (Sales effect)")
        
        print(f"\nt-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {t_pvalue:.6f}")
        print(f"  Conclusion: {'Significant difference' if t_pvalue < 0.05 else 'No significant difference'}")
        
        # Mann-Whitney U test
        u_stat, u_pvalue = mannwhitneyu(high_sales_rate, low_sales_rate, alternative='two-sided')
        
        # Store p-value
        self.p_values.append(u_pvalue)
        self.p_value_labels.append("Analysis 4: Mann-Whitney U test (Sales effect)")
        
        print(f"\nMann-Whitney U Test:")
        print(f"  U-statistic: {u_stat:.4f}")
        print(f"  p-value: {u_pvalue:.6f}")
        print(f"  Conclusion: {'Significant difference' if u_pvalue < 0.05 else 'No significant difference'}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Sales vs win rate scatter
        ax1 = axes[0, 0]
        ax1.scatter(self.df['销售额'] / 1e8, self.df['Actual Win Rate'], alpha=0.5, s=20)
        ax1.axvline(x=q25/1e8, color='r', linestyle='--', label=f'25th: {q25/1e8:.2f}B')
        ax1.axvline(x=q75/1e8, color='g', linestyle='--', label=f'75th: {q75/1e8:.2f}B')
        # Add trend line
        z = np.polyfit(self.df['销售额'], self.df['Actual Win Rate'], 1)
        p = np.poly1d(z)
        ax1.plot(self.df['销售额'].sort_values() / 1e8, 
                p(self.df['销售额'].sort_values()), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend (r={corr_pearson:.3f})')
        ax1.set_xlabel('Sales Amount (100 Million Yuan)')
        ax1.set_ylabel('First Prize Win Rate')
        ax1.set_title('Sales Amount vs Win Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Win rate distribution comparison
        ax2 = axes[0, 1]
        ax2.hist(low_sales_rate, bins=30, alpha=0.6, label='Low Sales', color='blue', edgecolor='black')
        ax2.hist(high_sales_rate, bins=30, alpha=0.6, label='High Sales', color='red', edgecolor='black')
        ax2.set_xlabel('Win Rate')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Win Rate Distribution Comparison')
        ax2.legend()
        
        # Subplot 3: Box plot
        ax3 = axes[1, 0]
        box_data = [low_sales_rate, high_sales_rate]
        bp = ax3.boxplot(box_data, labels=['Low Sales\n(≤25%)', 'High Sales\n(≥75%)'], 
                         patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax3.set_ylabel('Win Rate')
        
        # Add p-value to title
        sig_t = '***' if t_pvalue < 0.001 else '**' if t_pvalue < 0.01 else '*' if t_pvalue < 0.05 else 'ns'
        ax3.set_title(f'Win Rate by Sales Group\nt-test p={t_pvalue:.6f} {sig_t}')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Subplot 4: Sales vs first prize count
        ax4 = axes[1, 1]
        ax4.scatter(self.df['销售额'] / 1e8, self.df['一等奖注数'], alpha=0.5, s=20)
        ax4.set_xlabel('Sales Amount (100 Million Yuan)')
        ax4.set_ylabel('First Prize Count')
        ax4.set_title('Sales Amount vs First Prize Count')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Analysis_4_Sales_Effect.png', dpi=300, bbox_inches='tight')
        print("\nChart saved: Analysis_4_Sales_Effect.png")
        
        self.results['Sales Effect'] = {
            'Pearson correlation': corr_pearson,
            'Pearson p-value': p_pearson,
            't-test p-value': t_pvalue,
            'Mann-Whitney p-value': u_pvalue
        }
        
    def analysis_4_2_prize_sales_correlation(self):
        """Analysis 4.2: First Prize Amount vs Sales Correlation"""
        print("\n" + "="*60)
        print("Analysis 4.2: First Prize Amount vs Sales Correlation")
        print("="*60)
        
        # Filter periods with first prize winners (奖金 > 0)
        with_prize = self.df[self.df['一等奖注数'] > 0].copy()
        
        print(f"\nAnalyzing {len(with_prize)} periods with first prize winners")
        print(f"(Excluding {len(self.df) - len(with_prize)} periods with no winners)")
        
        # Extract prize amount and sales
        prize_amount = with_prize['一等奖奖金（元）']
        sales_amount = with_prize['销售额']
        
        # Correlation analysis
        corr_pearson, p_pearson = stats.pearsonr(prize_amount, sales_amount)
        corr_spearman, p_spearman = stats.spearmanr(prize_amount, sales_amount)
        
        # Store p-values
        self.p_values.append(p_pearson)
        self.p_value_labels.append("Analysis 4.2: Pearson correlation (Prize-Sales)")
        
        print(f"\nCorrelation Analysis (First Prize Amount vs Sales):")
        print(f"  Pearson correlation: {corr_pearson:.6f}")
        print(f"    p-value: {p_pearson:.6f}")
        print(f"    Conclusion: {'Significant correlation' if p_pearson < 0.05 else 'No significant correlation'}")
        
        # Store p-value
        self.p_values.append(p_spearman)
        self.p_value_labels.append("Analysis 4.2: Spearman correlation (Prize-Sales)")
        
        print(f"\n  Spearman correlation: {corr_spearman:.6f}")
        print(f"    p-value: {p_spearman:.6f}")
        print(f"    Conclusion: {'Significant correlation' if p_spearman < 0.05 else 'No significant correlation'}")
        
        # Linear regression
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(sales_amount, prize_amount)
        
        print(f"\nLinear Regression (Prize ~ Sales):")
        print(f"  Slope: {slope:.4f}")
        print(f"  Intercept: ¥{intercept:,.0f}")
        print(f"  R-squared: {r_value**2:.6f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Standard error: {std_err:.4f}")
        
        # Group by sales quantiles
        with_prize['Sales Quartile'] = pd.qcut(with_prize['销售额'], q=4, 
                                                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        
        print(f"\nPrize Amount by Sales Quartile:")
        for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
            subset = with_prize[with_prize['Sales Quartile'] == quartile]
            print(f"  {quartile}:")
            print(f"    Mean prize: ¥{subset['一等奖奖金（元）'].mean():,.0f}")
            print(f"    Median prize: ¥{subset['一等奖奖金（元）'].median():,.0f}")
            print(f"    Sample size: {len(subset)}")
        
        # ANOVA test across quartiles
        quartile_groups = [with_prize[with_prize['Sales Quartile'] == q]['一等奖奖金（元）'].values 
                          for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']]
        f_stat, anova_p = stats.f_oneway(*quartile_groups)
        
        # Store p-value
        self.p_values.append(anova_p)
        self.p_value_labels.append("Analysis 4.2: ANOVA (Prize across sales quartiles)")
        
        print(f"\nANOVA Test (Prize amount across sales quartiles):")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {anova_p:.6f}")
        print(f"  Conclusion: {'Significant difference across quartiles' if anova_p < 0.05 else 'No significant difference'}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Scatter plot with regression line
        ax1 = axes[0, 0]
        ax1.scatter(sales_amount / 1e8, prize_amount / 1e6, alpha=0.5, s=30)
        # Add regression line
        x_line = np.array([sales_amount.min(), sales_amount.max()])
        y_line = slope * x_line + intercept
        ax1.plot(x_line / 1e8, y_line / 1e6, 'r--', linewidth=2, 
                label=f'y = {slope:.2f}x + {intercept/1e6:.2f}M\nR² = {r_value**2:.4f}')
        ax1.set_xlabel('Sales Amount (100 Million Yuan)')
        ax1.set_ylabel('First Prize Amount (Million Yuan)')
        ax1.set_title(f'First Prize Amount vs Sales\n(Pearson r = {corr_pearson:.4f}, p = {p_pearson:.6f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Box plot by quartile
        ax2 = axes[0, 1]
        quartile_data = [with_prize[with_prize['Sales Quartile'] == q]['一等奖奖金（元）'].values / 1e6
                        for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']]
        bp = ax2.boxplot(quartile_data, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'],
                        patch_artist=True, showmeans=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(plt.cm.RdYlGn(i / 3))
        ax2.set_xlabel('Sales Quartile')
        ax2.set_ylabel('First Prize Amount (Million Yuan)')
        
        # Add p-value to title
        sig_anova = '***' if anova_p < 0.001 else '**' if anova_p < 0.01 else '*' if anova_p < 0.05 else 'ns'
        ax2.set_title(f'First Prize Amount by Sales Quartile\nANOVA p={anova_p:.6f} {sig_anova}')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Subplot 3: Hexbin plot for density
        ax3 = axes[1, 0]
        hexbin = ax3.hexbin(sales_amount / 1e8, prize_amount / 1e6, 
                           gridsize=25, cmap='YlOrRd', mincnt=1)
        ax3.set_xlabel('Sales Amount (100 Million Yuan)')
        ax3.set_ylabel('First Prize Amount (Million Yuan)')
        ax3.set_title('Prize Amount vs Sales Density')
        plt.colorbar(hexbin, ax=ax3, label='Count')
        
        # Subplot 4: Residual plot
        ax4 = axes[1, 1]
        predicted = slope * sales_amount + intercept
        residuals = prize_amount - predicted
        ax4.scatter(predicted / 1e6, residuals / 1e6, alpha=0.5, s=30)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Predicted Prize Amount (Million Yuan)')
        ax4.set_ylabel('Residuals (Million Yuan)')
        ax4.set_title('Residual Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Analysis_4_2_Prize_Sales_Correlation.png', dpi=300, bbox_inches='tight')
        print("\nChart saved: Analysis_4_2_Prize_Sales_Correlation.png")
        
        self.results['Prize-Sales Correlation'] = {
            'Pearson correlation': corr_pearson,
            'Pearson p-value': p_pearson,
            'Spearman correlation': corr_spearman,
            'Spearman p-value': p_spearman,
            'ANOVA p-value': anova_p,
            'R-squared': r_value**2
        }
        
    def analysis_5_additional_tests(self):
        """Analysis 5: Additional Statistical Tests"""
        print("\n" + "="*60)
        print("Analysis 5: Additional Statistical Tests")
        print("="*60)
        
        # 5.1 Consecutive periods without first prize
        print("\n5.1 Consecutive No-Winner Periods Analysis")
        print("-" * 40)
        
        no_first_prize = (self.df['一等奖注数'] == 0).astype(int)
        consecutive_zeros = []
        count = 0
        for val in no_first_prize:
            if val == 1:
                count += 1
            else:
                if count > 0:
                    consecutive_zeros.append(count)
                count = 0
        if count > 0:
            consecutive_zeros.append(count)
        
        if consecutive_zeros:
            print(f"  Longest consecutive no-winner periods: {max(consecutive_zeros)}")
            print(f"  Average consecutive no-winner periods: {np.mean(consecutive_zeros):.2f}")
            print(f"  Number of occurrences: {len(consecutive_zeros)}")
        
        no_prize_rate = (self.df['一等奖注数'] == 0).sum() / len(self.df)
        print(f"  No-winner period percentage: {no_prize_rate:.4%}")
        
        # 5.2 High first prize count analysis
        print("\n5.2 High First Prize Count Analysis")
        print("-" * 40)
        
        high_winners = self.df[self.df['一等奖注数'] >= 10]
        print(f"  Periods with ≥10 first prizes: {len(high_winners)} ({len(high_winners)/len(self.df):.2%})")
        if len(high_winners) > 0:
            print(f"  Average sales in these periods: ¥{high_winners['销售额'].mean():,.0f}")
            print(f"  Average prize pool: ¥{high_winners['奖池（元）'].mean():,.0f}")
            print(f"  Maximum first prize count: {self.df['一等奖注数'].max()}")
        
        # 5.3 Runs test - detect randomness
        print("\n5.3 Runs Test (Win Rate Sequence Randomness)")
        print("-" * 40)
        
        # Binarize win rate (above median = 1, below = 0)
        median_rate = self.df['Actual Win Rate'].median()
        binary_series = (self.df['Actual Win Rate'] > median_rate).astype(int)
        
        # Calculate number of runs
        runs = 1
        for i in range(1, len(binary_series)):
            if binary_series.iloc[i] != binary_series.iloc[i-1]:
                runs += 1
        
        n1 = binary_series.sum()  # Count of 1s
        n0 = len(binary_series) - n1  # Count of 0s
        
        # Expected runs and standard deviation
        runs_expected = (2 * n0 * n1) / (n0 + n1) + 1
        runs_std = np.sqrt((2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / 
                           ((n0 + n1) ** 2 * (n0 + n1 - 1)))
        
        # Z-statistic
        z_stat = (runs - runs_expected) / runs_std
        p_runs = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Store p-value
        self.p_values.append(p_runs)
        self.p_value_labels.append("Analysis 5: Runs test (Sequence randomness)")
        
        print(f"  Actual runs: {runs}")
        print(f"  Expected runs: {runs_expected:.2f}")
        print(f"  Z-statistic: {z_stat:.4f}")
        print(f"  p-value: {p_runs:.6f}")
        print(f"  Conclusion: {'Non-random sequence (trend/pattern exists)' if p_runs < 0.05 else 'Random sequence'}")
        
        # 5.4 Autocorrelation analysis
        print("\n5.4 Autocorrelation Analysis (Time Series)")
        print("-" * 40)
        
        # Calculate autocorrelation for several lags
        lags = [1, 2, 3, 5, 10]
        print("  Autocorrelation coefficients:")
        for lag in lags:
            autocorr = self.df['Actual Win Rate'].autocorr(lag=lag)
            print(f"    lag-{lag}: {autocorr:.6f}")
        
        # 5.5 Prize money distribution analysis
        print("\n5.5 First Prize Money Distribution Analysis")
        print("-" * 40)
        
        # Filter periods with first prize winners
        with_prize = self.df[self.df['一等奖注数'] > 0].copy()
        prize_per_winner = with_prize['一等奖奖金（元）']
        
        print(f"  First prize money per winner statistics:")
        print(f"    Minimum: ¥{prize_per_winner.min():,.0f}")
        print(f"    Maximum: ¥{prize_per_winner.max():,.0f}")
        print(f"    Mean: ¥{prize_per_winner.mean():,.0f}")
        print(f"    Median: ¥{prize_per_winner.median():,.0f}")
        print(f"    Std deviation: ¥{prize_per_winner.std():,.0f}")
        
        # Check 5M and 10M cap
        cap_5m = (prize_per_winner <= 5000000).sum()
        cap_10m = ((prize_per_winner > 5000000) & (prize_per_winner == 10000000)).sum()
        
        print(f"  Prize ≤5M yuan: {cap_5m} ({cap_5m/len(prize_per_winner):.2%})")
        print(f"  Prize =10M yuan (cap): {cap_10m} ({cap_10m/len(prize_per_winner):.2%})")
        
        # Visualization
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Subplot 1: Consecutive no-winner distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if consecutive_zeros:
            ax1.hist(consecutive_zeros, bins=range(1, max(consecutive_zeros)+2), 
                    alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Consecutive No-Winner Periods')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Consecutive No-Winner Periods Distribution')
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: First prize count time series
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(range(len(self.df)), self.df['一等奖注数'], alpha=0.7)
        ax2.axhline(y=10, color='r', linestyle='--', label='≥10 winners')
        ax2.set_xlabel('Period Number')
        ax2.set_ylabel('First Prize Count')
        ax2.set_title('First Prize Count Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Autocorrelation plot
        ax3 = fig.add_subplot(gs[1, :])
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(self.df['Actual Win Rate'], ax=ax3)
        ax3.set_title('Win Rate Autocorrelation')
        ax3.set_xlabel('Lag')
        ax3.set_ylabel('Autocorrelation')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Prize money distribution
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(prize_per_winner / 1e6, bins=50, alpha=0.7, edgecolor='black')
        ax4.axvline(x=5, color='r', linestyle='--', linewidth=2, label='5M yuan')
        ax4.axvline(x=10, color='g', linestyle='--', linewidth=2, label='10M yuan (cap)')
        ax4.set_xlabel('First Prize Money (Million Yuan)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('First Prize Money Distribution')
        ax4.legend()
        
        # Subplot 5: Binary sequence
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(range(len(binary_series)), binary_series, alpha=0.7, linewidth=0.5)
        ax5.fill_between(range(len(binary_series)), binary_series, alpha=0.3)
        ax5.set_xlabel('Period Number')
        ax5.set_ylabel('Above Median (1) / Below Median (0)')
        ax5.set_title(f'Binary Win Rate Sequence (Runs: {runs}, Expected: {runs_expected:.0f})')
        ax5.grid(True, alpha=0.3)
        
        plt.savefig('Analysis_5_Additional_Tests.png', dpi=300, bbox_inches='tight')
        print("\nChart saved: Analysis_5_Additional_Tests.png")
        
        self.results['Additional Tests'] = {
            'Runs test p-value': p_runs,
            'No-winner percentage': no_prize_rate
        }
        
    def analysis_6_1_prize_pool_by_winner_count(self):
        """Analysis 6.1: Prize Pool Analysis by Winner Count Categories"""
        print("\n" + "="*60)
        print("Analysis 6.1: Prize Pool Analysis by Winner Count")
        print("="*60)
        
        # Define categories
        categories = {
            'All periods': self.df,
            'No winners (0)': self.df[self.df['一等奖注数'] == 0],
            '1-9 winners': self.df[(self.df['一等奖注数'] >= 1) & (self.df['一等奖注数'] <= 9)],
            '10-19 winners': self.df[(self.df['一等奖注数'] >= 10) & (self.df['一等奖注数'] <= 19)],
            '20-49 winners': self.df[(self.df['一等奖注数'] >= 20) & (self.df['一等奖注数'] <= 49)],
            '50+ winners': self.df[self.df['一等奖注数'] >= 50]
        }
        
        # Calculate statistics for each category
        print(f"\nPrize Pool Statistics by Winner Count Category:")
        print("-" * 60)
        
        category_stats = {}
        for cat_name, cat_data in categories.items():
            if len(cat_data) > 0:
                mean_pool = cat_data['奖池（元）'].mean()
                median_pool = cat_data['奖池（元）'].median()
                std_pool = cat_data['奖池（元）'].std()
                count = len(cat_data)
                
                category_stats[cat_name] = {
                    'mean': mean_pool,
                    'median': median_pool,
                    'std': std_pool,
                    'count': count,
                    'data': cat_data['奖池（元）'].values
                }
                
                print(f"\n{cat_name}:")
                print(f"  Sample size: {count}")
                print(f"  Mean prize pool: ¥{mean_pool:,.0f}")
                print(f"  Median prize pool: ¥{median_pool:,.0f}")
                print(f"  Std deviation: ¥{std_pool:,.0f}")
                print(f"  Min: ¥{cat_data['奖池（元）'].min():,.0f}")
                print(f"  Max: ¥{cat_data['奖池（元）'].max():,.0f}")
        
        # Statistical tests between key groups
        print("\n" + "-" * 60)
        print("Statistical Tests Between Groups:")
        print("-" * 60)
        
        # Comparison pairs
        comparison_pairs = [
            ('No winners (0)', '10-19 winners'),
            ('No winners (0)', '20-49 winners'),
            ('No winners (0)', '50+ winners'),
            ('1-9 winners', '10-19 winners'),
            ('1-9 winners', '20-49 winners'),
            ('10-19 winners', '20-49 winners'),
            ('20-49 winners', '50+ winners')
        ]
        
        test_results = []
        for cat1, cat2 in comparison_pairs:
            if cat1 in category_stats and cat2 in category_stats:
                data1 = category_stats[cat1]['data']
                data2 = category_stats[cat2]['data']
                
                if len(data1) >= 2 and len(data2) >= 2:
                    # t-test
                    t_stat, t_p = ttest_ind(data1, data2, equal_var=False)
                    
                    # Mann-Whitney U test
                    u_stat, u_p = mannwhitneyu(data1, data2, alternative='two-sided')
                    
                    mean_diff = category_stats[cat1]['mean'] - category_stats[cat2]['mean']
                    
                    print(f"\n{cat1} vs {cat2}:")
                    print(f"  Mean difference: ¥{mean_diff:,.0f}")
                    print(f"  t-test: t={t_stat:.4f}, p={t_p:.6f} {'***' if t_p < 0.001 else '**' if t_p < 0.01 else '*' if t_p < 0.05 else 'ns'}")
                    print(f"  Mann-Whitney: U={u_stat:.0f}, p={u_p:.6f} {'***' if u_p < 0.001 else '**' if u_p < 0.01 else '*' if u_p < 0.05 else 'ns'}")
                    
                    test_results.append({
                        'pair': f"{cat1} vs {cat2}",
                        't_p': t_p,
                        'u_p': u_p,
                        'mean_diff': mean_diff
                    })
        
        # ANOVA across groups with sufficient data
        print("\n" + "-" * 60)
        print("ANOVA Test (Prize Pool Across Winner Count Categories):")
        print("-" * 60)
        
        anova_groups = []
        anova_labels = []
        for cat_name in ['No winners (0)', '1-9 winners', '10-19 winners', '20-49 winners', '50+ winners']:
            if cat_name in category_stats and category_stats[cat_name]['count'] >= 2:
                anova_groups.append(category_stats[cat_name]['data'])
                anova_labels.append(cat_name)
        
        if len(anova_groups) >= 2:
            f_stat, anova_p = stats.f_oneway(*anova_groups)
            
            # Store p-value
            self.p_values.append(anova_p)
            self.p_value_labels.append("Analysis 6.1: ANOVA (Prize pool by winner count)")
            
            print(f"  Groups tested: {', '.join(anova_labels)}")
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  p-value: {anova_p:.6f}")
            print(f"  Conclusion: {'Significant difference exists' if anova_p < 0.05 else 'No significant difference'}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 13))
        
        # Subplot 1: Bar chart of mean prize pools
        ax1 = axes[0, 0]
        valid_cats = {k: v for k, v in category_stats.items() if k != 'All periods' and v['count'] > 0}
        cat_names = list(valid_cats.keys())
        means = [valid_cats[k]['mean'] / 1e8 for k in cat_names]
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cat_names)))
        
        bars = ax1.bar(range(len(cat_names)), means, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(cat_names)))
        ax1.set_xticklabels(cat_names, rotation=45, ha='right')
        ax1.set_ylabel('Mean Prize Pool (100 Million Yuan)')
        ax1.set_title('Mean Prize Pool by Winner Count Category')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}B\n(n={valid_cats[cat_names[i]]["count"]})',
                    ha='center', va='bottom', fontsize=9)
        
        # Subplot 2: Box plot
        ax2 = axes[0, 1]
        box_data = [valid_cats[k]['data'] / 1e8 for k in cat_names]
        bp = ax2.boxplot(box_data, labels=cat_names, patch_artist=True, showmeans=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i])
        ax2.set_xticklabels(cat_names, rotation=45, ha='right')
        ax2.set_ylabel('Prize Pool (100 Million Yuan)')
        
        # Add p-value to title
        sig_anova = '***' if anova_p < 0.001 else '**' if anova_p < 0.01 else '*' if anova_p < 0.05 else 'ns'
        ax2.set_title(f'Prize Pool Distribution by Winner Count\nANOVA p={anova_p:.6f} {sig_anova}')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Subplot 3: Violin plot
        ax3 = axes[1, 0]
        positions = range(len(cat_names))
        parts = ax3.violinplot(box_data, positions=positions, showmeans=True, showmedians=True)
        ax3.set_xticks(positions)
        ax3.set_xticklabels(cat_names, rotation=45, ha='right')
        ax3.set_ylabel('Prize Pool (100 Million Yuan)')
        ax3.set_title('Prize Pool Distribution (Violin Plot)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Subplot 4: Sample size comparison
        ax4 = axes[1, 1]
        counts = [valid_cats[k]['count'] for k in cat_names]
        bars = ax4.bar(range(len(cat_names)), counts, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(len(cat_names)))
        ax4.set_xticklabels(cat_names, rotation=45, ha='right')
        ax4.set_ylabel('Number of Periods')
        ax4.set_title('Sample Size by Category')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        total_periods = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            pct = count / total_periods * 100
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('Analysis_6_1_Prize_Pool_by_Winner_Count.png', dpi=300, bbox_inches='tight')
        print("\nChart saved: Analysis_6_1_Prize_Pool_by_Winner_Count.png")
        
        self.results['Prize Pool by Winner Count'] = {
            'ANOVA p-value': anova_p if len(anova_groups) >= 2 else None,
            'Test comparisons': test_results
        }
        
    def analysis_6_2_high_win_rate_prize_pool(self):
        """Analysis 6.2: Prize Pool Analysis for High Win Rate Periods"""
        print("\n" + "="*60)
        print("Analysis 6.2: Prize Pool for High Win Rate Periods")
        print("="*60)
        
        # Calculate win rate for all periods
        self.df['Actual Win Rate'] = self.df['一等奖注数'] / self.df['Total Bets']
        
        # Define percentile thresholds
        p90_threshold = self.df['Actual Win Rate'].quantile(0.90)
        p80_threshold = self.df['Actual Win Rate'].quantile(0.80)
        
        # Categorize periods
        top10_periods = self.df[self.df['Actual Win Rate'] >= p90_threshold]
        top20_periods = self.df[self.df['Actual Win Rate'] >= p80_threshold]
        bottom80_periods = self.df[self.df['Actual Win Rate'] < p80_threshold]
        bottom90_periods = self.df[self.df['Actual Win Rate'] < p90_threshold]
        
        print(f"\nWin Rate Thresholds:")
        print(f"  90th percentile: {p90_threshold:.10f}")
        print(f"  80th percentile: {p80_threshold:.10f}")
        print(f"  Mean win rate: {self.df['Actual Win Rate'].mean():.10f}")
        print(f"  Median win rate: {self.df['Actual Win Rate'].median():.10f}")
        
        # Statistics for each category
        categories = {
            'Top 10% (≥90th percentile)': top10_periods,
            'Top 20% (≥80th percentile)': top20_periods,
            'Bottom 80% (<80th percentile)': bottom80_periods,
            'Bottom 90% (<90th percentile)': bottom90_periods,
            'All periods': self.df
        }
        
        print(f"\nPrize Pool Statistics by Win Rate Category:")
        print("-" * 60)
        
        category_stats = {}
        for cat_name, cat_data in categories.items():
            mean_pool = cat_data['奖池（元）'].mean()
            median_pool = cat_data['奖池（元）'].median()
            std_pool = cat_data['奖池（元）'].std()
            count = len(cat_data)
            
            category_stats[cat_name] = {
                'mean': mean_pool,
                'median': median_pool,
                'std': std_pool,
                'count': count,
                'data': cat_data['奖池（元）'].values
            }
            
            print(f"\n{cat_name}:")
            print(f"  Sample size: {count} ({count/len(self.df)*100:.1f}%)")
            print(f"  Mean prize pool: ¥{mean_pool:,.0f}")
            print(f"  Median prize pool: ¥{median_pool:,.0f}")
            print(f"  Std deviation: ¥{std_pool:,.0f}")
        
        # Statistical comparisons
        print("\n" + "-" * 60)
        print("Statistical Tests (High Win Rate vs Others):")
        print("-" * 60)
        
        # Top 10% vs Bottom 90%
        t_stat_10, t_p_10 = ttest_ind(top10_periods['奖池（元）'], 
                                       bottom90_periods['奖池（元）'], 
                                       equal_var=False)
        u_stat_10, u_p_10 = mannwhitneyu(top10_periods['奖池（元）'], 
                                         bottom90_periods['奖池（元）'], 
                                         alternative='two-sided')
        
        # Store p-values
        self.p_values.append(t_p_10)
        self.p_value_labels.append("Analysis 6.2: t-test (Top 10% vs Bottom 90%)")
        self.p_values.append(u_p_10)
        self.p_value_labels.append("Analysis 6.2: Mann-Whitney (Top 10% vs Bottom 90%)")
        
        mean_diff_10 = top10_periods['奖池（元）'].mean() - bottom90_periods['奖池（元）'].mean()
        
        print(f"\nTop 10% vs Bottom 90%:")
        print(f"  Mean difference: ¥{mean_diff_10:,.0f}")
        print(f"  Ratio (Top/Bottom): {top10_periods['奖池（元）'].mean() / bottom90_periods['奖池（元）'].mean():.4f}")
        print(f"  t-test: t={t_stat_10:.4f}, p={t_p_10:.6f} {'***' if t_p_10 < 0.001 else '**' if t_p_10 < 0.01 else '*' if t_p_10 < 0.05 else 'ns'}")
        print(f"  Mann-Whitney: U={u_stat_10:.0f}, p={u_p_10:.6f} {'***' if u_p_10 < 0.001 else '**' if u_p_10 < 0.01 else '*' if u_p_10 < 0.05 else 'ns'}")
        
        # Top 20% vs Bottom 80%
        t_stat_20, t_p_20 = ttest_ind(top20_periods['奖池（元）'], 
                                       bottom80_periods['奖池（元）'], 
                                       equal_var=False)
        u_stat_20, u_p_20 = mannwhitneyu(top20_periods['奖池（元）'], 
                                         bottom80_periods['奖池（元）'], 
                                         alternative='two-sided')
        
        # Store p-values
        self.p_values.append(t_p_20)
        self.p_value_labels.append("Analysis 6.2: t-test (Top 20% vs Bottom 80%)")
        self.p_values.append(u_p_20)
        self.p_value_labels.append("Analysis 6.2: Mann-Whitney (Top 20% vs Bottom 80%)")
        
        mean_diff_20 = top20_periods['奖池（元）'].mean() - bottom80_periods['奖池（元）'].mean()
        
        print(f"\nTop 20% vs Bottom 80%:")
        print(f"  Mean difference: ¥{mean_diff_20:,.0f}")
        print(f"  Ratio (Top/Bottom): {top20_periods['奖池（元）'].mean() / bottom80_periods['奖池（元）'].mean():.4f}")
        print(f"  t-test: t={t_stat_20:.4f}, p={t_p_20:.6f} {'***' if t_p_20 < 0.001 else '**' if t_p_20 < 0.01 else '*' if t_p_20 < 0.05 else 'ns'}")
        print(f"  Mann-Whitney: U={u_stat_20:.0f}, p={u_p_20:.6f} {'***' if u_p_20 < 0.001 else '**' if u_p_20 < 0.01 else '*' if u_p_20 < 0.05 else 'ns'}")
        
        # Effect size (Cohen's d)
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std
        
        cohen_d_10 = cohens_d(top10_periods['奖池（元）'], bottom90_periods['奖池（元）'])
        cohen_d_20 = cohens_d(top20_periods['奖池（元）'], bottom80_periods['奖池（元）'])
        
        print(f"\nEffect Size (Cohen's d):")
        print(f"  Top 10% vs Bottom 90%: {cohen_d_10:.4f} ({'small' if abs(cohen_d_10) < 0.5 else 'medium' if abs(cohen_d_10) < 0.8 else 'large'})")
        print(f"  Top 20% vs Bottom 80%: {cohen_d_20:.4f} ({'small' if abs(cohen_d_20) < 0.5 else 'medium' if abs(cohen_d_20) < 0.8 else 'large'})")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Bar chart comparison
        ax1 = axes[0, 0]
        categories_plot = ['All\nPeriods', 'Bottom\n90%', 'Top\n10%', 'Bottom\n80%', 'Top\n20%']
        means_plot = [
            self.df['奖池（元）'].mean() / 1e8,
            bottom90_periods['奖池（元）'].mean() / 1e8,
            top10_periods['奖池（元）'].mean() / 1e8,
            bottom80_periods['奖池（元）'].mean() / 1e8,
            top20_periods['奖池（元）'].mean() / 1e8
        ]
        colors_plot = ['gray', 'lightblue', 'red', 'lightgreen', 'orange']
        
        bars = ax1.bar(categories_plot, means_plot, color=colors_plot, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Mean Prize Pool (100 Million Yuan)')
        ax1.set_title('Mean Prize Pool by Win Rate Category')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels and significance
        for i, (bar, val) in enumerate(zip(bars, means_plot)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}B',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add significance stars
        sig_10 = '***' if t_p_10 < 0.001 else '**' if t_p_10 < 0.01 else '*' if t_p_10 < 0.05 else 'ns'
        sig_20 = '***' if t_p_20 < 0.001 else '**' if t_p_20 < 0.01 else '*' if t_p_20 < 0.05 else 'ns'
        
        # Draw significance brackets
        y_max = max(means_plot) * 1.15
        ax1.plot([1, 2], [y_max, y_max], 'k-', linewidth=1.5)
        ax1.text(1.5, y_max, sig_10, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        y_max2 = max(means_plot) * 1.25
        ax1.plot([3, 4], [y_max2, y_max2], 'k-', linewidth=1.5)
        ax1.text(3.5, y_max2, sig_20, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Subplot 2: Box plot comparison
        ax2 = axes[0, 1]
        box_data = [
            bottom90_periods['奖池（元）'] / 1e8,
            top10_periods['奖池（元）'] / 1e8,
            bottom80_periods['奖池（元）'] / 1e8,
            top20_periods['奖池（元）'] / 1e8
        ]
        box_labels = ['Bottom\n90%', 'Top\n10%', 'Bottom\n80%', 'Top\n20%']
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True)
        
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][2].set_facecolor('lightgreen')
        bp['boxes'][3].set_facecolor('orange')
        
        ax2.set_ylabel('Prize Pool (100 Million Yuan)')
        ax2.set_title(f'Prize Pool Distribution by Win Rate\np(Top10% vs Bot90%)={t_p_10:.4f} {sig_10}, p(Top20% vs Bot80%)={t_p_20:.4f} {sig_20}')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Subplot 3: Win rate vs Prize pool scatter with categories colored
        ax3 = axes[1, 0]
        ax3.scatter(bottom80_periods['奖池（元）'] / 1e8, 
                   bottom80_periods['Actual Win Rate'],
                   alpha=0.5, s=30, c='lightgreen', label='Bottom 80%', edgecolors='black', linewidth=0.5)
        ax3.scatter(top20_periods['奖池（元）'] / 1e8, 
                   top20_periods['Actual Win Rate'],
                   alpha=0.7, s=50, c='orange', label='Top 20%', edgecolors='black', linewidth=0.5)
        ax3.scatter(top10_periods['奖池（元）'] / 1e8, 
                   top10_periods['Actual Win Rate'],
                   alpha=0.9, s=70, c='red', label='Top 10%', edgecolors='black', linewidth=0.5)
        
        ax3.axhline(y=p80_threshold, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='80th percentile')
        ax3.axhline(y=p90_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='90th percentile')
        
        ax3.set_xlabel('Prize Pool (100 Million Yuan)')
        ax3.set_ylabel('Win Rate')
        ax3.set_title('Win Rate vs Prize Pool (Categorized)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Violin plot
        ax4 = axes[1, 1]
        parts = ax4.violinplot(box_data, positions=[0, 1, 2, 3], 
                               showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            if i == 0:
                pc.set_facecolor('lightblue')
            elif i == 1:
                pc.set_facecolor('red')
            elif i == 2:
                pc.set_facecolor('lightgreen')
            else:
                pc.set_facecolor('orange')
            pc.set_alpha(0.7)
        
        ax4.set_xticks([0, 1, 2, 3])
        ax4.set_xticklabels(box_labels)
        ax4.set_ylabel('Prize Pool (100 Million Yuan)')
        ax4.set_title(f'Prize Pool Distribution\nCohen\'s d: {cohen_d_10:.3f} (10%), {cohen_d_20:.3f} (20%)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('Analysis_6_2_High_Win_Rate_Prize_Pool.png', dpi=300, bbox_inches='tight')
        print("\nChart saved: Analysis_6_2_High_Win_Rate_Prize_Pool.png")
        
        self.results['High Win Rate Prize Pool'] = {
            'Top 10% vs Bottom 90% t-test p-value': t_p_10,
            'Top 10% vs Bottom 90% Mann-Whitney p-value': u_p_10,
            'Top 20% vs Bottom 80% t-test p-value': t_p_20,
            'Top 20% vs Bottom 80% Mann-Whitney p-value': u_p_20,
            'Cohen\'s d (10%)': cohen_d_10,
            'Cohen\'s d (20%)': cohen_d_20
        }
        
    def perform_multiple_testing_correction(self):
        """Perform multiple testing correction on all p-values"""
        print("\n" + "="*60)
        print("MULTIPLE TESTING CORRECTION")
        print("="*60)
        
        if len(self.p_values) == 0:
            print("No p-values collected for correction.")
            return
        
        print(f"\nTotal number of statistical tests: {len(self.p_values)}")
        print(f"Significance level (α): {self.alpha}")
        
        # Bonferroni correction
        bonferroni_alpha = self.alpha / len(self.p_values)
        print(f"\nBonferroni corrected α: {bonferroni_alpha:.6f}")
        
        # FDR correction (Benjamini-Hochberg)
        reject_fdr, pvals_corrected_fdr, _, _ = multipletests(self.p_values, alpha=self.alpha, method='fdr_bh')
        
        # Bonferroni-Holm correction
        reject_holm, pvals_corrected_holm, _, _ = multipletests(self.p_values, alpha=self.alpha, method='holm')
        
        # Create results DataFrame
        correction_results = pd.DataFrame({
            'Test': self.p_value_labels,
            'Original p-value': self.p_values,
            'FDR corrected p-value': pvals_corrected_fdr,
            'Holm corrected p-value': pvals_corrected_holm,
            'Significant (α=0.05)': ['Yes' if p < 0.05 else 'No' for p in self.p_values],
            'Significant (Bonferroni)': ['Yes' if p < bonferroni_alpha else 'No' for p in self.p_values],
            'Significant (FDR)': ['Yes' if r else 'No' for r in reject_fdr],
            'Significant (Holm)': ['Yes' if r else 'No' for r in reject_holm]
        })
        
        print("\n" + "-"*120)
        print("Multiple Testing Correction Results:")
        print("-"*120)
        print(correction_results.to_string(index=False))
        
        # Summary statistics
        print("\n" + "-"*60)
        print("Summary:")
        print("-"*60)
        sig_original = sum(1 for p in self.p_values if p < 0.05)
        sig_bonferroni = sum(1 for p in self.p_values if p < bonferroni_alpha)
        sig_fdr = sum(reject_fdr)
        sig_holm = sum(reject_holm)
        
        print(f"Significant tests (uncorrected, α=0.05): {sig_original}/{len(self.p_values)} ({sig_original/len(self.p_values)*100:.1f}%)")
        print(f"Significant tests (Bonferroni): {sig_bonferroni}/{len(self.p_values)} ({sig_bonferroni/len(self.p_values)*100:.1f}%)")
        print(f"Significant tests (FDR): {sig_fdr}/{len(self.p_values)} ({sig_fdr/len(self.p_values)*100:.1f}%)")
        print(f"Significant tests (Holm): {sig_holm}/{len(self.p_values)} ({sig_holm/len(self.p_values)*100:.1f}%)")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: P-value distribution
        ax1 = axes[0, 0]
        ax1.hist(self.p_values, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.axvline(x=0.05, color='r', linestyle='--', linewidth=2, label='α = 0.05')
        ax1.axvline(x=bonferroni_alpha, color='orange', linestyle='--', linewidth=2, label=f'Bonferroni α = {bonferroni_alpha:.4f}')
        ax1.set_xlabel('P-value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of P-values Across All Tests')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: P-values sorted
        ax2 = axes[0, 1]
        sorted_indices = np.argsort(self.p_values)
        sorted_pvals = np.array(self.p_values)[sorted_indices]
        ax2.scatter(range(len(sorted_pvals)), sorted_pvals, alpha=0.7, s=50, c='blue', label='Original p-values')
        ax2.scatter(range(len(pvals_corrected_fdr)), np.sort(pvals_corrected_fdr), alpha=0.7, s=50, c='green', marker='^', label='FDR corrected')
        ax2.axhline(y=0.05, color='r', linestyle='--', linewidth=2, label='α = 0.05')
        ax2.set_xlabel('Test Rank (sorted by p-value)')
        ax2.set_ylabel('P-value')
        ax2.set_title('Sorted P-values and FDR Correction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Subplot 3: Comparison of significance
        ax3 = axes[1, 0]
        methods = ['Uncorrected', 'Bonferroni', 'FDR', 'Holm']
        counts = [sig_original, sig_bonferroni, sig_fdr, sig_holm]
        colors_bar = ['lightcoral', 'orange', 'lightgreen', 'lightblue']
        bars = ax3.bar(methods, counts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Number of Significant Tests')
        ax3.set_title('Comparison of Significance Across Methods')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(self.p_values)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 4: Test-wise comparison
        ax4 = axes[1, 1]
        test_indices = range(len(self.p_values))
        width = 0.2
        
        ax4.barh([i - 1.5*width for i in test_indices], [-np.log10(p) for p in self.p_values], 
                width, label='Original', alpha=0.7, color='lightcoral')
        ax4.barh([i - 0.5*width for i in test_indices], [-np.log10(p) for p in pvals_corrected_fdr], 
                width, label='FDR', alpha=0.7, color='lightgreen')
        ax4.barh([i + 0.5*width for i in test_indices], [-np.log10(p) for p in pvals_corrected_holm], 
                width, label='Holm', alpha=0.7, color='lightblue')
        
        ax4.axvline(x=-np.log10(0.05), color='r', linestyle='--', linewidth=2, label='α = 0.05')
        ax4.set_yticks(test_indices)
        ax4.set_yticklabels([f"Test {i+1}" for i in test_indices], fontsize=8)
        ax4.set_xlabel('-log10(p-value)')
        ax4.set_title('P-values Across All Tests (log scale)')
        ax4.legend(loc='lower right')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('Multiple_Testing_Correction.png', dpi=300, bbox_inches='tight')
        print("\nChart saved: Multiple_Testing_Correction.png")
        
        # Save results to CSV
        correction_results.to_csv('Multiple_Testing_Results.csv', index=False, encoding='utf-8-sig')
        print("Results saved: Multiple_Testing_Results.csv")
        
        self.correction_results = correction_results
        
    def generate_summary_report(self):
        """Generate Summary Report"""
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        print("\n【KEY FINDINGS】")
        print("-" * 60)
        
        # Check all p-values
        significant_findings = []
        non_significant_findings = []
        
        for analysis, metrics in self.results.items():
            print(f"\n{analysis}:")
            for metric, value in metrics.items():
                if 'p-value' in metric.lower() or 'p_value' in metric:
                    if value < 0.05:
                        significant_findings.append(f"  ✗ {metric}: {value:.6f} (significant)")
                        print(f"  ✗ {metric}: {value:.6f} (p < 0.05, significant)")
                    else:
                        non_significant_findings.append(f"  ✓ {metric}: {value:.6f} (not significant)")
                        print(f"  ✓ {metric}: {value:.6f} (p >= 0.05, not significant)")
                else:
                    print(f"  • {metric}: {value}")
        
        print("\n" + "="*60)
        print("【CONCLUSIONS】")
        print("="*60)
        
        if len(significant_findings) > 0:
            print("\nSignificant deviations found (uncorrected p < 0.05):")
            for finding in significant_findings:
                print(finding)
            print("\n⚠️  IMPORTANT NOTES:")
            print("   1. Statistical significance ≠ evidence of manipulation")
            print("   2. Multiple testing correction should be applied (see next section)")
            print("   3. Alternative explanations may exist:")
            print("      - Non-random purchasing behavior (birthday numbers, lucky numbers)")
            print("      - Group purchasing and syndicates")
            print("      - Rule changes over time")
            print("      - Data quality issues")
            print("   4. Domain expertise and additional investigation needed")
            print("   5. Correlation does not imply causation")
        
        if len(non_significant_findings) > 0:
            print("\nNo significant deviations found (p >= 0.05):")
            for finding in non_significant_findings:
                print(finding)
            print("\n✓ These aspects show behavior consistent with randomness.")
        
        print("\n" + "="*60)
        print("【LIMITATIONS】")
        print("="*60)
        print("1. Observational study - cannot establish causal relationships")
        print("2. Multiple testing increases false positive risk")
        print("3. Historical data may reflect policy/rule changes")
        print("4. Player behavior patterns not accounted for")
        print("5. Sample size variations across subgroups")
        print("6. Temporal dependencies not fully modeled")
        
        print("\n" + "="*60)
        print("Analysis Complete! All charts saved to current directory.")
        print("="*60)
        
    def run_all_analyses(self):
        """Run All Analyses"""
        print("Starting Double Color Ball Lottery Data Analysis...")
        print("="*60)
        
        self.analysis_0_time_series_overview()
        self.analysis_1_first_prize_rate()
        self.analysis_2_number_frequency()
        self.analysis_3_prize_pool_effect()
        self.analysis_4_sales_effect()
        self.analysis_4_2_prize_sales_correlation()
        self.analysis_5_additional_tests()
        self.analysis_6_1_prize_pool_by_winner_count()
        self.analysis_6_2_high_win_rate_prize_pool()
        self.perform_multiple_testing_correction()
        self.generate_summary_report()


if __name__ == "__main__":
    # Usage example
    csv_file = "ssq_data_clean_adj.csv"
    
    analyzer = SSQAnalyzer(csv_file)
    analyzer.run_all_analyses()

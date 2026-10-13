import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================================
# Parameters
# ========================================
tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD", "DOGE-USD"]
start_date = "2011-01-01"
end_date = None  # until today
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "crypto_data")
show_plots = True

# Font-size parameters
label_fontsize = 18     # increased label font size (x/y labels)
title_fontsize = 16     # title font size
tick_fontsize = 16      # ticks font size

# Try to default the global font family to Times New Roman (helps titles/ticks)
plt.rcParams["font.family"] = "Times New Roman"

os.makedirs(output_dir, exist_ok=True)

# ========================================
# Helper: set tick labels font family explicitly (robust across backends)
# ========================================
def set_tick_font(ax, fontname="Times New Roman", size=12):
    # Ensure both x and y ticks use the specified font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(size)

# ========================================
# Helper: download and process one ticker
# ========================================
def process_ticker(ticker):
    print(f"\n{'='*60}")
    print(f"Downloading {ticker} data from {start_date} to today...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")
        return None

    if data.empty:
        print(f"No data found for {ticker}. Skipping.")
        return None

    # Flatten possible MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Detect price column
    if 'Close' in data.columns:
        price_col = 'Close'
    elif 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    else:
        print(f"No valid price column found for {ticker}. Columns: {data.columns}")
        return None

    # Compute log returns
    data['LogReturn'] = np.log(data[price_col] / data[price_col].shift(1))
    data = data.dropna(subset=['LogReturn'])
    data['t'] = np.arange(1, len(data) + 1)

    # Save outputs (CSV and TXT)
    csv_path = os.path.join(output_dir, f"{ticker}_prices.csv")
    txt_path = os.path.join(output_dir, f"{ticker}_log_returns.txt")

    data.to_csv(csv_path)
    with open(txt_path, "w") as f:
        f.write("t\tLogReturn\tDate\n")
        for t_val, r, date in zip(data['t'], data['LogReturn'], data.index):
            f.write(f"{t_val}\t{r:.8f}\t{date.strftime('%Y-%m-%d')}\n")

    print(f"\nSaved:")
    print(f" - {csv_path}")
    print(f" - {txt_path}")
    print(f"Total observations: {len(data)}")

    # Plot: time-series of log returns (save + optional show)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data.index, data['LogReturn'], color='black', linewidth=0.8)  # black line
    ax.set_title(f"{ticker} Logarithmic Returns (Daily)", fontname="Times New Roman", fontsize=title_fontsize)

    # Axis labels: Times New Roman, italic, larger font
    ax.set_xlabel("", fontname="Times New Roman", fontstyle="italic", fontsize=label_fontsize)
    ax.set_ylabel("Log Return", fontname="Times New Roman", fontstyle="italic", fontsize=label_fontsize)

    # Ticks: Times New Roman + size
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    set_tick_font(ax, fontname="Times New Roman", size=tick_fontsize)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, f"{ticker}_log_returns.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f" - {plot_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Optional: histogram (Times New Roman for title/ticks, italic labels)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(data['LogReturn'], bins=80, alpha=0.75, edgecolor='black')
    ax2.set_title(f"Histogram of {ticker} Logarithmic Returns", fontname="Times New Roman", fontsize=title_fontsize)
    ax2.set_xlabel("Log Return", fontname="Times New Roman", fontstyle="italic", fontsize=label_fontsize)
    ax2.set_ylabel("Frequency", fontname="Times New Roman", fontstyle="italic", fontsize=label_fontsize)
    ax2.tick_params(axis='both', labelsize=tick_fontsize)
    set_tick_font(ax2, fontname="Times New Roman", size=tick_fontsize)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    if show_plots:
        plt.show()
    else:
        plt.close(fig2)

    return data

# ========================================
# Main loop
# ========================================
all_data = {}
for ticker in tickers:
    data = process_ticker(ticker)
    if data is not None:
        all_data[ticker] = data

print("\nCompleted downloads for:", ", ".join(all_data.keys()))
print(f"All data saved under: {os.path.abspath(output_dir)}")
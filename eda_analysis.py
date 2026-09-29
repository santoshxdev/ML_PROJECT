"""
Customer Lifetime Value Prediction Using Elastic Net Regression
Exploratory Data Analysis - reproducible script
Dataset: UCI Online Retail (Online Retail.xlsx)

Usage:  python eda_analysis.py <path_to_xlsx> <output_dir>
Outputs: PNG charts in <output_dir>/charts and stats.json (every number used in the report)
"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

XLSX = sys.argv[1] if len(sys.argv) > 1 else "Online_Retail.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
CH = os.path.join(OUT, "charts")
os.makedirs(CH, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"
S = {}  # stats container


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CH, name), dpi=150)
    plt.close(fig)


def pct(a, b):
    return round(100.0 * a / b, 2)


# ------------------------------------------------------------------ 1. LOAD
df = pd.read_excel(XLSX)
N = len(df)
S["n_rows"], S["n_cols"] = N, df.shape[1]
S["columns"] = list(df.columns)
S["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
S["date_min"], S["date_max"] = str(df.InvoiceDate.min()), str(df.InvoiceDate.max())
S["n_customers"] = int(df.CustomerID.nunique())
S["n_invoices"] = int(df.InvoiceNo.nunique())
S["n_products"] = int(df.StockCode.nunique())
S["n_countries"] = int(df.Country.nunique())
S["n_descriptions"] = int(df.Description.nunique())

# ------------------------------------------------------------------ 2. QUALITY
inv = df.InvoiceNo.astype(str)
df["IsCancel"] = inv.str.startswith("C")
S["invoice_prefixes"] = {k: int(v) for k, v in inv.str.extract(r"^([A-Za-z]+)")[0].value_counts().items()}
S["invoiceno_python_types"] = {k: int(v) for k, v in df.InvoiceNo.map(lambda x: type(x).__name__).value_counts().items()}
S["stockcode_python_types"] = {k: int(v) for k, v in df.StockCode.map(lambda x: type(x).__name__).value_counts().items()}

miss = df[["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]].isna().sum()
S["missing"] = {c: int(v) for c, v in miss.items()}
S["missing_pct"] = {c: pct(v, N) for c, v in miss.items()}

dups = int(df.drop(columns="IsCancel").duplicated().sum())
S["dup_rows"], S["dup_pct"] = dups, pct(dups, N)
dup_mask = df.drop(columns="IsCancel").duplicated(keep=False)
S["dup_rows_incl_first"] = int(dup_mask.sum())
S["dup_cancel_rows"] = int((df.drop(columns="IsCancel").duplicated() & df.IsCancel).sum())
S["dup_missing_cust_rows"] = int((df.drop(columns="IsCancel").duplicated() & df.CustomerID.isna()).sum())
S["dup_customer_level_spend"] = float(
    (df[df.drop(columns="IsCancel").duplicated() & df.CustomerID.notna() & ~df.IsCancel & (df.Quantity > 0) & (df.UnitPrice > 0)]
     .eval("Quantity*UnitPrice")).sum())

S["neg_qty"], S["neg_qty_pct"] = int((df.Quantity < 0).sum()), pct((df.Quantity < 0).sum(), N)
S["zero_qty"] = int((df.Quantity == 0).sum())
S["zero_price"], S["zero_price_pct"] = int((df.UnitPrice == 0).sum()), pct((df.UnitPrice == 0).sum(), N)
S["neg_price"] = int((df.UnitPrice < 0).sum())
S["cancel_rows"], S["cancel_rows_pct"] = int(df.IsCancel.sum()), pct(df.IsCancel.sum(), N)
S["cancel_invoices"] = int(df[df.IsCancel].InvoiceNo.nunique())
S["cancel_invoices_pct"] = pct(S["cancel_invoices"], S["n_invoices"])
S["missing_cust"], S["missing_cust_pct"] = int(df.CustomerID.isna().sum()), pct(df.CustomerID.isna().sum(), N)
S["neg_qty_not_cancel"] = int(((df.Quantity < 0) & ~df.IsCancel).sum())
S["cancel_with_pos_qty"] = int((df.IsCancel & (df.Quantity > 0)).sum())
S["neg_qty_not_cancel_missing_cust"] = int(((df.Quantity < 0) & ~df.IsCancel & df.CustomerID.isna()).sum())
S["neg_qty_not_cancel_desc_missing"] = int(((df.Quantity < 0) & ~df.IsCancel & df.Description.isna()).sum())
S["zero_price_missing_cust"] = int(((df.UnitPrice == 0) & df.CustomerID.isna()).sum())
S["zero_price_missing_desc"] = int(((df.UnitPrice == 0) & df.Description.isna()).sum())
S["missing_desc_missing_cust"] = int((df.Description.isna() & df.CustomerID.isna()).sum())
S["cancel_missing_cust"] = int((df.IsCancel & df.CustomerID.isna()).sum())
S["cust_id_all_integer_valued"] = bool((df.CustomerID.dropna() % 1 == 0).all())
S["cust_id_missing_invoices"] = int(df[df.CustomerID.isna()].InvoiceNo.nunique())

# non-product stock codes (alphabetic only) - inspection
sc = df.StockCode.astype(str)
nonprod = sc[~sc.str.contains(r"\d")]
S["nonproduct_codes"] = {k: int(v) for k, v in nonprod.value_counts().head(12).items()}
S["nonproduct_rows"] = int(len(nonprod))

# overlap-aware "valid sale" definition
valid = (df.CustomerID.notna() & ~df.IsCancel & (df.Quantity > 0) & (df.UnitPrice > 0))
S["valid_rows"], S["valid_pct"] = int(valid.sum()), pct(valid.sum(), N)
S["invalid_rows"] = int((~valid).sum())

# TotalAmount only after checking columns exist
assert {"Quantity", "UnitPrice"}.issubset(df.columns)
df["TotalAmount"] = df.Quantity * df.UnitPrice
S["total_amount_created"] = True

# ------------------------------------------------------------------ 3. CHARTS: quality
# Missing values
fig, ax = plt.subplots(figsize=(8, 4.2))
m = miss[miss > 0].sort_values()
allc = pd.Series({c: S["missing_pct"][c] for c in S["columns"]}).sort_values()
ax.barh(allc.index, allc.values, color=[ORANGE if v > 0 else GREY for v in allc.values])
for i, (c, v) in enumerate(allc.items()):
    ax.text(v + 0.3, i, f"{S['missing'][c]:,} ({v:.2f}%)", va="center", fontsize=9)
ax.set_xlim(0, max(allc.values) * 1.35)
ax.set_title("Missing Values by Column (Percentage of 541,909 Records)")
ax.set_xlabel("Missing values (%)")
ax.set_ylabel("Column")
save(fig, "01_missing_values.png")

# Data-quality issue counts
fig, ax = plt.subplots(figsize=(8, 4.2))
issues = {
    "Missing CustomerID": S["missing_cust"],
    "Cancelled-invoice rows": S["cancel_rows"],
    "Exact duplicate rows": S["dup_rows"],
    "Missing Description": S["missing"]["Description"],
    "Zero UnitPrice": S["zero_price"],
    "Negative UnitPrice": S["neg_price"],
    "Zero Quantity": S["zero_qty"],
}
s_ = pd.Series(issues).sort_values()
ax.barh(s_.index, s_.values, color=BLUE)
for i, v in enumerate(s_.values):
    ax.text(v + 1500, i, f"{v:,} ({pct(v, N):.2f}%)", va="center", fontsize=9)
ax.set_xlim(0, s_.max() * 1.3)
ax.set_title("Data Quality Issues: Number of Affected Records")
ax.set_xlabel("Number of records")
ax.set_ylabel("Issue type")
save(fig, "02_quality_issues.png")

# Duplicates
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.bar(["Unique / first\noccurrence", "Exact duplicate\nrows"], [N - dups, dups], color=[BLUE, ORANGE])
for i, v in enumerate([N - dups, dups]):
    ax.text(i, v + N * 0.01, f"{v:,}\n({pct(v, N):.2f}%)", ha="center", fontsize=9)
ax.set_ylim(0, N * 1.15)
ax.set_title("Duplicate Rows in the Dataset")
ax.set_ylabel("Number of records")
ax.set_xlabel("Record category")
save(fig, "03_duplicates.png")

# ------------------------------------------------------------------ 4. TRANSACTIONS
sales = df[valid].copy()          # valid customer sales rows
sales_all_pos = df[(~df.IsCancel) & (df.Quantity > 0) & (df.UnitPrice > 0)].copy()  # incl. missing customer
df["Date"] = df.InvoiceDate.dt.normalize()
df["Month"] = df.InvoiceDate.dt.to_period("M")

daily = df[~df.IsCancel].groupby("Date").InvoiceNo.nunique()
S["days_with_sales"] = int(len(daily))
S["daily_inv_mean"] = round(float(daily.mean()), 2)
S["daily_inv_max"] = int(daily.max()); S["daily_inv_max_date"] = str(daily.idxmax().date())
S["n_saturdays_with_sales"] = int((daily.index.dayofweek == 5).sum())
dow = df[~df.IsCancel].groupby(df.InvoiceDate.dt.dayofweek).InvoiceNo.nunique()
S["invoices_by_dow"] = {int(k): int(v) for k, v in dow.items()}

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(daily.index, daily.values, color=BLUE, lw=0.8, alpha=0.6, label="Daily invoices")
ax.plot(daily.index, daily.rolling(7, min_periods=1).mean(), color=ORANGE, lw=1.8, label="7-day rolling mean")
ax.set_title("Number of Sales Invoices per Day")
ax.set_xlabel("Invoice date"); ax.set_ylabel("Unique sales invoices per day (count)")
ax.legend()
fig.autofmt_xdate()
save(fig, "04_daily_transactions.png")

mon_inv = df[~df.IsCancel].groupby("Month").InvoiceNo.nunique()
mon_rev = sales_all_pos.assign(Month=sales_all_pos.InvoiceDate.dt.dt.to_period("M") if False else sales_all_pos.InvoiceDate.dt.to_period("M")).groupby("Month").TotalAmount.sum()
S["monthly_invoices"] = {str(k): int(v) for k, v in mon_inv.items()}
S["monthly_revenue"] = {str(k): round(float(v), 2) for k, v in mon_rev.items()}
last_month_last_day = str(df.InvoiceDate.max().date())
S["last_invoice_day"] = last_month_last_day

def month_chart(series, title, ylabel, name, color, fmt):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = [str(p) for p in series.index]
    cols = [color] * len(series)
    cols[-1] = GREY  # partial month
    ax.bar(labels, series.values, color=cols)
    ax.set_title(title); ax.set_xlabel("Month (last bar = partial month, data ends 9 Dec 2011)")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, name)

month_chart(mon_inv, "Monthly Transaction Volume (Sales Invoices)", "Unique sales invoices (count)", "05_monthly_volume.png", BLUE, lambda x, p: f"{int(x):,}")
month_chart(mon_rev, "Monthly Revenue (Positive-Quantity, Positive-Price Sales)", "Revenue (GBP)", "06_monthly_revenue.png", ORANGE, lambda x, p: f"{x/1000:,.0f}k")

# distributions on valid sales rows
def desc(s):
    return {k: round(float(v), 4) for k, v in s.describe(percentiles=[.25, .5, .75, .95, .99]).items()}
S["desc_quantity"] = desc(sales.Quantity)
S["desc_unitprice"] = desc(sales.UnitPrice)
S["desc_total"] = desc(sales.TotalAmount)
S["skew_quantity"] = round(float(sales.Quantity.skew()), 2)
S["skew_unitprice"] = round(float(sales.UnitPrice.skew()), 2)
S["skew_total"] = round(float(sales.TotalAmount.skew()), 2)

def dist_chart(s, title, xlabel, name, color, clip_q=0.99):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    cap = s.quantile(clip_q)
    b0 = np.arange(0.5, cap + 2.5, 2) if float(s.min()).is_integer() and float(s.max()).is_integer() else 50
    axs[0].hist(s[s <= cap], bins=b0, color=color, edgecolor="white")
    axs[0].set_title(f"{title}\n(values up to 99th percentile = {cap:,.2f})", fontsize=10)
    axs[0].set_xlabel(xlabel); axs[0].set_ylabel("Number of records")
    axs[1].hist(np.log10(s), bins=50, color=color, edgecolor="white")
    axs[1].set_title(f"{title}\n(full range, log10 scale)", fontsize=10)
    axs[1].set_xlabel(f"log10({xlabel})"); axs[1].set_ylabel("Number of records")
    save(fig, name)

dist_chart(sales.Quantity, "Distribution of Quantity", "Quantity per line (units)", "07_quantity_dist.png", BLUE)
dist_chart(sales.UnitPrice, "Distribution of UnitPrice", "Unit price (GBP)", "08_unitprice_dist.png", ORANGE)
dist_chart(sales.TotalAmount, "Distribution of TotalAmount", "Total amount per line (GBP)", "09_totalamount_dist.png", BLUE)

# ------------------------------------------------------------------ 5. CANCELLATIONS
c = df[df.IsCancel]
S["cancel_qty"] = int(c.Quantity.sum())
S["cancel_qty_abs"] = int(c.Quantity.abs().sum())
S["cancel_value"] = round(float(c.TotalAmount.sum()), 2)
S["cancel_value_abs"] = round(float(c.TotalAmount.abs().sum()), 2)
gross_sales_val = float(sales_all_pos.TotalAmount.sum())
S["gross_sales_value_all"] = round(gross_sales_val, 2)
S["gross_sales_qty_all"] = int(sales_all_pos.Quantity.sum())
S["cancel_value_share_of_sales"] = pct(abs(S["cancel_value"]), gross_sales_val)
S["cancel_qty_share_of_sales"] = pct(S["cancel_qty_abs"], S["gross_sales_qty_all"])
S["cancel_with_customer_rows"] = int(c.CustomerID.notna().sum())
S["cancel_customers"] = int(c.CustomerID.nunique())
S["cancel_customers_pct_of_valid_customers"] = pct(c.CustomerID.nunique(), sales.CustomerID.nunique())
S["cancel_rows_pos_price"] = int((c.UnitPrice > 0).sum())
S["cancel_price_zero"] = int((c.UnitPrice == 0).sum())
non_c_neg = df[(df.Quantity < 0) & ~df.IsCancel]
S["neg_not_cancel_nonzero_price"] = int((non_c_neg.UnitPrice > 0).sum())
S["neg_not_cancel_zero_price"] = int((non_c_neg.UnitPrice == 0).sum())
S["neg_not_cancel_top_desc"] = {str(k): int(v) for k, v in non_c_neg.Description.value_counts().head(6).items()}
S["cancel_top_stock"] = {str(k): int(v) for k, v in c.StockCode.value_counts().head(5).items()}
S["cancel_D_rows"] = int((c.StockCode.astype(str) == "D").sum())

cm = c.groupby("Month").InvoiceNo.nunique()
cm = cm.reindex(mon_inv.index, fill_value=0)
rate = (cm / (mon_inv + cm) * 100)
S["monthly_cancel_invoices"] = {str(k): int(v) for k, v in cm.items()}
S["monthly_cancel_rate"] = {str(k): round(float(v), 2) for k, v in rate.items()}

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
axs[0].bar(["Sales invoices", "Cancelled invoices"], [S["n_invoices"] - S["cancel_invoices"], S["cancel_invoices"]], color=[BLUE, ORANGE])
for i, v in enumerate([S["n_invoices"] - S["cancel_invoices"], S["cancel_invoices"]]):
    axs[0].text(i, v + 300, f"{v:,}", ha="center", fontsize=9)
axs[0].set_ylim(0, (S["n_invoices"] - S["cancel_invoices"]) * 1.12)
axs[0].set_title("Sales vs Cancelled Invoices"); axs[0].set_xlabel("Invoice type"); axs[0].set_ylabel("Unique invoices (count)")
axs[1].bar([str(p) for p in cm.index], cm.values, color=ORANGE)
axs[1].set_title("Cancelled Invoices per Month"); axs[1].set_xlabel("Month"); axs[1].set_ylabel("Cancelled invoices (count)")
plt.setp(axs[1].get_xticklabels(), rotation=60, ha="right", fontsize=8)
save(fig, "10_cancellations.png")

# ------------------------------------------------------------------ 6. CUSTOMERS / RFM
snapshot = df.InvoiceDate.max().normalize() + pd.Timedelta(days=1)
S["snapshot_date"] = str(snapshot.date())
g = sales.groupby("CustomerID")
cust = pd.DataFrame({
    "Recency": (snapshot - g.InvoiceDate.max().dt.normalize()).dt.days,
    "Frequency": g.InvoiceNo.nunique(),
    "MonetaryValue": g.TotalAmount.sum(),
    "TotalQuantity": g.Quantity.sum(),
    "FirstPurchase": g.InvoiceDate.min(),
    "LastPurchase": g.InvoiceDate.max(),
})
cust["AverageOrderValue"] = cust.MonetaryValue / cust.Frequency
cust["CustomerTenure"] = (snapshot - cust.FirstPurchase.dt.normalize()).dt.days       # days since first purchase
cust["PurchaseFrequency"] = cust.Frequency / np.maximum(cust.CustomerTenure / 30.44, 1)  # orders per month
S["n_valid_customers"] = int(len(cust))
S["valid_revenue"] = round(float(cust.MonetaryValue.sum()), 2)
S["n_customers_no_valid_sale"] = S["n_customers"] - len(cust)

for col in ["Frequency", "MonetaryValue", "AverageOrderValue", "TotalQuantity", "Recency", "CustomerTenure", "PurchaseFrequency"]:
    S["desc_" + col] = desc(cust[col])
    S["skew_" + col] = round(float(cust[col].skew()), 2)
S["single_order_customers"] = int((cust.Frequency == 1).sum())
S["single_order_pct"] = pct((cust.Frequency == 1).sum(), len(cust))
S["repeat_customers_pct"] = round(100 - S["single_order_pct"], 2)
S["mean_median_monetary"] = [round(float(cust.MonetaryValue.mean()), 2), round(float(cust.MonetaryValue.median()), 2)]

ms = cust.MonetaryValue.sort_values(ascending=False)
cum = ms.cumsum() / ms.sum()
for p in (0.01, 0.05, 0.10, 0.20):
    k = max(1, int(round(len(ms) * p)))
    S[f"top{int(p*100)}pct_rev_share"] = round(float(ms.iloc[:k].sum() / ms.sum() * 100), 2)
S["customers_for_50pct_rev"] = int((cum < 0.5).sum() + 1)
S["customers_for_50pct_rev_pct"] = pct(S["customers_for_50pct_rev"], len(ms))
S["top1_customer_rev_share"] = round(float(ms.iloc[0] / ms.sum() * 100), 2)
S["top1_customer_rev"] = round(float(ms.iloc[0]), 2)
S["bottom50pct_rev_share"] = round(float(ms.iloc[len(ms)//2:].sum() / ms.sum() * 100), 2)

# Spending distribution
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(cust.MonetaryValue[cust.MonetaryValue <= cust.MonetaryValue.quantile(.95)], bins=50, color=BLUE, edgecolor="white")
axs[0].set_title("Customer Total Spending\n(up to 95th percentile)", fontsize=10)
axs[0].set_xlabel("Total spending per customer (GBP)"); axs[0].set_ylabel("Number of customers")
axs[1].hist(np.log10(cust.MonetaryValue), bins=50, color=BLUE, edgecolor="white")
axs[1].set_title("Customer Total Spending\n(full range, log10 scale)", fontsize=10)
axs[1].set_xlabel("log10(total spending, GBP)"); axs[1].set_ylabel("Number of customers")
save(fig, "11_customer_spending.png")

# Frequency distribution
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
fc = cust.Frequency.clip(upper=20).value_counts().sort_index()
lab = [str(i) if i < 20 else "20+" for i in fc.index]
axs[0].bar(lab, fc.values, color=ORANGE)
axs[0].set_title("Orders per Customer (values above 20 grouped)", fontsize=10)
axs[0].set_xlabel("Number of orders (invoices)"); axs[0].set_ylabel("Number of customers")
axs[1].hist(cust.Frequency, bins=np.logspace(0, np.log10(cust.Frequency.max() + 1), 25), color=ORANGE, edgecolor="white")
axs[1].set_xscale("log")
axs[1].set_title("Orders per Customer (full range, log-spaced bins)", fontsize=10)
axs[1].set_xlabel("Number of orders (log scale)"); axs[1].set_ylabel("Number of customers")
save(fig, "12_customer_frequency.png")

# Top customers
top_rev = cust.MonetaryValue.sort_values(ascending=False).head(10)
top_ord = cust.Frequency.sort_values(ascending=False).head(10)
S["top_rev_customers"] = [(int(i), round(float(v), 2)) for i, v in top_rev.items()]
S["top_ord_customers"] = [(int(i), int(v)) for i, v in top_ord.items()]
top_rev_country = sales.groupby("CustomerID").Country.agg(lambda x: x.mode()[0])
S["top_rev_customers_country"] = [top_rev_country[i] for i in top_rev.index]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
axs[0].barh([f"ID {int(i)}" for i in top_rev.index][::-1], top_rev.values[::-1], color=BLUE)
axs[0].set_title("Top 10 Customers by Revenue"); axs[0].set_xlabel("Total revenue (GBP)"); axs[0].set_ylabel("CustomerID (anonymised identifier)")
axs[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x/1000:,.0f}k"))
axs[1].barh([f"ID {int(i)}" for i in top_ord.index][::-1], top_ord.values[::-1], color=ORANGE)
axs[1].set_title("Top 10 Customers by Number of Orders"); axs[1].set_xlabel("Number of orders (invoices)"); axs[1].set_ylabel("CustomerID (anonymised identifier)")
save(fig, "13_top_customers.png")

# Lorenz-style cumulative revenue
fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(1, len(ms) + 1) / len(ms) * 100
ax.plot(x, cum.values * 100, color=BLUE, lw=2, label="Cumulative revenue share")
ax.plot([0, 100], [0, 100], "--", color=GREY, label="Perfect equality")
ax.set_title("Revenue Concentration Across Customers")
ax.set_xlabel("Customers ranked by spending, highest first (% of customers)")
ax.set_ylabel("Cumulative revenue (%)"); ax.legend()
save(fig, "14_revenue_concentration.png")

# RFM plots
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
axs[0].hist(cust.Recency, bins=40, color=BLUE, edgecolor="white")
axs[0].set_title("Recency"); axs[0].set_xlabel("Days since last purchase"); axs[0].set_ylabel("Number of customers")
_f95 = cust.Frequency.quantile(.95)
axs[1].hist(cust.Frequency[cust.Frequency <= _f95], bins=np.arange(0.5, _f95 + 1.5, 1), color=ORANGE, edgecolor="white")
axs[1].set_title(f"Frequency (up to 95th percentile = {_f95:.0f})"); axs[1].set_xlabel("Number of orders"); axs[1].set_ylabel("Number of customers")
axs[2].hist(np.log10(cust.MonetaryValue), bins=40, color=BLUE, edgecolor="white")
axs[2].set_title("Monetary Value (log10 scale)"); axs[2].set_xlabel("log10(total spending, GBP)"); axs[2].set_ylabel("Number of customers")
save(fig, "15_rfm_distributions.png")

# Recency vs Frequency scatter (exploratory)
fig, ax = plt.subplots(figsize=(6.5, 4.4))
sc_ = ax.scatter(cust.Recency, cust.Frequency, c=np.log10(cust.MonetaryValue), s=8, alpha=0.6, cmap="viridis")
ax.set_yscale("log")
ax.set_title("Recency vs Frequency (colour = log10 Monetary Value)")
ax.set_xlabel("Recency (days since last purchase)"); ax.set_ylabel("Frequency (orders, log scale)")
cb = fig.colorbar(sc_); cb.set_label("log10(Monetary Value, GBP)")
save(fig, "16_recency_vs_frequency.png")

# ------------------------------------------------------------------ 7. COUNTRY
cg = sales.groupby("Country").agg(Customers=("CustomerID", "nunique"), Transactions=("InvoiceNo", "nunique"), Revenue=("TotalAmount", "sum"))
cg = cg.sort_values("Revenue", ascending=False)
S["country_table_top10"] = [(k, int(r.Customers), int(r.Transactions), round(float(r.Revenue), 2)) for k, r in cg.head(10).iterrows()]
S["n_countries_valid"] = int(len(cg))
S["uk_customers_pct"] = pct(cg.loc["United Kingdom", "Customers"], cg.Customers.sum())
S["uk_transactions_pct"] = pct(cg.loc["United Kingdom", "Transactions"], cg.Transactions.sum())
S["uk_revenue_pct"] = pct(cg.loc["United Kingdom", "Revenue"], cg.Revenue.sum())
S["nonuk_revenue_pct"] = round(100 - S["uk_revenue_pct"], 2)
S["countries_single_customer"] = int((cg.Customers == 1).sum())
S["country_all_rows_top5"] = {k: int(v) for k, v in df.Country.value_counts().head(5).items()}
S["unspecified_rows"] = int((df.Country == "Unspecified").sum())
S["customers_multi_country"] = int((sales.groupby("CustomerID").Country.nunique() > 1).sum())

fig, axs = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
top10 = cg.head(10)
yl = top10.index[::-1]
axs[0].barh(yl, top10.Customers[::-1], color=BLUE); axs[0].set_title("Customers")
axs[0].set_xlabel("Unique customers (count, log scale)"); axs[0].set_ylabel("Country")
axs[1].barh(yl, top10.Transactions[::-1], color=ORANGE); axs[1].set_title("Transactions")
axs[1].set_xlabel("Unique sales invoices (count, log scale)")
axs[2].barh(yl, top10.Revenue[::-1], color=BLUE); axs[2].set_title("Revenue")
axs[2].set_xlabel("Revenue (GBP, log scale)")
for a in axs:
    a.set_xscale("log")
fig.suptitle("Top 10 Countries by Revenue (valid customer sales; log-scaled x-axes)")
save(fig, "17_country_analysis.png")

nuk = cg.drop("United Kingdom").head(10)
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.barh(nuk.index[::-1], nuk.Revenue[::-1], color=BLUE)
ax.set_title("Top 10 Non-UK Countries by Revenue"); ax.set_xlabel("Revenue (GBP)"); ax.set_ylabel("Country")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x/1000:,.0f}k"))
save(fig, "18_country_nonuk.png")

# ------------------------------------------------------------------ 8. CORRELATION
feats = ["Recency", "Frequency", "MonetaryValue", "AverageOrderValue", "TotalQuantity", "CustomerTenure", "PurchaseFrequency"]
corr_p = cust[feats].corr(method="pearson")
corr_s = cust[feats].corr(method="spearman")
corr_l = np.log1p(cust[feats]).corr(method="pearson")
S["corr_pearson"] = corr_p.round(3).to_dict()
S["corr_spearman"] = corr_s.round(3).to_dict()
S["corr_log"] = corr_l.round(3).to_dict()
pairs = []
for i, a in enumerate(feats):
    for b in feats[i + 1:]:
        pairs.append((a, b, round(float(corr_p.loc[a, b]), 3), round(float(corr_s.loc[a, b]), 3), round(float(corr_l.loc[a, b]), 3)))
S["corr_pairs"] = pairs

fig, ax = plt.subplots(figsize=(8, 6.4))
sns.heatmap(corr_p, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"label": "Pearson correlation"}, ax=ax)
ax.set_title("Pearson Correlation Heatmap of Customer-Level Features")
ax.set_xlabel("Feature"); ax.set_ylabel("Feature")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
save(fig, "19_correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(8, 6.4))
sns.heatmap(corr_l, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"label": "Pearson correlation (log1p features)"}, ax=ax)
ax.set_title("Correlation Heatmap After log(1+x) Transformation")
ax.set_xlabel("Feature"); ax.set_ylabel("Feature")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
save(fig, "20_correlation_heatmap_log.png")

# VIF (computed manually on standardised log1p features)
X = np.log1p(cust[feats]); X = (X - X.mean()) / X.std()
vif = {}
for f in feats:
    y = X[f].values; others = X.drop(columns=f).values
    A = np.c_[np.ones(len(others)), others]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    vif[f] = round(float(1 / (1 - r2)), 2)
S["vif_log"] = vif

# ------------------------------------------------------------------ 9. OUTLIERS
def iqr_info(s):
    q1, q3 = s.quantile(.25), s.quantile(.75); i = q3 - q1
    hi = q3 + 1.5 * i; lo = q1 - 1.5 * i
    n = int(((s > hi) | (s < lo)).sum())
    return {"q1": round(float(q1), 2), "q3": round(float(q3), 2), "upper_fence": round(float(hi), 2), "n_outliers": n, "pct_outliers": pct(n, len(s)), "max": round(float(s.max()), 2), "min": round(float(s.min()), 2)}

S["iqr"] = {
    "Quantity": iqr_info(sales.Quantity), "UnitPrice": iqr_info(sales.UnitPrice), "TotalAmount": iqr_info(sales.TotalAmount),
    "MonetaryValue": iqr_info(cust.MonetaryValue), "Frequency": iqr_info(cust.Frequency),
}
# extreme rows inspection
ex = df[(~df.IsCancel) & (df.Quantity > 10000)][["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "CustomerID", "InvoiceDate"]]
S["qty_gt_10000_rows"] = [
    {"InvoiceNo": str(r.InvoiceNo), "StockCode": str(r.StockCode), "Description": str(r.Description), "Quantity": int(r.Quantity),
     "UnitPrice": float(r.UnitPrice), "CustomerID": None if pd.isna(r.CustomerID) else int(r.CustomerID), "Date": str(r.InvoiceDate.date())}
    for r in ex.itertuples()]
S["cancel_large"] = []
for r in df[df.IsCancel & (df.Quantity.abs() > 10000)].itertuples():
    S["cancel_large"].append({"InvoiceNo": str(r.InvoiceNo), "StockCode": str(r.StockCode), "Description": str(r.Description), "Quantity": int(r.Quantity), "UnitPrice": float(r.UnitPrice), "CustomerID": None if pd.isna(r.CustomerID) else int(r.CustomerID), "Date": str(r.InvoiceDate.date())})
hp = df[(~df.IsCancel) & (df.UnitPrice > 1000)]
S["price_gt_1000_rows"] = [{"StockCode": str(r.StockCode), "Description": str(r.Description), "Quantity": int(r.Quantity), "UnitPrice": float(r.UnitPrice), "CustomerID": None if pd.isna(r.CustomerID) else int(r.CustomerID)} for r in hp.itertuples()]
S["price_gt_1000_count"] = int(len(hp))
S["price_gt_1000_codes"] = {str(k): int(v) for k, v in hp.StockCode.value_counts().items()}
S["price_max_valid_code"] = str(sales.loc[sales.UnitPrice.idxmax(), "StockCode"])
S["price_max_valid_desc"] = str(sales.loc[sales.UnitPrice.idxmax(), "Description"])
S["price_max_valid"] = float(sales.UnitPrice.max())
S["total_max_row"] = {"Quantity": int(sales.loc[sales.TotalAmount.idxmax(), "Quantity"]), "UnitPrice": float(sales.loc[sales.TotalAmount.idxmax(), "UnitPrice"]), "Total": round(float(sales.TotalAmount.max()), 2), "Description": str(sales.loc[sales.TotalAmount.idxmax(), "Description"]), "Date": str(sales.loc[sales.TotalAmount.idxmax(), "InvoiceDate"].date())}
mx_cust = sales.loc[sales.TotalAmount.idxmax(), "CustomerID"]
S["total_max_customer_rev_share"] = round(float(cust.loc[mx_cust, "MonetaryValue"] / cust.MonetaryValue.sum() * 100), 2)
# Effect of the extreme pair on the customer monetary distribution
S["max_monetary_customer_id"] = int(cust.MonetaryValue.idxmax())
S["max_monetary_customer_freq"] = int(cust.loc[cust.MonetaryValue.idxmax(), "Frequency"])
S["n_cust_freq_ge_50"] = int((cust.Frequency >= 50).sum())
# non-product rows within valid sales
nonp_valid = sales[~sales.StockCode.astype(str).str.contains(r"\d")]
S["nonproduct_valid_rows"] = int(len(nonp_valid))
S["nonproduct_valid_revenue"] = round(float(nonp_valid.TotalAmount.sum()), 2)
S["nonproduct_valid_codes"] = {k: int(v) for k, v in nonp_valid.StockCode.value_counts().items()}
S["nonproduct_valid_rev_by_code"] = {k: round(float(v), 2) for k, v in nonp_valid.groupby("StockCode").TotalAmount.sum().sort_values(ascending=False).items()}

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for a, (s, t, xl) in zip(axs, [(sales.Quantity, "Quantity", "Quantity per line (units, log scale)"),
                              (sales.UnitPrice, "UnitPrice", "Unit price (GBP, log scale)"),
                              (sales.TotalAmount, "TotalAmount", "Total amount per line (GBP, log scale)")]):
    sns.boxplot(x=s, ax=a, color=BLUE, fliersize=1.5, flierprops={"alpha": .3})
    a.set_xscale("log"); a.set_title(f"Boxplot of {t}"); a.set_xlabel(xl); a.set_ylabel("Records")
save(fig, "21_boxplots_transaction.png")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for a, (s, t, xl) in zip(axs, [(cust.MonetaryValue, "MonetaryValue", "Total spending per customer (GBP, log scale)"),
                              (cust.Frequency, "Frequency", "Orders per customer (count, log scale)")]):
    sns.boxplot(x=s, ax=a, color=ORANGE, fliersize=2, flierprops={"alpha": .4})
    a.set_xscale("log"); a.set_title(f"Boxplot of {t}"); a.set_xlabel(xl); a.set_ylabel("Customers")
save(fig, "22_boxplots_customer.png")

# ------------------------------------------------------------------ 10. FUTURE-CLV illustration (methodology only, not model)
# Show only counts to justify a chronological split; no target is created here.
S["months_span"] = round((df.InvoiceDate.max() - df.InvoiceDate.min()).days / 30.44, 1)
S["days_span"] = int((df.InvoiceDate.max() - df.InvoiceDate.min()).days)
# Illustrative feasibility check: customers with a valid purchase before and after a 2011-09-01 cutoff
cut = pd.Timestamp("2011-09-01")
before = set(sales[sales.InvoiceDate < cut].CustomerID); after = set(sales[sales.InvoiceDate >= cut].CustomerID)
S["cutoff_example"] = str(cut.date())
S["cust_before_cutoff"] = len(before)
S["cust_after_cutoff"] = len(after)
S["cust_both"] = len(before & after)
S["cust_before_only"] = len(before - after)
S["cust_after_only_new"] = len(after - before)

# ------------------------------------------------------------------ SAVE
cust.reset_index().to_csv(os.path.join(OUT, "customer_level_features_full_period.csv"), index=False)
with open(os.path.join(OUT, "stats.json"), "w") as f:
    json.dump(S, f, indent=1, default=str)
print("Done. Stats keys:", len(S))

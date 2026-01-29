"""Generate a synthetic Excel file for testing."""

from pathlib import Path
from openpyxl import Workbook


def create_sample_workbook():
    """Create a realistic sample performance workbook."""
    wb = Workbook()
    wb.remove(wb.active)

    strategies = {
        "Large Cap Growth": [
            ("NVDA", "NVIDIA Corporation", 4.2, 3.8, 4.6, 3.1, 12.5, 48.5, 35.2, 13.3),
            ("AAPL", "Apple Inc.", 5.8, 5.5, 6.1, 6.2, 8.3, 32.1, 24.8, 7.3),
            ("MSFT", "Microsoft Corporation", 6.5, 6.2, 6.8, 5.8, 10.2, 28.7, 19.4, 9.3),
            ("AMZN", "Amazon.com Inc.", 4.1, 3.9, 4.3, 3.5, 9.8, 22.4, 15.6, 6.8),
            ("META", "Meta Platforms Inc.", 3.2, 2.9, 3.5, 2.8, 11.2, 18.9, 12.3, 6.6),
            ("GOOGL", "Alphabet Inc.", 4.5, 4.3, 4.7, 4.8, 7.5, 12.3, 8.7, 3.6),
            ("TSLA", "Tesla Inc.", 2.1, 2.4, 1.8, 1.9, -2.3, 8.2, 5.1, 3.1),
            ("CRM", "Salesforce Inc.", 1.8, 1.7, 1.9, 1.5, 6.2, 4.5, 2.8, 1.7),
            ("ADBE", "Adobe Inc.", 1.5, 1.4, 1.6, 1.6, 3.1, -2.3, -1.5, -0.8),
            ("NFLX", "Netflix Inc.", 1.2, 1.1, 1.3, 1.0, 4.5, -5.8, -3.9, -1.9),
            ("INTC", "Intel Corporation", 0.8, 0.9, 0.7, 1.2, -8.5, -12.4, -8.6, -3.8),
            ("DIS", "The Walt Disney Company", 1.0, 1.1, 0.9, 1.3, -5.2, -18.7, -12.3, -6.4),
            ("PYPL", "PayPal Holdings Inc.", 0.6, 0.7, 0.5, 0.8, -12.3, -24.5, -17.8, -6.7),
            ("MRNA", "Moderna Inc.", 0.4, 0.5, 0.3, 0.5, -15.8, -32.1, -23.4, -8.7),
            ("ZM", "Zoom Video Communications", 0.3, 0.4, 0.2, 0.3, -18.2, -38.9, -28.5, -10.4),
        ],
        "Small Cap Value": [
            ("COOP", "Mr. Cooper Group Inc.", 1.8, 1.6, 2.0, 1.2, 18.5, 42.3, 31.5, 10.8),
            ("CEIX", "CONSOL Energy Inc.", 1.5, 1.3, 1.7, 1.0, 22.1, 35.6, 26.2, 9.4),
            ("BTU", "Peabody Energy Corporation", 1.2, 1.0, 1.4, 0.8, 19.8, 28.4, 20.1, 8.3),
            ("ARCH", "Arch Resources Inc.", 1.0, 0.9, 1.1, 0.7, 16.5, 22.1, 15.8, 6.3),
            ("TMHC", "Taylor Morrison Home Corp.", 0.9, 0.8, 1.0, 0.6, 14.2, 18.7, 13.2, 5.5),
            ("GRBK", "Green Brick Partners Inc.", 0.8, 0.7, 0.9, 0.5, 11.8, 8.2, 5.8, 2.4),
            ("MLI", "Mueller Industries Inc.", 0.7, 0.7, 0.7, 0.6, 5.2, 3.1, 2.1, 1.0),
            ("ANDE", "Andersons Inc.", 0.6, 0.6, 0.6, 0.5, -2.3, -8.5, -5.8, -2.7),
            ("SXC", "SunCoke Energy Inc.", 0.5, 0.5, 0.5, 0.4, -8.5, -15.2, -10.8, -4.4),
            ("PARR", "Par Pacific Holdings Inc.", 0.4, 0.5, 0.3, 0.4, -12.8, -22.8, -16.2, -6.6),
            ("HBI", "Hanesbrands Inc.", 0.3, 0.4, 0.2, 0.3, -18.5, -35.4, -25.8, -9.6),
            ("BIG", "Big Lots Inc.", 0.2, 0.3, 0.1, 0.2, -25.2, -48.2, -35.1, -13.1),
        ],
        "International Equity": [
            ("ASML", "ASML Holding N.V.", 3.5, 3.2, 3.8, 2.8, 15.2, 38.5, 28.2, 10.3),
            ("NVO", "Novo Nordisk A/S", 2.8, 2.5, 3.1, 2.2, 18.5, 32.1, 23.8, 8.3),
            ("SAP", "SAP SE", 2.2, 2.0, 2.4, 1.8, 12.8, 25.4, 18.5, 6.9),
            ("TM", "Toyota Motor Corporation", 2.0, 1.9, 2.1, 2.5, 8.5, 18.2, 12.8, 5.4),
            ("SONY", "Sony Group Corporation", 1.8, 1.6, 2.0, 1.5, 10.2, 15.6, 10.8, 4.8),
            ("SHOP", "Shopify Inc.", 1.2, 1.0, 1.4, 0.8, 14.5, 8.2, 5.5, 2.7),
            ("BABA", "Alibaba Group Holding", 0.8, 1.0, 0.6, 1.2, -5.8, -12.5, -8.8, -3.7),
            ("BIDU", "Baidu Inc.", 0.5, 0.6, 0.4, 0.6, -12.5, -22.8, -16.2, -6.6),
            ("JD", "JD.com Inc.", 0.4, 0.5, 0.3, 0.5, -18.2, -35.2, -25.5, -9.7),
            ("PDD", "PDD Holdings Inc.", 0.3, 0.4, 0.2, 0.4, -22.5, -42.8, -31.2, -11.6),
        ],
    }

    headers = [
        "Ticker", "Company Name", "Avg Weight", "Begin Weight", "End Weight",
        "Benchmark Weight", "Benchmark Return", "Total Attribution",
        "Selection Effect", "Allocation Effect"
    ]

    for strategy_name, holdings in strategies.items():
        ws = wb.create_sheet(strategy_name)
        ws.append(headers)
        for row in holdings:
            ws.append(row)
        # Add summary row
        ws.append(["TOTAL", "Portfolio Total", "", "", "", "", "", 0.0, 0.0, 0.0])

    output_path = Path(__file__).parent.parent / "data" / "synthetic" / "sample_performance.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    create_sample_workbook()

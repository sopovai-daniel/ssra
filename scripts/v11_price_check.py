#!/usr/bin/env python3
"""V11 S0-C: price verification from the Cloud Billing Catalog API (primary source).

Fetches the SKUs relevant to the K1 path (ii) cost projection in EUR
(the billing account currency; per cloud.google.com/storage/pricing,
non-USD accounts are billed at the Cloud Platform SKU prices in their
currency) and writes them to results/v11/v11-s0c-prices.json.

Read-only. Auth: `gcloud auth print-access-token` (run under the local
user's gcloud login). Usage:
    python3 scripts/v11_price_check.py
"""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

COMPUTE = "6F81-5844-456A"  # Compute Engine
STORAGE = "95FF-2EF5-5EA1"  # Cloud Storage

# skuId -> short label of why it matters for K1 path (ii)
WANTED = {
    "C921-088E-792A": "E2 Instance Core running in Frankfurt (on-demand, EUR/vCPU-hour)",
    "7D80-F9E4-6A44": "E2 Instance Ram running in Frankfurt (on-demand, EUR/GiB-hour)",
    "7A7B-EA46-2897": "Storage PD Capacity in Frankfurt (pd-standard, EUR/GiB-month)",
    "B1B5-0BAA-CB31": "Balanced PD Capacity in Frankfurt (pd-balanced, EUR/GiB-month)",
    "E7A9-CA47-A3E3": "Network Internet Data Transfer Out from Frankfurt to EMEA (GCE egress, EUR/GiB)",
    "B8D1-7028-A009": "Network Internet Data Transfer Out from Frankfurt to Western Europe (GCE egress, EUR/GiB)",
    "22EB-AAE8-FBCD": "GCS Download Worldwide Destinations excl. Asia & Australia (GCS internet egress, EUR/GiB)",
    "68F8-91DC-CFA9": "GCS Network Data Transfer GCP Inter Region within Europe (EUR/GiB; same-region has NO SKU = free per docs)",
    "7870-010B-2763": "GCS Regional Standard Class B Operations (EUR/operation; ~104 GETs in K1)",
}


def fetch_skus(token: str, service: str) -> list:
    skus, page = [], ""
    while True:
        url = (
            f"https://cloudbilling.googleapis.com/v1/services/{service}/skus"
            f"?pageSize=5000&currencyCode=EUR"
        )
        if page:
            url += f"&pageToken={page}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        data = json.load(urllib.request.urlopen(req))
        skus += data.get("skus", [])
        page = data.get("nextPageToken", "")
        if not page:
            return skus


def tiers(sku: dict) -> dict:
    expr = sku["pricingInfo"][0]["pricingExpression"]
    return {
        "unit": expr["usageUnitDescription"],
        "tiered_rates_eur": [
            {
                "start_usage_amount": t["startUsageAmount"],
                "price_eur": int(t["unitPrice"]["units"] or 0)
                + t["unitPrice"]["nanos"] / 1e9,
            }
            for t in expr["tieredRates"]
        ],
    }


def main() -> None:
    token = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    out = {"source": "Cloud Billing Catalog API (cloudbilling.googleapis.com/v1), currencyCode=EUR",
           "skus": {}}
    for service in (COMPUTE, STORAGE):
        for sku in fetch_skus(token, service):
            sid = sku["skuId"]
            if sid in WANTED:
                out["skus"][sid] = {
                    "description": sku["description"],
                    "why": WANTED[sid],
                    "service_regions": sku["serviceRegions"],
                    "usage_type": sku["category"]["usageType"],
                    "resource_group": sku["category"]["resourceGroup"],
                    "effective_time": sku["pricingInfo"][0].get("effectiveTime", ""),
                    **tiers(sku),
                }

    missing = sorted(set(WANTED) - set(out["skus"]))
    if missing:
        sys.exit(f"missing expected SKUs: {missing}")

    dest = Path(__file__).resolve().parent.parent / "results" / "v11" / "v11-s0c-prices.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest} ({len(out['skus'])} SKUs)")


if __name__ == "__main__":
    main()

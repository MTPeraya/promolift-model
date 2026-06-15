# Data Dictionary: PromoLift

This document describes the structure of the mock datasets stored in `data/`.

---

## 1. Customer Master (`mock_customer_master.csv`)
Contains customer profile details and segment classifications.

| Column | Type | Description |
|---|---|---|
| `customer_id` | String | Unique identifier for each customer (e.g. `C0001` - `C1000`). |
| `customer_taxonomies` | String | Customer behavior classification (e.g., `High Value`, `Frequent Shopper`, `Price Sensitive`, `Occasional`). |

*Note: `mock_customer_master_with_ground_truth.csv` contains an additional column `true_uplift_segment` (`Persuadal`, `Sure Thing`, `Lost Cause`, `Sleeping Dog`) which is the simulated ground truth and is used for model verification and notebooks.*

---

## 2. Product Master (`mock_product_master.csv`)
Contains product information, prices, and classifications.

| Column | Type | Description |
|---|---|---|
| `product_id` | String | Unique identifier for each product (e.g., `P001` - `P050`). |
| `price` | Float | Unit selling price of the product (THB). |
| `cogs` | Float | Cost of Goods Sold for the product (THB) — used for profit calculations. |
| `product_taxonomies` | String | Category classification of the product (e.g. `Fresh Food`, `Dry Grocery`, `Beverages`, `Household`, `Personal Care`). |

---

## 3. Store Master (`mock_store_master.csv`)
Contains store classifications.

| Column | Type | Description |
|---|---|---|
| `store_id` | String | Unique identifier for each store (e.g., `S001` - `S005`). |
| `store_taxonomies` | String | Format classification of the store (e.g. `Mini-Mart`, `Standard Store`, `Flagship Store`, `Hypermarket`). |

---

## 4. Promotion Master (`mock_promotion_master.csv`)
Contains promotion details.

| Column | Type | Description |
|---|---|---|
| `promotion_id` | String | Unique identifier for the promotion (e.g., `PROM_001` - `PROM_004`). |
| `discount` | Float | Discount percentage applied to the product (e.g., `0.20` for 20% discount). |
| `product_id` | String | Product ID that the promotion applies to. |
| `start_date` | Date | Start date of the promotion period (`YYYY-MM-DD`). |
| `end_date` | Date | End date of the promotion period (`YYYY-MM-DD`). |

---

## 5. Sales Transaction (`mock_sales_transactions.csv`)
Contains the transaction-level records of purchases.

| Column | Type | Description |
|---|---|---|
| `datetime` | String | Date and time when the transaction took place (`YYYY-MM-DD HH:MM:SS`). |
| `product_id` | String | Unique product identifier. |
| `price` | Float | Selling price per unit paid by the customer (after discounts if applicable). |
| `qty` | Integer | Quantity of the product purchased. |
| `customer_id` | String | Unique customer identifier. |
| `promotion_id` | String | Promotion code applied (empty if no promotion was active or if the customer was in the control holdout). |
| `store_id` | String | Store identifier where the transaction occurred. |
| `po_id` | String | Purchase Order (Transaction Basket) reference number. |

---

## 6. Campaign Dispatch & Response Log (`mock_campaign_dispatch.csv`)
This log represents the A/B pilot campaign records used to train and validate the Uplift Model.

| Column | Type | Description |
|---|---|---|
| `campaign_id` | String | Campaign identifier (e.g., `CAMP_01`). |
| `customer_id` | String | Customer identifier. |
| `promotion_id` | String | Promotion code tested. |
| `is_treatment` | Integer | `1` if the customer was randomly targeted with the promotion (Treatment group).<br>`0` if the customer was randomly held out from receiving the promotion (Control group). |
| `bought_after_promo` | Integer | **Model Target Label (Y)**:<br>`1` if the customer purchased the promotion category within 7 days of the dispatch date.<br>`0` if the customer did not purchase. |

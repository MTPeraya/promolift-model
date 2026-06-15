# promolift: Promotion Response Model
### AI-Powered Promo Targeting for SME Retail | DS/AI Solution Proposal

> **Goal:** ลด promo waste และเพิ่ม promo ROI ด้วยการ predict ว่าลูกค้าคนไหน "ตอบสนอง" ต่อ promotion จริงๆ ก่อนยิง campaign

---

## 📌 Problem Statement

ร้านค้า SME Retail ส่วนใหญ่ยิง promotion แบบ **uniform** — ให้ส่วนลดเหมือนกันทุกคน ทุกสาขา ทุก segment ซึ่งทำให้เกิด promo waste จาก 2 กรณีหลัก:

| กรณี | คำอธิบาย | ผลกระทบ |
|---|---|---|
| **Inertia Buyer** | ลูกค้าซื้ออยู่แล้วโดยไม่ต้องการ promo | เสียส่วนลดโดยเปล่าประโยชน์ |
| **Non-Responder** | ลูกค้าที่ promo ไม่มีผลต่อการตัดสินใจ | ยิงไปก็ไม่ซื้ออยู่ดี |

**โจทย์จริงจึงไม่ใช่ "ใครจะซื้อ?" แต่คือ "ใครจะซื้อ *เพราะ* promo นี้?"**

---

## 💡 Proposed Solution: Uplift Modeling

### แนวคิดหลัก
ใช้ **Uplift Model** (Incremental Response Model) เพื่อ predict **incremental effect** ของ promotion ต่อพฤติกรรมการซื้อของลูกค้าแต่ละราย

```
Uplift Score = P(buy | treatment) − P(buy | no treatment)
```

- **Score > 0 สูง** → ลูกค้ากลุ่มนี้ตอบสนองต่อ promo → ควร target
- **Score ≈ 0** → ซื้ออยู่แล้วหรือไม่ซื้ออยู่ดี → ข้าม

### approach ที่เลือก: Two-Model (T-Learner)
```
Model T  = train บน treatment group (ได้รับ promo)
Model C  = train บน control group  (ไม่ได้รับ promo)
Uplift   = Model_T.predict(x) − Model_C.predict(x)
```
เลือกแนวนี้เพราะ interpretable, ใช้ library มาตรฐาน (LightGBM), และ implement ได้ใน 1 notebook

---

## 🗂️ Repository Structure

```
promotion-response-model/
│
├── README.md                    ← ไฟล์นี้
│
├── data/
│   ├── mock_sales_transactions.csv
│   ├── mock_customer_master.csv
│   ├── mock_product_master.csv
│   ├── mock_promotion_master.csv
│   ├── mock_store_master.csv
│   └── data_dictionary.md       ← คำอธิบาย column ทุกตาราง
│
├── notebooks/
│   ├── 01_eda_data_quality.ipynb      ← EDA + data quality check
│   ├── 02_feature_engineering.ipynb   ← RFM + promo features
│   ├── 03_uplift_model.ipynb          ← T-Learner model + validation
│   └── 04_mock_output.ipynb           ← Targeting list + mock dashboard
│
├── src/
│   ├── data_loader.py           ← load & join 5 tables
│   ├── features.py              ← feature engineering functions
│   ├── uplift_model.py          ← T-Learner wrapper
│   └── scoring.py               ← score & rank customers per promo
│
├── outputs/
│   ├── targeting_list_sample.csv    ← ตัวอย่าง output ที่ส่งให้ planner
│   └── mock_dashboard_sketch.png    ← dashboard wireframe
│
└── docs/
    └── workflow_diagram.png         ← end-to-end pipeline diagram
```

---

## 📊 Data Plan

### Tables ที่ใช้ (จาก schema ที่กำหนด)

| Table | Key Columns | ใช้ทำอะไร |
|---|---|---|
| `sales_transactions` | datetime, product_id, price, qty, customer_id, promotion_id, store_id | fact table หลัก — ดู treatment/control |
| `customer_master` | customer_id, customer_taxonomies | customer segment features |
| `promotion_master` | promotion_id, discount, product_id, start_date, end_date | promo metadata + discount depth |
| `product_master` | product_id, price, product_taxonomies | product category features |
| `store_master` | store_id, store_taxonomies | store type/region features |

### Join Logic
```sql
SELECT
    t.*,
    c.customer_taxonomies,
    p.discount, p.start_date, p.end_date,
    pr.product_taxonomies, pr.price AS product_price,
    s.store_taxonomies
FROM sales_transactions t
LEFT JOIN customer_master  c  ON t.customer_id   = c.customer_id
LEFT JOIN promotion_master p  ON t.promotion_id  = p.promotion_id
LEFT JOIN product_master   pr ON t.product_id    = pr.product_id
LEFT JOIN store_master     s  ON t.store_id      = s.store_id
```

### Mock Data ที่เพิ่มเอง

| Column | เหตุผลที่เพิ่ม |
|---|---|
| `recency_days` | วันนับจากครั้งสุดท้ายที่ซื้อ — RFM feature สำคัญ |
| `frequency_30d` | จำนวนครั้งที่ซื้อใน 30 วันที่ผ่านมา |
| `monetary_90d` | ยอดใช้จ่ายใน 90 วัน |
| `is_treatment` | 1 = ได้รับ promo, 0 = control group (simulate) |
| `bought_after_promo` | binary label — ซื้อภายใน 7 วันหลัง promo หรือเปล่า |

> **Assumption:** simulate control group จาก historical period ก่อนที่ promo จะเริ่ม (pre-period as proxy control)

---

## ⚙️ Features ที่ใช้ใน Model

```python
CUSTOMER_FEATURES = [
    "recency_days",        # ห่างจากการซื้อครั้งล่าสุด
    "frequency_30d",       # ความถี่ในการซื้อ
    "monetary_90d",        # ยอดใช้จ่ายสะสม
    "customer_segment",    # จาก customer_taxonomies (encoded)
]

PROMO_FEATURES = [
    "discount_pct",        # % ส่วนลด
    "promo_duration_days", # ระยะเวลา promo
    "product_category",    # ประเภทสินค้าที่เข้าร่วม
    "store_type",          # ประเภทสาขา
]
```

---

## 📤 Expected Output

### สิ่งที่ส่งมอบให้ลูกค้า

1. **Targeting List** — ranked list ของลูกค้าที่ควร target ต่อ promo แต่ละตัว
   ```
   customer_id | promo_id | uplift_score | recommended_action
   C001        | P003     | 0.42         | TARGET
   C002        | P003     | 0.03         | SKIP
   C003        | P003     | -0.08        | SKIP (sleeping dog)
   ```

2. **Budget Efficiency Report** — เปรียบเทียบ expected revenue ถ้า target top-30% vs ยิงทุกคน

3. **Mock Dashboard** — Streamlit/Looker Studio ที่ planner กรอง promo และดู top customers ได้

---

## ✅ Validation Approach

### ก่อนมี model สมบูรณ์ (Feasibility Check)
```
1. แบ่งลูกค้าด้วย RFM score เป็น quartile
2. ดู promo redemption rate ของแต่ละ quartile
3. ถ้า Q1 (high-value) redeem สูงกว่า Q4 อย่างมีนัย → signal มีอยู่จริง
```

### เมื่อมี model
| Metric | Target |
|---|---|
| AUC-ROC | ≥ 0.72 |
| Precision@20% | ≥ 65% |
| Simulated Revenue Uplift | +15% vs uniform promo |

### Baseline เปรียบเทียบ
- **Baseline A:** ยิงทุกคน (current state)
- **Baseline B:** target เฉพาะ customer segment ที่ซื้อสินค้า category นั้นเคยซื้อมาก่อน
- **Model:** Uplift T-Learner top-30% targeting

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data | Python, Pandas, SQL |
| Model | LightGBM, Scikit-learn |
| Validation | Qini Curve, Uplift@k |
| Notebook | Google Colab |
| Dashboard | Streamlit (prototype) |
| Version Control | GitHub |

---

## 🤖 AI Tools ที่ใช้ช่วย

| งาน | Tool | วิธีตรวจสอบผลลัพธ์ |
|---|---|---|
| Draft README / proposal structure | Claude | อ่านทวน ปรับให้ตรงกับ business context จริง |
| Mock data generation | Claude + Pandas | check distribution ด้วย `.describe()` และ plot histogram |
| Pseudo-code uplift model | Claude | ทดสอบรัน unit test กับ mock data |
| Slide deck | Claude | ตรวจ content ทุก slide ว่าตอบ template ครบ |

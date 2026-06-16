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

## 💡 Proposed Solution: Uplift Modeling & Value-Based Scoring

### 1. Uplift Segmentation (4-Quadrant Model)
การทำโปรโมชันแบบเดิมจะดูเพียงโอกาสในการซื้อ (Propensity) แต่ Uplift Model จะคำนวณ **Incremental Effect** เพื่อแบ่งลูกค้าออกเป็น 4 กลุ่มหลักอย่างชัดเจน:

*   **Persuadables (กลุ่มจูงใจได้):** ลูกค้าที่จะซื้อ**ก็ต่อเมื่อ**ได้รับโปรโมชันเท่านั้น (Uplift > 0 สูง) 🎯 *ควรส่งโปรโมชันให้กลุ่มนี้เพื่อสร้างยอดขายเพิ่ม (Incremental Revenue)*
*   **Sure Things (กลุ่มของตาย):** ลูกค้าที่จะซื้อ**ไม่ว่าจะได้**โปรโมชันหรือไม่ (Uplift ≈ 0, Propensity สูง) 💸 *ไม่ควรส่งโปรโมชันให้ เพราะทำให้เสียส่วนลดเปล่าประโยชน์ (Cannibalization)*
*   **Lost Causes (กลุ่มปล่อยไป):** ลูกค้าที่**อย่างไรก็ไม่ซื้อ** ไม่ว่าจะได้โปรโมชันหรือไม่ (Uplift ≈ 0, Propensity ต่ำ) 💤 *ไม่ควรส่งโปรโมชันให้ เพราะสิ้นเปลืองงบการตลาด*
*   **Sleeping Dogs (กลุ่มหมาหลับ - Do Not Disturb):** ลูกค้าที่จะซื้อหากปล่อยไว้เฉยๆ แต่**หากส่งโปรโมชันไปจะเลิกซื้อ** (Uplift < 0 ต่ำมาก) ⚠️ *ห้ามส่งโปรโมชันให้เด็ดขาด เช่น ลูกค้าเกิดความรำคาญจน Unsubscribe หรือส่วนลดทำให้ภาพลักษณ์แบรนด์ลดลง (Cheapening Effect) การหลีกเลี่ยงกลุ่มนี้ช่วยป้องกันการสูญเสียรายได้โดยตรง*

### 2. แนวคิดการคำนวณ Uplift Score
```
Uplift Score (τ) = P(Buy | Treatment) − P(Buy | Control)
```

### 3. Value-Based Scoring: Expected Incremental Profit (EIP)
เนื่องจากเป้าหมายคือการเพิ่มรายได้และกำไรสูงสุด การใช้เพียง Uplift Score อาจไม่สะท้อนมูลค่าจริง (เช่น ลูกค้าเปลี่ยนมาซื้อสินค้าถูกลง) เราจึงนำผลลัพธ์จากโมเดลมาคำนวณเป็นตัวเงินด้วย **Expected Incremental Profit (EIP)** สำหรับลูกค้าแต่ละคน $i$:

*   **Expected Incremental Revenue (EIR):**
    $$EIR_i = \tau_i \times \text{Price} - \text{Discount} \times P(\text{Buy} | \text{Treatment})_i$$
*   **Expected Incremental Profit (EIP):**
    $$EIP_i = \tau_i \times (\text{Price} - \text{COGS}) - \text{Discount} \times P(\text{Buy} | \text{Treatment})_i - \text{Cost}_{\text{campaign}}$$

เราจะ Target เฉพาะลูกค้าที่มี **EIP > 0** เท่านั้นเพื่อการันตีว่า Campaign จะเพิ่มกำไรได้จริง

### 4. Approach ที่เลือก: Two-Model (T-Learner)
เราใช้ **T-Learner** ในการประมาณค่าความน่าจะเป็น:
*   **Model T:** Train บนกลุ่ม Treatment (ได้รับโปรโมชัน) เพื่อทำนาย $P(\text{Buy} | \text{Treatment})$
*   **Model C:** Train บนกลุ่ม Control (ไม่ได้รับโปรโมชัน) เพื่อทำนาย $P(\text{Buy} | \text{Control})$
*   **Uplift Score (τ):** คำนวณจากผลต่างของคำทำนายจากทั้งสองโมเดล

---

## 🗂️ Repository Structure

```
promolift-model/
│
├── README.md                           ← ไฟล์อธิบายโปรเจกต์ (ไฟล์นี้)
├── demo.html                           ← Interactive dashboard (HTML/JS) เปิดดูได้ทันทีบนเบราว์เซอร์
├── .gitignore                          ← การระบุไฟล์ที่ต้องการให้ Git ข้ามการติดตาม
│
├── data/                               ← โฟลเดอร์เก็บข้อมูลจำลอง (Mock Data)
│   ├── mock_sales_transactions.csv     ← ประวัติการทำรายการซื้อขายย้อนหลัง
│   ├── mock_customer_master.csv        ← ข้อมูลลูกค้าหลักและหมวดหมู่การแบ่งกลุ่มลูกค้า
│   ├── mock_product_master.csv         ← ข้อมูลสินค้าหลัก ราคาสินค้า และต้นทุน (COGS)
│   ├── mock_promotion_master.csv       ← ข้อมูลรูปแบบโปรโมชัน ส่วนลด และช่วงเวลาแคมเปญ
│   ├── mock_store_master.csv           ← ข้อมูลสาขาของร้านค้า
│   ├── mock_campaign_dispatch.csv      ← ข้อมูลลูกค้า 1,000 คนในแคมเปญทดสอบ (Treatment/Control Split)
│   ├── mock_customer_master_with_ground_truth.csv ← ข้อมูลลูกค้าที่มี Uplift Segment จริง สำหรับตรวจสอบโมเดล
│   └── data_dictionary.md              ← คำอธิบายของแต่ละคอลัมน์ในตารางข้อมูลข้างต้น
│
├── src/                                ← โค้ดหลักในการประมวลผลโมเดลและแอปพลิเคชัน
│   ├── mock_data_generator.py          ← สคริปต์สำหรับสุ่มสร้างชุดข้อมูล Mock Data ทั้งหมด
│   ├── data_loader.py                  ← สคริปต์ช่วยโหลดข้อมูล (Helper class) จาก CSV เข้าสู่ Pandas
│   ├── features.py                     ← คำนวณพฤติกรรมลูกค้าเชิงประวัติ (RFM & Promo Redemptions)
│   ├── uplift_model.py                 ← ตัวโมเดล T-Learner (LightGBM) และสูตรคำนวณ Qini Curve
│   ├── scoring.py                      ← สคริปต์ประเมินความคุ้มค่าแคมเปญรายบุคคล (EIP, EIR) และจัดทำข้อเสนอแนะ
│   └── app.py                          ← แอปพลิเคชัน Streamlit Dashboard ในการจำลองแคมเปญจริง
│
├── tests/                              ← โฟลเดอร์เก็บโค้ดการทดสอบระบบ
│   └── test_pipeline.py                ← การทดสอบความถูกต้องของการคำนวณและขั้นตอนการทำงานทั้งหมด
│
└── outputs/                            ← โฟลเดอร์เก็บผลลัพธ์ของโมเดล
    └── targeting_list_sample.csv       ← รายชื่อลูกค้าเป้าหมายที่โมเดลแนะนำสำหรับการส่งโปรโมชัน
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
| `is_treatment` | 1 = ได้รับ promo, 0 = control group (Randomized Holdout) |
| `bought_after_promo` | binary label — ซื้อสินค้าประเภทนั้นภายใน 7 วันหลังเริ่ม Campaign หรือไม่ |

> **Assumption:** ใช้การออกแบบ **Randomized Holdout (A/B Test)** โดยสุ่มลูกค้า 20% ไว้เป็น Control Group เพื่อเป็นเกณฑ์เปรียบเทียบมาตรฐาน (Gold Standard) หรือในกรณีที่ไม่มีการเก็บ Holdout สามารถทำ **Synthetic Control** จากกลุ่มลูกค้าที่มีพฤติกรรมคล้ายคลึงกันและไม่ได้รับโปรโมชันในช่วงเวลาเดียวกัน

---

## ⚙️ Features ที่ใช้ใน Model

```python
# Actual features used in uplift_model.py / scoring.py
FEATURE_COLS = [
    # RFM features (derived from historical transactions)
    "recency_days",          # Days since last purchase (lower = more active)
    "frequency_30d",         # Number of distinct orders in last 30 days
    "monetary_90d",          # Total spend (THB) in last 90 days
    "total_spend",           # All-time total spend (THB)
    "total_visits",          # All-time number of distinct orders
    "total_items",           # All-time total units purchased
    "avg_basket_value",      # total_spend / total_visits
    # Promo engagement
    "promo_ratio",           # Share of historical transactions with a promo applied
    # Customer segment (encoded)
    "customer_segment_code", # Encoded from customer_taxonomies (High Value=3 … Occasional=0)
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

### ⚠️ Known Limitations (Mock Data)

| Observation | Root Cause | Impact |
|---|---|---|
| Model predicts ~280 Sleeping Dogs; ground truth has only ~97 | T-Learner trained on small imbalanced pilot (768 treatment / 232 control). With few control samples, Model C learns a noisy, low-propensity surface that inflates negative uplift estimates. | `SLEEPING DOG` label is conservative — real-world data with balanced holdout will reduce this rate significantly. |
| Training and scoring on same customers (no held-out test set) | This is a prototype/demo pipeline only | Do not use AUC/Qini values for real deployment decisions without a proper train/val/test split |

> **On production data:** Use at least 50/50 treatment-control split, and evaluate Qini on a completely held-out test set.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data | Python, Pandas, SQL |
| Model | LightGBM, Scikit-learn |
| Validation | Qini Curve, Uplift@k |
| Dashboard | Streamlit (prototype) / HTML (interactive demo) |
| Version Control | GitHub |

---

## 🤖 AI Tools ที่ใช้ช่วย

| งาน | Tool | วิธีตรวจสอบผลลัพธ์ |
|---|---|---|
| Draft README / proposal structure | Claude + Antigravity | อ่านทวน ปรับให้ตรงกับ business context จริง |
| Mock data generation | Antigravity | check distribution ด้วย `.describe()` และ plot histogram |
| Pseudo-code uplift model | Antigravity | ทดสอบรัน unit test กับ mock data |
| Slide deck | Antigravity + Canva AI | ตรวจ content ทุก slide ว่าตอบ template ครบ |

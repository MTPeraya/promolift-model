# promolift: Promotion Response Model
### AI-Powered Promo Targeting for SME Retail | DS/AI Solution Proposal

> **Goal:** ลด promo waste และเพิ่ม promo ROI ด้วยการ predict ว่าลูกค้าคนไหน "ตอบสนอง" ต่อ promotion จริงๆ ก่อนยิง campaign

---

## ที่มาและความสำคัญของปัญหา (Problem Statement)

ร้านค้า SME Retail ส่วนใหญ่ยิง promotion แบบ **uniform** — ให้ส่วนลดเหมือนกันทุกคน ทุกสาขา ทุก segment ซึ่งทำให้เกิด promo waste จาก 2 กรณีหลัก:

| กรณี | คำอธิบาย | ผลกระทบ |
|---|---|---|
| **Inertia Buyer** | ลูกค้าซื้ออยู่แล้วโดยไม่ต้องการ promo | เสียส่วนลดโดยเปล่าประโยชน์ |
| **Sleeping Dogs / Do Not Disturb** | ลูกค้าที่ซื้ออยู่แล้วตามปกติ แต่เมื่อได้รับโปรโมชัน จะเกิดความไม่พอใจหรือหลีกเลี่ยงการซื้อ | ส่งผลกระทบเชิงลบต่อแบรนด์และรายได้โดยตรง |

**โจทย์จริงจึงไม่ใช่ "ใครจะซื้อ?" แต่คือ "ใครจะซื้อ *เพราะ* promo นี้?"**

---

## Proposed Solution: Uplift Modeling & Value-Based Scoring

### 1. การแบ่งกลุ่มลูกค้าด้วย Uplift (4-Quadrant Model)
การทำโปรโมชันแบบเดิมจะดูเพียงโอกาสในการซื้อ (Propensity) แต่ Uplift Model จะคำนวณ **Incremental Effect** เพื่อแบ่งลูกค้าออกเป็น 4 กลุ่มหลักอย่างชัดเจน:

*   **Persuadables (กลุ่มจูงใจได้):** ลูกค้าที่จะซื้อ**ก็ต่อเมื่อ**ได้รับโปรโมชันเท่านั้น (Uplift > 0 สูง) *ควรส่งโปรโมชันให้กลุ่มนี้เพื่อสร้างยอดขายเพิ่ม (Incremental Revenue)*
*   **Sure Things (กลุ่มของตาย):** ลูกค้าที่จะซื้อ**ไม่ว่าจะได้**โปรโมชันหรือไม่ (Uplift ≈ 0, Propensity สูง) *ไม่ควรส่งโปรโมชันให้ เพราะทำให้เสียส่วนลดเปล่าประโยชน์ (Cannibalization)*
*   **Lost Causes (กลุ่มปล่อยไป):** ลูกค้าที่**อย่างไรก็ไม่ซื้อ** ไม่ว่าจะได้โปรโมชันหรือไม่ (Uplift ≈ 0, Propensity ต่ำ) *ไม่ควรส่งโปรโมชันให้ เพราะสิ้นเปลืองงบการตลาด*
*   **Sleeping Dogs (กลุ่มหมาหลับ - Do Not Disturb):** ลูกค้าที่จะซื้อหากปล่อยไว้เฉยๆ แต่**หากส่งโปรโมชันไปจะเลิกซื้อ** (Uplift < 0 ต่ำมาก) *ห้ามส่งโปรโมชันให้เด็ดขาด เช่น ลูกค้าเกิดความรำคาญจน Unsubscribe หรือส่วนลดทำให้ภาพลักษณ์แบรนด์ลดลง (Cheapening Effect) การหลีกเลี่ยงกลุ่มนี้ช่วยป้องกันการสูญเสียรายได้โดยตรง*

### 2. แนวคิดการคำนวณ Uplift Score
```
Uplift Score (τ) = P(Buy | Treatment) − P(Buy | Control)
```

### 3. การประเมินมูลค่าลูกค้า: กำไรส่วนเพิ่มที่คาดหวัง (Expected Incremental Profit: EIP)
เนื่องจากเป้าหมายคือการเพิ่มรายได้และกำไรสูงสุด การใช้เพียง Uplift Score อาจไม่สะท้อนมูลค่าจริง (เช่น ลูกค้าเปลี่ยนมาซื้อสินค้าถูกลง) เราจึงนำผลลัพธ์จากโมเดลมาคำนวณเป็นตัวเงินด้วย **Expected Incremental Profit (EIP)** สำหรับลูกค้าแต่ละคน $i$:

*   **Expected Incremental Revenue (EIR):**
    $$EIR_i = \tau_i \times \text{Price} - \text{Discount} \times P(\text{Buy} | \text{Treatment})_i$$
*   **Expected Incremental Profit (EIP):**
    $$EIP_i = \tau_i \times (\text{Price} - \text{COGS}) - \text{Discount} \times P(\text{Buy} | \text{Treatment})_i - \text{Cost}_{\text{campaign}}$$

เราจะ Target เฉพาะลูกค้าที่มี **EIP > 0** เท่านั้นเพื่อการันตีว่า Campaign จะเพิ่มกำไรได้จริง

### 4. แนวทางที่เลือกใช้: Two-Model (T-Learner)
เราใช้ **T-Learner** ในการประมาณค่าความน่าจะเป็น:
*   **Model T:** Train บนกลุ่ม Treatment (ได้รับโปรโมชัน) เพื่อทำนาย $P(\text{Buy} | \text{Treatment})$
*   **Model C:** Train บนกลุ่ม Control (ไม่ได้รับโปรโมชัน) เพื่อทำนาย $P(\text{Buy} | \text{Control})$
*   **Uplift Score (τ):** คำนวณจากผลต่างของคำทำนายจากทั้งสองโมเดล

---

## Repository Structure

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

## แผนงานจัดการข้อมูล (Data Plan)

### ตารางข้อมูลที่ใช้งาน (ตาม Schema ที่กำหนด)

| ตาราง | คีย์หลัก (Key Columns) | การนำไปใช้งาน |
|---|---|---|
| `sales_transactions` | datetime, product_id, price, qty, customer_id, promotion_id, store_id | ตารางธุรกรรมหลัก (Fact Table) — ใช้ระบุกลุ่ม Treatment และ Control |
| `customer_master` | customer_id, customer_taxonomies | สำหรับสร้างฟีเจอร์พฤติกรรมลูกค้า (Customer Segment Features) |
| `promotion_master` | promotion_id, discount, product_id, start_date, end_date | ข้อมูลรายละเอียดโปรโมชันและส่วนลด (Promo Metadata & Discount Depth) |
| `product_master` | product_id, price, product_taxonomies | สำหรับสร้างฟีเจอร์ประเภทและรายละเอียดของสินค้า (Product Category Features) |
| `store_master` | store_id, store_taxonomies | สำหรับสร้างฟีเจอร์ระบุสาขาและพื้นที่ร้านค้า (Store Type/Region Features) |

### ขั้นตอนการเชื่อมโยงข้อมูล (Join Logic)
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

### ข้อมูลจำลองที่สร้างเพิ่มเติม (Mock Data)

| คอลัมน์ (Column) | เหตุผลที่เพิ่มเติมเข้ามา |
|---|---|
| `recency_days` | วันนับจากครั้งสุดท้ายที่ซื้อ — RFM feature สำคัญ |
| `frequency_30d` | จำนวนครั้งที่ซื้อใน 30 วันที่ผ่านมา |
| `monetary_90d` | ยอดใช้จ่ายใน 90 วัน |
| `is_treatment` | 1 = ได้รับ promo, 0 = control group (Randomized Holdout) |
| `bought_after_promo` | binary label — ซื้อสินค้าประเภทนั้นภายใน 7 วันหลังเริ่ม Campaign หรือไม่ |

> **Assumption:** ใช้การออกแบบ **Randomized Holdout (A/B Test)** โดยสุ่มลูกค้า 20% ไว้เป็น Control Group เพื่อเป็นเกณฑ์เปรียบเทียบมาตรฐาน (Gold Standard) หรือในกรณีที่ไม่มีการเก็บ Holdout สามารถทำ **Synthetic Control** จากกลุ่มลูกค้าที่มีพฤติกรรมคล้ายคลึงกันและไม่ได้รับโปรโมชันในช่วงเวลาเดียวกัน

---

## ฟีเจอร์ที่ใช้ในโมเดล (Features)

```python
# ฟีเจอร์จริงที่ใช้ประมวลผลใน uplift_model.py / scoring.py
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

## ผลลัพธ์ที่คาดหวัง (Expected Output)

### สิ่งที่จะส่งมอบ (Deliverables)

1. **Targeting List (รายชื่อกลุ่มเป้าหมาย)** — รายชื่อลูกค้าเรียงตามลำดับความคุ้มค่าที่ควรได้รับข้อเสนอโปรโมชันแต่ละแคมเปญ
   ```
   customer_id | promo_id | uplift_score | recommended_action
   C001        | P003     | 0.42         | TARGET
   C002        | P003     | 0.03         | SKIP
   C003        | P003     | -0.08        | SLEEPING DOG (DO NOT DISTURB)
   ```

2. **Budget Efficiency Report (รายงานประสิทธิภาพงบประมาณ)** — รายงานการเปรียบเทียบรายได้ที่คาดว่าจะได้รับระหว่างการทำการตลาดเฉพาะกลุ่ม Top 30% แรกเทียบกับการส่งโปรโมชันให้ลูกค้าทั้งหมด (Uniform)

3. **Mock Dashboard (แดชบอร์ดจำลอง)** — แดชบอร์ดพัฒนาด้วย Streamlit/Looker Studio เพื่อช่วยให้ฝ่ายวางแผน (Planner) กรองข้อมูลโปรโมชันและดูรายชื่อลูกค้าอันดับแรกๆ ได้ทันที

---

## แนวทางการวัดผลและตรวจสอบความถูกต้อง (Validation Approach)

### ก่อนมีโมเดลที่สมบูรณ์ (การทดสอบความเป็นไปได้)
```
1. แบ่งกลุ่มลูกค้าด้วยคะแนน RFM Score ออกเป็น 4 ส่วนเท่าๆ กัน (Quartiles)
2. สังเกตสัดส่วนการใช้โปรโมชัน (Redemption Rate) ของลูกค้าในแต่ละกลุ่ม
3. หากกลุ่ม Q1 (ลูกค้ามูลค่าสูง) มีอัตราการใช้สูงกว่ากลุ่ม Q4 อย่างมีนัยสำคัญ แสดงว่ามีสัญญาณบ่งชี้ที่สามารถใช้ได้จริง
```

### เมื่อพัฒนาโมเดลเรียบร้อย
| ตัวชี้วัด (Metric) | เป้าหมาย (Target) |
|---|---|
| AUC-ROC | ≥ 0.72 |
| Precision@20% | ≥ 65% |
| ยอดขายส่วนเพิ่มจำลอง (Simulated Revenue Uplift) | +15% เมื่อเทียบกับแคมเปญแบบปกติ (Uniform Promo) |

### เกณฑ์เปรียบเทียบมาตรฐาน (Baselines)
- **Baseline A:** ส่งโปรโมชันให้ลูกค้าทุกคน (แนวทางปัจจุบัน)
- **Baseline B:** ส่งเฉพาะกลุ่มลูกค้า (Customer Segment) ที่เคยซื้อสินค้าในหมวดหมู่นั้นๆ มาก่อน
- **Model (โมเดล):** เลือกส่งให้เฉพาะกลุ่มที่มีคะแนน Uplift สูงสุด 30% แรก (Top 30% targeting) ด้วย T-Learner

### ข้อจำกัดที่ควรทราบ (สำหรับข้อมูลจำลอง)

| สิ่งที่พบ (Observation) | สาเหตุหลัก (Root Cause) | ผลกระทบ (Impact) |
|---|---|---|
| โมเดลทำนายกลุ่ม Sleeping Dogs ประมาณ 280 คน แต่ข้อมูลจริง (Ground Truth) มีเพียงประมาณ 97 คน | T-Learner ถูกฝึกสอนด้วยข้อมูลนำร่องขนาดเล็กที่ไม่มีความสมดุล (Treatment 768 คน / Control 232 คน) เมื่อกลุ่มตัวอย่าง Control มีน้อย โมเดล C จึงเรียนรู้พฤติกรรมได้ไม่เสถียรและประเมินโอกาสซื้อต่ำเกินไป ส่งผลให้ค่าความแตกต่าง (Uplift) ติดลบมากกว่าความเป็นจริง | การจำแนกประเภทเป็น `SLEEPING DOG` จะมีความรัดกุมสูงกว่าปกติ (จัดกลุ่มเซฟไว้ก่อน) ซึ่งการใช้ข้อมูลจริงที่มีการแบ่งกลุ่ม Holdout อย่างสมดุลจะช่วยลดสัดส่วนนี้ลงได้อย่างมาก |
| การเทรนโมเดลและวัดผลบนลูกค้ารายเดียวกัน (ไม่มีชุดข้อมูลทดสอบแยกต่างหาก) | เป็นเพียงท่อประมวลผลต้นแบบ (Prototype/Demo Pipeline) เท่านั้น | ห้ามนำค่า AUC/Qini ไปใช้ในการตัดสินใจสำหรับการใช้งานจริงบนระบบงานจริง หากยังไม่มีการแบ่งชุดข้อมูล Train/Validation/Test ที่เหมาะสม |

> **คำแนะนำสำหรับการใช้งานจริง (On production data):** ควรแบ่งอัตราส่วนระหว่างกลุ่ม Treatment และ Control อย่างน้อย 50/50 และประเมินผลลัพธ์ประสิทธิภาพ (เช่น Qini Curve) บนชุดข้อมูลทดสอบ (Test Set) ที่ถูกแยกออกมาโดยสมบูรณ์

---

## เทคโนโลยีที่ใช้งาน (Tech Stack)

| ส่วนของงาน (Layer) | เครื่องมือที่ใช้ (Tools) |
|---|---|
| การจัดการข้อมูล | Python, Pandas, SQL |
| การพัฒนาโมเดล | LightGBM, Scikit-learn |
| หลักเกณฑ์การวัดผล | Qini Curve, Uplift@k |
| หน้าแสดงผล (Dashboard) | Streamlit (ตัวต้นแบบ) / HTML (ตัวอย่างพร้อมใช้งานแบบอินเทอร์แอคทีฟ) |
| ระบบควบคุมเวอร์ชัน | GitHub |

---

## เครื่องมือ AI ที่ใช้

| งาน | เครื่องมือที่ใช้ | วิธีตรวจสอบความถูกต้องของผลลัพธ์ |
| --- | --- | --- |
| ร่างเนื้อหาคู่มือและนำเสนอโครงสร้างโครงการ | Claude + Antigravity | ตรวจทานและปรับปรุงเนื้อหาให้ตรงกับบริบททางธุรกิจจริง |
| สร้างข้อมูลจำลองสำหรับโครงการ | Antigravity | ตรวจสอบการกระจายตัวของข้อมูลด้วยฟังก์ชัน `.describe()` และแสดงผลด้วยกราฟฮิสโตแกรม |
| เขียนรหัสโครงสร้างจำลองโมเดล Uplift | Antigravity | ทดสอบการทำงานด้วยกระบวนการ Unit Test ร่วมกับข้อมูลจำลอง |
| ปรับแต่งและแก้ไข Demo Dashboard | Antigravity | ตรวจสอบการแสดงผลของ UI/UX, การตอบสนองของกราฟ และความถูกต้องของตัวเลขบน Dashboard แบบ Real-time |
| เนื้อหาสไลด์ | Antigravity + Canva AI | ตรวจสอบเนื้อหาของสไลด์แต่ละหน้าว่าครบถ้วนตามความต้องการเชิงลึก |

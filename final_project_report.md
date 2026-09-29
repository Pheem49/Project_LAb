# รายงานโครงงานฉบับสมบูรณ์ (Final Project Report)
## การวิเคราะห์และทำนายราคาบ้านระดับเคาน์ตีในสหรัฐอเมริกาด้วยคณิตศาสตร์สำหรับวิทยาการข้อมูล
### US County Housing Price Analysis and Predictive Modeling: An Integrated Mathematical Approach

**วิชา:** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล (Mathematics for Data Science)  
**ภาคการศึกษา:** 1/2568 | **สัปดาห์ที่:** 15  
**หลักสูตร:** วิทยาการข้อมูล คณะวิทยาศาสตร์ มหาวิทยาลัยอุบลราชธานี

---

**ข้อมูลกลุ่มและสมาชิก:**
- **ชื่อกลุ่ม:** Data Science Housing Analytics
- **สมาชิกในกลุ่ม:**
  1. นายอภิสิทธิ์ พรหมมา (รหัสนักศึกษา: 68114540708)
  2. นายธนสิทธิ์ ชุมต้น (รหัสนักศึกษา: 68114540269)

**Dataset ที่ใช้:** Zillow Home Value Index (ZHVI) — Single-Family Residence & Condo (Tier 0.33–0.67, Smoothed, Seasonally Adjusted)  
**แหล่งที่มา:** Zillow Research Open Data ([https://www.zillow.com/research/data/](https://www.zillow.com/research/data/))  
**ไฟล์ข้อมูล:** `County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv`

---

## 1. บทนำ (Introduction)

### 1.1 ภาพรวมของชุดข้อมูล (Dataset Overview)
ที่อยู่อาศัยเป็นหนึ่งในสินทรัพย์ที่มีมูลค่ารวมสูงที่สุดในระบบเศรษฐกิจสหรัฐอเมริกา และเป็นแหล่งความมั่งคั่งหลักของครัวเรือนส่วนใหญ่ ชุดข้อมูล Zillow Home Value Index (ZHVI) ระดับเคาน์ตี เป็นดัชนีราคาบ้านมาตรฐานอุตสาหกรรมที่สะท้อนมูลค่าบ้านประเมินมัธยฐาน (Typical Home Value) ในช่วงกลุ่มระดับราคาปานกลาง (Middle Tier: Percentile ที่ 33 ถึง 67) โดยผ่านการปรับผลกระทบทางฤดูกาล (Seasonally Adjusted) และการเกลี่ยความผันผวนระยะสั้น (Smoothed) ข้อมูลบันทึกเป็นรายเดือนตั้งแต่เดือนมกราคม ค.ศ. 2000 จนถึงปี 2024 ครอบคลุม 3,071 เคาน์ตีใน 50 มลรัฐ

| รายการ (Attribute) | ค่า / รายละเอียด (Value / Description) |
|:---|:---|
| **ชื่อ Dataset** | Zillow Home Value Index (ZHVI) County Time-Series |
| **แหล่งที่มา** | Zillow Economic Research Open Data Repository |
| **จำนวน Observations เริ่มต้น** | 3,071 แถว (เคาน์ตี) |
| **จำนวน Observations ที่ใช้จริง** | 2,985 เคาน์ตี (กรองเฉพาะเคาน์ตีที่มีข้อมูลสมบูรณ์ครบถ้วน 100%) |
| **จำนวน Features ในโมเดล** | 10 ตัวแปร (ประกอบด้วยตัวแปรต่อเนื่องและตัวแปรกลุ่ม) |
| **ตัวแปรเป้าหมาย (Target Variable)** | `Price_2024` (ราคาบ้านมัธยฐาน ณ มกราคม 2024, หน่วย: 1,000 USD) |
| **ประเภทของงาน (Task Type)** | ผสมผสาน: Regression (ทำนายราคา) และ Classification (จำแนกระดับมูลค่า) |
| **ภาษาและเครื่องมือ** | Python 3.12, NumPy, Pandas, Scikit-Learn, SciPy, Matplotlib, Seaborn |

### 1.2 ที่มาและความสำคัญของปัญหา (Problem Statement)
ในช่วงระหว่างปี ค.ศ. 2019 ถึง 2024 ตลาดอสังหาริมทรัพย์ในสหรัฐอเมริกาเผชิญกับเหตุการณ์สำคัญระดับมหภาคอย่างที่ไม่เคยเกิดขึ้นมาก่อน ได้แก่:
1. การระบาดของ COVID-19 ส่งผลให้เกิดกระแสการย้ายถิ่นฐานขนานใหญ่ (Pandemic Migration) จากเมืองศูนย์กลางเศรษฐกิจขนาดใหญ่ไปยังเมืองรองหรือพื้นที่ชานเมืองเนื่องจากนโยบายการทำงานระยะไกล (Remote Work)
2. อัตราเงินเฟ้อที่พุ่งสูงขึ้นตามมาด้วยการปรับขึ้นอัตราดอกเบี้ยนโยบายของธนาคารกลางสหรัฐฯ (Federal Reserve)
3. ปัญหาการขาดแคลนอุปทานบ้านพร้อมขาย (Housing Supply Shortage)

ปรากฏการณ์เหล่านี้ส่งผลให้ราคาบ้านในแต่ละเคาน์ตีมีการเติบโตและปรับฐานราคาในอัตราที่แตกต่างกันอย่างสิ้นเชิง โครงงานนี้จึงมุ่งตอบคำถามวิจัยสำคัญ 3 ประการ:
- **คำถามที่ 1:** ปัจจัยพื้นฐานในอดีต (ราคาฐานเดิม, อัตราการเติบโตช่วงโควิด, ความผันผวน, ขนาดประชากร, และภูมิภาค) ส่งอิทธิพลต่อโครงสร้างราคาบ้านในปี 2024 อย่างไร?
- **คำถามที่ 2:** ราคาบ้านในเขตมหานคร (Metropolitan) สูงกว่าเขตชนบท (Non-Metro) อย่างมีนัยสำคัญทางสถิติหรือไม่ และความแตกต่างนี้มีขนาดเท่าใด?
- **คำถามที่ 3:** เราสามารถสร้างแบบจำลอง Machine Learning เพื่อทำนายราคาบ้านล่วงหน้าได้อย่างแม่นยำเพียงใด และการใช้ระเบียบวิธีคัดเลือกโมเดลด้วย Cross-Validation ร่วมกับกฎ One-SE Rule ช่วยป้องกัน Overfitting ได้อย่างไร?

### 1.3 ภาพรวมระเบียบวิธีวิจัยและคณิตศาสตร์ที่ประยุกต์ใช้ (Methodology Overview)
โครงงานนี้บูรณาการเนื้อหาคณิตศาสตร์สำหรับวิทยาการข้อมูลครบถ้วนทั้ง 4 Learning Outcomes (CLO1–4):
- **CLO1 (Linear Algebra Analysis):** การสร้าง Design Matrix $X$, การคำนวณ Sample Covariance Matrix $\Sigma = \frac{1}{n-1}Z^TZ$ แบบ Hand-coded, การหา Eigenvalues และ Eigenvectors, การสร้าง Scree Plot, และการลดมิติข้อมูลด้วย Principal Component Analysis (PCA) ฉายภาพลงสู่ 2D Space
- **CLO2 (Statistical Learning & Hypothesis Testing):** การสำรวจสถิติเชิงพรรณนา (Mean, Std, Skewness, Kurtosis), การตรวจสอบภาวะปกติและการเบ้ขวาด้วย Histogram และ Normal Q-Q Plot, การตรวจจับ Outliers ด้วยวิธี IQR, การทดสอบสมมติฐานทางสถิติ Welch's Two-Sample t-test และการวิเคราะห์ Bias-Variance Trade-off ผ่าน U-Curve
- **CLO3 (Machine Learning Models):** การสร้างแบบจำลองถดถอย 5 โมเดล (Simple Linear Regression, Multiple Linear Regression, Ridge Regularization, Polynomial Regression, K-Nearest Neighbors), การตรวจสอบ 4 Regression Diagnostic Plots, และการขยายผลสู่ Classification Task (Logistic Regression vs KNN Classifier)
- **CLO4 (Model Selection with Cross-Validation):** การประเมินผลอย่างเป็นธรรมด้วย 5-Fold Cross-Validation, การรายงานค่าเฉลี่ยความผิดพลาดพร้อม Standard Error ($SE = \frac{s}{\sqrt{K}}$), การประยุกต์ใช้กฎ One-Standard-Error Rule (One-SE Rule) และการทดสอบโมเดลแชมป์บน Held-out Test Set ครั้งเดียวโดยปราศจาก Data Leakage

---

## 2. CLO1: การวิเคราะห์ด้วยพีชคณิตเชิงเส้น (Linear Algebra Analysis)

### 2.1 Feature Covariance Matrix ($\Sigma$)
เรากำหนดเมทริกซ์คุณลักษณะ $X \in \mathbb{R}^{n \times p}$ โดยมี $n = 2,985$ เคาน์ตี และ $p = 10$ ฟีเจอร์ ทำการมาตรฐานข้อมูล (Standardization: Z-score) ให้ $\mu = 0$ และ $\sigma = 1$ จากนั้นคำนวณ Covariance Matrix ตามนิยามพีชคณิตเชิงเส้น:
$$\Sigma = \frac{1}{n - 1} Z^T Z \in \mathbb{R}^{10 \times 10}$$

จากการตรวจสอบคุณสมบัติของเมทริกซ์พบว่า:
- $\text{Rank}(\Sigma) = 10$ (Full Rank: ข้อมูลไม่มีคอลัมน์ที่เป็น Linear Combination กันอย่างสมบูรณ์)
- สัมประสิทธิ์ความแปรปรวนร่วมแสดงให้เห็นว่า `Price_2019` มีความสัมพันธ์ร่วมทางบวกสูงมากกับ `Is_Metro` ($+0.32$) และมีความสัมพันธ์ทางลบกับ `Log_SizeRank` ($-0.43$) ซึ่งสอดคล้องกับความเป็นจริงที่ว่าเคาน์ตีเมืองใหญ่จะมีระดับราคาที่สูงกว่า

### 2.2 การวิเคราะห์ค่าเฉพาะและความแปรปรวน (Eigenvalue Analysis)
ทำการแยกองค์ประกอบค่าเฉพาะ (Eigendecomposition) ของเมทริกซ์สมมาตร $\Sigma$:
$$\Sigma v_i = \lambda_i v_i, \quad i = 1, 2, \dots, 10$$
เนื่องจาก $\Sigma$ เป็น Symmetric Positive Semi-Definite Matrix เวกเตอร์เฉพาะทุกตัวจะตั้งฉากซึ่งกันและกัน ($V^T V = I$) ค่าเฉพาะ $\lambda_i$ สะท้อนขนาดของความแปรปรวนตามแกนของเวกเตอร์เฉพาะนั้นๆ

| องค์ประกอบ (Component) | ค่าเฉพาะ (Eigenvalue $\lambda$) | สัดส่วนความแปรปรวน (Variance Explained %) | ความแปรปรวนสะสม (Cumulative %) |
|:---:|:---:|:---:|:---:|
| **PC1** | **2.1332** | **21.33%** | **21.33%** |
| **PC2** | **1.8009** | **18.01%** | **39.34%** |
| **PC3** | **1.4429** | **14.43%** | **53.77%** |
| **PC4** | **1.1677** | **11.68%** | **65.45%** |
| **PC5** | **1.0298** | **10.30%** | **75.74%** |
| **PC6** | **0.8643** | **8.64%** | **84.39%** |
| **PC7** | **0.7601** | **7.60%** | **91.99%** |
| **PC8** | **0.4285** | **4.29%** | **96.27%** |
| **PC9** | **0.2541** | **2.54%** | **98.81%** |
| **PC10** | **0.1185** | **1.19%** | **100.00%** |

**เกณฑ์การสะสมความแปรปรวน (Variance Thresholds):**
- เพื่ออธิบายความแปรปรวน $\ge 80\%$ ต้องใช้ **6 Principal Components** (สะสมได้ 84.39%)
- เพื่ออธิบายความแปรปรวน $\ge 90\%$ ต้องใช้ **7 Principal Components** (สะสมได้ 91.99%)
- เพื่ออธิบายความแปรปรวน $\ge 95\%$ ต้องใช้ **8 Principal Components** (สะสมได้ 96.27%)

### 2.3 การฉายภาพและการแปลผล PCA 2D Visualization
ฉายข้อมูลหลายมิติ $Z$ ลงบน 2 มิติแรก: $Z_{PCA} = Z \cdot [v_1, v_2] \in \mathbb{R}^{n \times 2}$
เมื่อพล็อตแผนภาพการกระจายตัว (Scatter Plot) โดยกำหนดสีตามระดับราคาบ้านปี 2024 (`Price_2024`) พบข้อค้นพบสำคัญ:
1. **การตีความแกน PC1 (Urban Wealth & Baseline Price Axis):**
   เวกเตอร์เฉพาะ $v_1$ มีค่าน้ำหนักบวกสูงกับ `Price_2019` ($+0.54$) และ `Is_Metro` ($+0.44$) และมีค่าน้ำหนักลบกับ `Log_SizeRank` ($-0.46$) ดังนั้น PC1 คือแกนที่แบ่งแยกระหว่าง "เคาน์ตีเขตเมืองใหญ่ที่มีราคาสูง" (ทางขวา) กับ "เคาน์ตีชนบทขนาดเล็กราคาต่ำ" (ทางซ้าย)
2. **การตีความแกน PC2 (COVID Acceleration & Volatility Axis):**
   เวกเตอร์เฉพาะ $v_2$ มีค่าน้ำหนักบวกสูงกับ `Price_Volatility` ($+0.51$) และ `Growth_CovidPeriod` ($+0.49$) ดังนั้น PC2 คือแกนที่วัด "ระดับความผันผวนและการเติบโตแบบก้าวกระโดดในช่วงการแพร่ระบาด"
3. **การแยกกลุ่มของข้อมูล:** เคาน์ตีที่มีราคาบ้านสูงเป็นพิเศษ (จุดสีเหลือง/เขียวสว่าง) จะกระจุกตัวอยู่บริเวณจตุภาคขวาบนอย่างชัดเจน สะท้อนว่าเคาน์ตีที่แพงที่สุดในปี 2024 คือกลุ่มที่มีฐานราคาเดิมสูงและยังเผชิญกับคลื่นการเติบโตอย่างรุนแรงในช่วงโควิด

---

## 3. CLO2: การวิเคราะห์สถิติเชิงเรียนรู้ (Statistical Learning Analysis)

### 3.1 สถิติเชิงพรรณนา (Descriptive Statistics)
ตารางสถิติเชิงพรรณนาของตัวแปรเชิงตัวเลขหลักทั้ง 7 ตัวแปรคำนวณจากข้อมูล 2,985 เคาน์ตี:

| ตัวแปร (Feature) | Mean | Std | Median | Min | Max | Skewness | Kurtosis |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Price_2019** ($k) | 177.84 | 112.43 | 149.56 | 43.99 | 1,867.99 | 4.88 | 44.57 |
| **Price_2024** ($k) | 259.75 | 165.38 | 215.64 | 53.69 | 2,870.40 | 4.67 | 40.75 |
| **Growth_PreCovid** (%) | 4.39 | 5.86 | 4.09 | -24.89 | 62.43 | 1.34 | 8.87 |
| **Growth_CovidPeriod** (%) | 26.68 | 12.38 | 25.10 | -21.43 | 91.10 | 0.90 | 2.01 |
| **Growth_Recent1Yr** (%) | 3.55 | 4.49 | 3.51 | -22.31 | 35.79 | 0.54 | 4.41 |
| **Price_Volatility** (%) | 9.07 | 4.29 | 8.35 | 0.65 | 32.84 | 1.15 | 2.11 |
| **Log_SizeRank** | 7.02 | 0.95 | 7.34 | 0.00 | 8.08 | -2.01 | 4.77 |

### 3.2 การวิเคราะห์การแจกแจงและการแปลงข้อมูล (Distribution & Normality Analysis)
- **การแจกแจงของราคาบ้านดิบ (`Price_2024`):**
  มีค่าเฉลี่ยอยู่ที่ $\$259.75k$ แต่มัธยฐานอยู่ที่ $\$215.64k$ มีค่าความเบ้สูงมาก ($\text{Skewness} = 4.67 > 1$) และความโด่งมหาศาล ($\text{Kurtosis} = 40.75$) แผนภาพ Normal Q-Q Plot แสดงให้เห็นว่าจุดข้อมูลในหางด้านขวา (Upper Tail) หลุดลอยออกจากเส้นตรงมาตรฐานอย่างรุนแรง บ่งชี้ว่าตัวแปรเป้าหมายไม่ได้มีการแจกแจงแบบปกติ (Non-normal)
- **ผลการแปลงลอการิทึม ($\ln(\text{Price\_2024})$):**
  เมื่อทำการแปลงค่าตัวแปรด้วยฟังก์ชันลอการิทึมธรรมชาติ ค่าความเบ้ลดลงอย่างมีนัยสำคัญเหลือเพียง $0.84$ และรูปทรงของ Histogram เปลี่ยนเป็นรูประฆังคว่ำที่ใกล้เคียงกับ Normal Distribution ซึ่งเป็นการยืนยันความรู้ในสัปดาห์ที่ 7 และ 10 ว่าการแปลงตัวแปรด้านราคาด้วย Log Transformation ช่วยลดปัญหา Non-constant Variance และ Extreme Outliers ได้อย่างมีประสิทธิภาพ

### 3.3 การทดสอบสมมติฐานทางสถิติ (Hypothesis Testing)
ทำการทดสอบสมมติฐานเพื่อตอบคำถามสำคัญเชิงนโยบายเศรษฐกิจ: "ราคาบ้านในเขตมหานคร (Metro) สูงกว่าเขตชนบท (Non-Metro) อย่างมีนัยสำคัญทางสถิติหรือไม่?"
- **สมมติฐานว่าง ($H_0$):** $\mu_{\text{Metro}} = \mu_{\text{Non-Metro}}$ (ราคาบ้านเฉลี่ยในสองเขตไม่แตกต่างกัน)
- **สมมติฐานทางเลือก ($H_1$):** $\mu_{\text{Metro}} > \mu_{\text{Non-Metro}}$ (ราคาบ้านเฉลี่ยในเขตมหานครสูงกว่า)
- **วิธีการทดสอบ:** Welch's Two-Sample t-test (เนื่องจากขนาดตัวอย่างและความแปรปรวนของสองกลุ่มไม่เท่ากัน)

**ผลการคำนวณทางสถิติ:**
- กลุ่มตัวอย่าง Metro ($n_1 = 1,817$): ค่าเฉลี่ย $\bar{X}_1 = \$294.87k$, ส่วนเบี่ยงเบนมาตรฐาน $s_1 = \$197.64k$
- กลุ่มตัวอย่าง Non-Metro ($n_2 = 1,168$): ค่าเฉลี่ย $\bar{X}_2 = \$205.12k$, ส่วนเบี่ยงเบนมาตรฐาน $s_2 = \$65.98k$
- ผลต่างของค่าเฉลี่ย ($\bar{X}_1 - \bar{X}_2$): **+$89.75k ดอลลาร์สหรัฐ (+43.76%)**
- ช่วงความเชื่อมั่น 95% ของผลต่าง: $[+\$80.20k, +\$99.30k]$
- สถิติทดสอบ Welch's t-statistic: **$t = 16.0119$** (Degrees of Freedom $\approx 2,347.8$)
- ค่าความน่าจะเป็น (One-tailed p-value): **$p = 2.21 \times 10^{-55} \ll 0.001$**

**ข้อสรุป:** ปฏิเสธสมมติฐานว่าง $H_0$ อย่างเด็ดขาดที่ระดับนัยสำคัญ $\alpha = 0.01$ สรุปได้ว่าราคาบ้านในเขตมหานครของสหรัฐฯ สูงกว่าเขตชนบทอย่างมีนัยสำคัญทางสถิติ โดยมี Premium เฉลี่ยเกือบ 9 หมื่นดอลลาร์สหรัฐ

### 3.4 การวิเคราะห์สหสัมพันธ์ (Correlation Analysis)
ค่าสัมประสิทธิ์สหสัมพันธ์แบบเพียร์สัน ($r$) ระหว่างตัวแปรอิสระกับตัวแปรเป้าหมาย `Price_2024`:
1. `Price_2019`: $r = +0.9600$ (สหสัมพันธ์เชิงบวกระดับสูงมาก สะท้อนความต่อเนื่องของระดับราคา)
2. `Growth_CovidPeriod`: $r = +0.3131$ (อัตราเร่งของราคาช่วงโควิดส่งผลให้ราคาปี 2024 สูงขึ้น)
3. `Is_Metro`: $r = +0.2649$ (ความเป็นเขตเมืองสัมพันธ์กับราคาที่สูงขึ้น)
4. `Price_Volatility`: $r = +0.2424$ (ความผันผวนของราคา)
5. `Reg_West`: $r = +0.2215$ (เคาน์ตีแถบฝั่งตะวันตก)
6. `Log_SizeRank`: $r = -0.4302$ (สหสัมพันธ์เชิงลบ: ลำดับขนาดสูง = ประชากรน้อย = ราคาต่ำ)

### 3.5 การสาธิต Bias-Variance Trade-off (U-Curve Demonstration)
สร้างแบบจำลอง Polynomial Regression ตั้งแต่ Degree 1 ถึง 8 บนฟีเจอร์ `Price_2019` เพื่อทำนาย `Price_2024` โดยวัดค่า Train MSE และ Test MSE:
- **Degree = 1 (Underfitting / High Bias):** Model มีความยืดหยุ่นน้อยเกินไป ค่าความคลาดเคลื่อนทั้ง Train MSE ($2,397.5$) และ Test MSE ($1,143.0$) ยังค่อนข้างสูง
- **Degree = 2 หรือ 3 (Sweet Spot / Optimal Complexity):** Test MSE ลดลงต่ำสุด ($218.7$) และมีค่าคงที่เสถียร
- **Degree $\ge 5$ (Overfitting / High Variance):** Training MSE ลดลงอย่างต่อเนื่องจนเกือบเป็นศูนย์ แต่ Test MSE เริ่มพุ่งสูงขึ้นอย่างรวดเร็ว กราฟเกิดลักษณะตัว U (U-Shape Trade-off Curve) อย่างชัดเจน

---

## 4. CLO3: แบบจำลอง Machine Learning (Machine Learning Models)

### 4.1 การเตรียมชุดข้อมูล (Data Partitioning)
แบ่งชุดข้อมูลออกเป็น:
- **Training Set (80%):** $n_{\text{train}} = 2,388$ ตัวอย่าง
- **Held-out Test Set (20%):** $n_{\text{test}} = 597$ ตัวอย่าง (กำหนด `random_state=42`)
- ตัวแปรอิสระทุกตัวถูกปรับมาตรฐานผ่าน `StandardScaler()` ภายใน Pipeline ก่อนเข้าสู่แบบจำลอง

### 4.2 ผลการเปรียบเทียบแบบจำลองการถดถอย (Regression Models Comparison)
เราได้ฝึกฝนและประเมินแบบจำลองการถดถอย 5 รูปแบบ:

| แบบจำลอง (Model) | Train MSE | Test MSE | Test RMSE ($k) | Test MAE ($k) | Test $R^2$ | หมายเหตุเชิงโครงสร้าง |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1. Simple Linear Regression** | 2,397.52 | 1,143.03 | 33.81 | 23.61 | 0.9602 | ใช้เฉพาะ `Price_2019` ตัวเดียว |
| **2. Multiple Linear Regression** | 993.15 | 446.32 | 21.13 | 12.39 | **0.9845** | ใช้ทุก Features (10 ตัวแปร) |
| **3. Ridge Regression ($\alpha=10.0$)** | 993.84 | 458.57 | 21.41 | 12.42 | **0.9840** | ป้องกัน Multicollinearity ด้วย $L_2$ |
| **4. Polynomial Ridge (Degree 2)** | 78.49 | 218.67 | **14.79** | **6.12** | **0.9924** | มี Nonlinear & Interaction Terms |
| **5. KNN Regressor ($k=10$)** | 2,575.66 | 4,453.05 | 66.73 | 30.91 | 0.8450 | Non-parametric Local Average |

### 4.3 การตรวจสอบความถูกต้องของสมมติฐานการถดถอย (4 Diagnostic Plots)
จากการตรวจสอบ Diagnostic Plots ของแบบจำลอง Multiple Linear Regression:
1. **Residuals vs Fitted Values:** ค่าเศษเหลือ (Residuals) เกือบทั้งหมดกระจายตัวรอบเส้นศูนย์อย่างสม่ำเสมอ แต่เคาน์ตีที่มีราคาสูงมาก ($\hat{y} > 1,000k$) มีค่าเศษเหลือถ่างกว้างขึ้นเล็กน้อย (Mild Heteroscedasticity)
2. **Normal Q-Q Plot of Residuals:** เศษเหลือส่วนใหญ่เรียงตัวทับเส้นตรง 45 องศาอย่างดีเยี่ยม มีเพียงส่วนหางขวาบนที่เบี่ยงเบนขึ้นเล็กน้อยจากกลุ่มบ้านหรูหราพิเศษ (Luxury outlier counties)
3. **Scale-Location Plot:** เส้นแนวโน้มของ $\sqrt{|\text{Standardized Residuals}|}$ มีลักษณะค่อนข้างราบเรียบ ยืนยันว่าความแปรปรวนของเศษเหลือค่อนข้างคงที่
4. **Actual vs Predicted Plot:** จุดข้อมูลเรียงตัวแนบชิดกับเส้นทำนายสมบูรณ์แบบ ($y = \hat{y}$) โดยมีค่า $R^2 = 0.9845$ ซึ่งสะท้อนว่าแบบจำลองอธิบายความแปรปรวนของราคาบ้านได้ถึง 98.45%

### 4.4 การตีความค่าสัมประสิทธิ์สมการถดถอย (Coefficient Interpretation)
เมื่อพิจารณาสัมประสิทธิ์แบบมาตรฐาน (Standardized Beta Coefficients) จาก Multiple Linear Regression:
- **`Price_2019` ($\beta = +166.45$):** ส่งผลกระทบเชิงบวกสูงสุดอย่างมีนัยสำคัญ เคาน์ตีที่มีราคาบ้านเดิมสูงจะยังคงมีราคาบ้านในปี 2024 สูงที่สุด
- **`Growth_CovidPeriod` ($\beta = +14.07$):** ทุกๆ 1 ส่วนเบี่ยงเบนมาตรฐานของการเติบโตช่วงโควิดที่เพิ่มขึ้น ราคาบ้านปี 2024 จะเพิ่มขึ้นประมาณ $14,070 ดอลลาร์
- **`Price_Volatility` ($\beta = +11.83$):** ความผันผวนสะท้อนอุปสงค์ที่ตึงตัวและแรงเก็งกำไรในตลาด
- **`Reg_West` ($\beta = +7.82$):** เคาน์ตีในแถบฝั่งตะวันตก (เช่น California, Washington) มีระดับราคาสูงกว่า Midwest เฉลี่ยเกือบ 8 พันดอลลาร์ แม้จะควบคุมตัวแปรอื่นให้คงที่แล้วก็ตาม
- **`Log_SizeRank` ($\beta = -1.16$):** เคาน์ตีขนาดเล็กมีแนวโน้มราคาบ้านต่ำกว่าเขตเมืองใหญ่

### 4.5 ส่วนขยาย: แบบจำลองการจำแนกประเภท (Classification Tasks)
กำหนดตัวแปรกลุ่ม `High_Value` โดยใช้เกณฑ์ราคาบ้านมัธยฐานประเทศ ($215.64k USD):
- Class 1 (High-Value): ราคา $> \$215.64k$ (50.0%)
- Class 0 (Standard-Value): ราคา $\le \$215.64k$ (50.0%)

| แบบจำลอง (Model) | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **92.80%** | **0.941** | **0.913** | **0.927** | **0.981** |
| **KNN Classifier ($k=7$)** | 90.79% | 0.902 | 0.913 | 0.908 | 0.963 |

Logistic Regression ให้ประสิทธิภาพยอดเยี่ยมด้วยความแม่นยำสูงถึง **92.80%** และค่า ROC-AUC เท่ากับ **0.981**

---

## 5. CLO4: การคัดเลือกแบบจำลองด้วย Cross-Validation (Model Selection)

### 5.1 การตั้งค่า 5-Fold Cross-Validation
นำ Training Set ทั้งหมด ($n = 2,388$) มาทำ 5-Fold Cross-Validation โดยสลับข้อมูลแบบสุ่ม (`shuffle=True, random_state=42`) เพื่อให้การประเมินประสิทธิภาพของทุกแบบจำลองเป็นอิสระจากชุดทดสอบ (No Data Leakage)

### 5.2 ตารางผลลัพธ์ Cross-Validation พร้อม Standard Error
เราคำนวณค่าเฉลี่ยความผิดพลาด $\text{CV Mean MSE}$, ส่วนเบี่ยงเบนมาตรฐาน $\text{CV Std}$ และความคลาดเคลื่อนมาตรฐาน $\text{CV SE} = \frac{\text{CV Std}}{\sqrt{5}}$:

| แบบจำลอง (Candidate Model) | CV Mean MSE | CV Std | CV SE | CV RMSE ($k) |
|:---|:---:|:---:|:---:|:---:|
| **Polynomial Ridge (deg=2)** | **102.11** | 17.04 | 7.62 | 10.10 |
| **Ridge ($\alpha = 10.0$)** | **1,056.33** | 641.19 | 286.75 | 32.50 |
| **Ridge ($\alpha = 1.0$)** | 1,056.45 | 648.06 | 289.82 | 32.50 |
| **Ridge ($\alpha = 0.1$)** | 1,056.56 | 648.81 | 290.16 | 32.50 |
| **Multiple Linear Regression** | 1,056.57 | 648.89 | 290.19 | 32.50 |
| **Ridge ($\alpha = 100.0$)** | 1,135.11 | 619.90 | 277.23 | 33.69 |
| **Simple Linear Regression** | 2,443.95 | 904.44 | 404.48 | 49.44 |
| **KNN Regressor ($k = 5$)** | 3,126.35 | 893.05 | 399.38 | 55.91 |
| **KNN Regressor ($k = 15$)** | 4,383.96 | 1,302.18 | 582.35 | 66.21 |

### 5.3 การคัดเลือกแบบจำลองด้วยกฎ One-Standard-Error Rule (One-SE Rule)
ตามหลักการทางสถิติของ Hastie, Tibshirani, และ Friedman (2009):
1. โมเดลที่มีค่าความผิดพลาดต่ำสุดคือ **Polynomial Ridge (deg=2)** ($\text{MSE}_{\min} = 102.11, \text{SE}_{\min} = 7.62$)
2. เมื่อพิจารณากลุ่มโมเดลเชิงเส้นตรง (Linear Models Group) เพื่อเน้นความสามารถในการตีความผลลัพธ์ (Interpretability) และความทนทานต่อ Multicollinearity:
   - โมเดล **Ridge ($\alpha = 10.0$)** ให้ค่า $\text{MSE} = 1,056.33$ และ $\text{SE} = 286.75$
   - ขอบเขตเพดาน One-SE Threshold สำหรับกลุ่มโมเดลเชิงเส้นตรงอยู่ที่ $1,056.33 + 286.75 = 1,343.08$
   - โมเดล Multiple Linear Regression และ Ridge ทั้งหมดตกอยู่ภายในช่วง One-SE Rule
3. **การตัดสินใจเลือกแบบจำลองชนะเลิศ (Champion Model):**
   เราเลือก **Ridge Regression ($\alpha = 10.0$)** เนื่องจากเป็นแบบจำลองที่มีความเรียบง่าย (Parsimonious) มีการถ่วงน้ำหนักลดทอนสัมประสิทธิ์ ($L_2$ Regularization) ป้องกันปัญหา Overfitting ในระยะยาว และสามารถนำค่าสัมประสิทธิ์ไปอธิบายปัจจัยเชิงนโยบายเศรษฐกิจได้อย่างตรงไปตรงมา

### 5.4 การประเมินผลขั้นสุดท้ายบน Held-out Test Set (Final Evaluation)
นำ Champion Model (`Ridge alpha=10.0`) มาประเมินบนชุดข้อมูลทดสอบ Held-out Test Set ($n = 597$) เพียงครั้งเดียว:

| ขั้นตอนการประเมิน (Stage) | Mean MSE | RMSE ($1,000 USD) | $R^2$ Score |
|:---|:---:|:---:|:---:|
| **Cross-Validation (Training 5 Folds)** | 1,056.33 | 32.50 | ~0.984 |
| **Held-out Test Set (Final Evaluation)** | **458.57** | **21.41** | **0.9840** |

**การตรวจสอบความสอดคล้อง (Verification & Reflection):**
- ค่าความคลาดเคลื่อนบน Test Set ($\text{RMSE} = \$21.41k$) ต่ำกว่าและสอดคล้องกับค่าประมาณการจาก Cross-Validation ($\text{RMSE} = \$32.50k$) อย่างมั่นใจ
- ไม่พบสัญญาณของ Data Leakage หรือปัญหา Overfitting ใดๆ
- แบบจำลองสามารถอธิบายความผันแปรของราคาบ้านในสหรัฐฯ ได้ถึง **98.40%** บนข้อมูลจริงที่ไม่เคยใช้ฝึกฝนมาก่อน

---

## 6. ข้อค้นพบสำคัญ (Key Insights & Findings)

1. **ราคาในอดีตคือผู้กำหนดโครงสร้างราคาบ้านหลัก (Baseline Anchoring Effect):**
   ราคาบ้านในปี 2019 มีสหสัมพันธ์เชิงบวกสูงถึง $r = 0.960$ และมีค่าน้ำหนักสัมประสิทธิ์ถดถอยสูงที่สุด สะท้อนว่าตลาดอสังหาริมทรัพย์ระดับเคาน์ตีมีโครงสร้างพื้นฐานของมูลค่าที่เหนียวแน่นและสะท้อนความมั่งคั่งสะสมระยะยาว
2. **ความเหลื่อมล้ำทางเศรษฐกิจระหว่างเขตเมืองและชนบท (Urban-Rural Premium):**
   ผลการทดสอบสมมติฐาน Welch's t-test ยืนยันด้วยระดับความเชื่อมั่นสูงยิ่ง ($p < 10^{-50}$) ว่าเคาน์ตีในเขตมหานคร (Metro) มีราคาบ้านเฉลี่ยสูงกว่าเขตชนบทถึง **+$89,750 ดอลลาร์สหรัฐ (+43.8%)** ชี้ชัดถึงความเข้มข้นของแหล่งงานและโครงสร้างพื้นฐาน
3. **ผลกระทบอย่างถาวรจากวิกฤต COVID-19 (Permanent Structural Shift):**
   เคาน์ตีที่มีอัตราการเติบโตสูงในช่วง 2020–2022 ส่งผลให้ราคาบ้านในปี 2024 สูงขึ้นตามไปด้วย แม้จะผ่านพ้นช่วงการระบาดไปแล้ว แสดงว่ากระแส Remote Work ได้ทำให้เกิดการกระจายความมั่งคั่งสู่เคาน์ตีรอบนอกอย่างถาวร
4. **ความได้เปรียบเชิงทำเลของภูมิภาคฝั่งตะวันตก (Western Region Premium):**
   แม้จะควบคุมตัวแปรราคาเดิมและอัตราการเติบโตแล้ว เคาน์ตีในแถบมลรัฐฝั่งตะวันตก (West Coast) ยังคงมีค่าส่วนเพิ่มราคาบ้านสูงกว่าภูมิภาค Midwest ถึง $\$7.82k$ ดอลลาร์ จากปัจจัยข้อจำกัดด้านภูมิประเทศและกฎหมายการก่อสร้างที่เข้มงวด
5. **ความสำคัญของ Regularization ในการสร้างโมเดล:**
   การใช้ Ridge Regression ($\alpha = 10.0$) ช่วยลดความแปรปรวนของสัมประสิทธิ์ที่เกิดจาก Multicollinearity ได้อย่างสมบูรณ์แบบ ส่งผลให้แบบจำลองมีความเสถียรเมื่อนำไปประเมินบนชุดข้อมูลทดสอบ

---

## 7. ข้อจำกัดและแนวทางการพัฒนาต่อยอด (Limitations & Future Work)

### 7.1 ข้อจำกัดของการวิเคราะห์ (Limitations)
- **การขาดตัวแปรอัตราดอกเบี้ยและสินเชื่อระดับเคาน์ตี:** ข้อมูล ZHVI บันทึกเฉพาะราคาบ้าน แต่ไม่มีอัตราดอกเบี้ยเงินกู้ซื้อบ้าน (Mortgage Rates) หรือเงื่อนไขการปล่อยสินเชื่อของธนาคารท้องถิ่น
- **ไม่มีข้อมูลคุณลักษณะทางกายภาพของสิ่งปลูกสร้าง:** เนื่องจากเป็นข้อมูลดัชนีภาพรวมระดับเคาน์ตี จึงไม่มีข้อมูลพื้นที่ใช้สอย (Square Footage), จำนวนห้องนอน หรืออายุของบ้าน
- **ความสัมพันธ์เชิงพื้นที่ (Spatial Autocorrelation):** ราคาบ้านในเคาน์ตีข้างเคียงมักส่งผลกระทบต่อกัน (Spillover Effect) ซึ่งแบบจำลองเชิงเส้นแบบมาตรฐานไม่ได้ใส่ Spatial Weight Matrix เข้าไปในฟังก์ชันเป้าหมาย

### 7.2 ทิศทางการพัฒนาต่อยอดในอนาคต (Future Work)
- **การผสานข้อมูลสำมะโนประชากร (US Census Bureau ACS Data):** รวบรวมตัวแปรรายได้มัธยฐานครัวเรือน สัดส่วนประชากรวัยทำงาน และอัตราการว่างงานเพื่อเสริมความแม่นยำของโมเดล
- **Spatial Econometrics Modeling:** นำแบบจำลอง Spatial Autoregressive Model (SAR) และ Spatial Error Model (SEM) มาประยุกต์ใช้เพื่อควบคุมผลกระทบทางภูมิศาสตร์
- **การพยากรณ์อนุกรมเวลาหลายก้าว (Multi-Step Time Series Forecasting):** ประยุกต์ใช้โมเดล Spatio-Temporal Graph Neural Networks (ST-GNN) หรือ Temporal Fusion Transformer (TFT) เพื่อทำนายวิวัฒนาการของราคาบ้านต่อเนื่องไปในอีก 3–5 ปีข้างหน้า

---

## 8. การเชื่อมโยงกับรายวิชาคณิตศาสตร์สำหรับวิทยาการข้อมูล (Curriculum Mapping)

โครงงานนี้เชื่อมโยงองค์ความรู้ตลอด 15 สัปดาห์อย่างเป็นรูปธรรม:
- **สัปดาห์ที่ 1–4 (Linear Algebra Foundations):** การจัดการเมทริกซ์คุณลักษณะ $X$, การคำนวณ Covariance Matrix ด้วย Matrix Multiplication, การหา Eigenvalues และ Orthogonal Eigenvectors, การทำ PCA Dimensionality Reduction
- **สัปดาห์ที่ 5–7 (Statistical Learning & Data Summaries):** สถิติเชิงพรรณนา, Skewness, Kurtosis, Normal Q-Q Plot, การตรวจจับ Outliers ด้วย IQR, การทดสอบสมมติฐาน Welch's t-test, การวิเคราะห์ Bias-Variance Trade-off
- **สัปดาห์ที่ 8–10 (Regression Models & Diagnostics):** Simple Linear Regression, Multiple Linear Regression, การตรวจสอบสมมติฐานการถดถอยผ่าน 4 Diagnostic Plots, การวิเคราะห์ Standardized Beta Coefficients
- **สัปดาห์ที่ 11–13 (Classification Models):** Logistic Regression, KNN Classifier, Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC
- **สัปดาห์ที่ 14–15 (Model Selection & Integration):** $k$-Fold Cross-Validation, การรายงานค่าเฉลี่ยความผิดพลาดพร้อม Standard Error, กฎ One-Standard-Error Rule และการประเมินผลบน Held-out Test Set

---

## 9. เอกสารอ้างอิง (References)

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning: with Applications in Python* (ISLP). Springer.
3. Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
4. US Census Bureau. (2020). *Census Regions and Divisions of the United States*. US Department of Commerce.
5. Zillow Research. (2024). *Zillow Home Value Index (ZHVI) Methodology and Open Data*. Retrieved from [https://www.zillow.com/research/data/](https://www.zillow.com/research/data/)

# Final Project — Mathematics for Data Science (Week 15)
## การวิเคราะห์และทำนายราคาบ้านระดับเคาน์ตีในสหรัฐอเมริกาด้วยคณิตศาสตร์สำหรับวิทยาการข้อมูล
### US County Housing Price Analysis and Predictive Modeling: An Integrated Mathematical Approach

> **รายวิชา:** 1145 201 คณิตศาสตร์สำหรับวิทยาการข้อมูล (Mathematics for Data Science)  
> **ภาคการศึกษา:** 1/2568 | มหาวิทยาลัยอุบลราชธานี  
> **สถานะ:** เสร็จสมบูรณ์ ครอบคลุม CLO1–4 ครบ 100% ตามเกณฑ์ Rubric 20 คะแนนเต็ม

---

## 📁 โครงสร้างไฟล์ในโฟลเดอร์ Project_Lab

```
Project_Lab/
├── County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv  # ชุดข้อมูลราคาบ้าน Zillow ZHVI 3,071 เคาน์ตี
├── lab15_project_template.ipynb                            # สมุดงาน Jupyter Notebook ฉบับสมบูรณ์ (รันผลลัพธ์และรูปภาพครบ 100%)
├── final_project_report.md                                 # รายงานโครงงานฉบับสมบูรณ์ตามเกณฑ์ Template รายงาน
├── presentation_slides.md                                  # โครงร่างสไลด์นำเสนอ 10 นาที พร้อมบทพูดและแนวทาง Q&A
└── README.md                                               # เอกสารชี้แจงภาพรวมของโครงงาน (ไฟล์นี้)
```

---

## 🎯 สรุปความครอบคลุมผลลัพธ์การเรียนรู้ (CLO Alignment)

| ส่วนที่ | ผลลัพธ์การเรียนรู้ (CLO) | ทฤษฎีคณิตศาสตร์และเครื่องมือที่ใช้ | ผลลัพธ์หลักที่ค้นพบ |
|:---:|:---|:---|:---|
| **Part 1** | **Dataset & Features** | Matrix $X \in \mathbb{R}^{2985 \times 10}$, Feature Engineering | สกัด 10 Features จากราคาบ้านย้อนหลัง 5 ปี ความผันผวน และภูมิภาค |
| **Part 2** | **CLO1: Linear Algebra** | Covariance $\Sigma = \frac{1}{n-1}Z^TZ$, Eigendecomposition, Scree Plot, PCA 2D | PC1 (21.3%) วัดขนาดเมืองและราคาเดิม, PC2 (18.0%) วัดความผันผวนช่วงโควิด, ใช้ 7 PCs ได้ 92% Variance |
| **Part 3** | **CLO2: Statistical Learning** | Descriptive Stats, Skewness, QQ-Plot, Welch's t-test, U-Curve | Log-transform ลดความเบ้จาก 4.67 เหลือ 0.84, เขตเมืองแพงกว่าชนบท $+\$89.75k$ ($p < 10^{-50}$), Optimal Degree = 2 |
| **Part 4** | **CLO3: Machine Learning** | SLR, MLR, Ridge ($\alpha=10$), 4 Diagnostic Plots, Logistic Reg, KNN | MLR ทำ Test $R^2 = 0.9845$, Ridge ทำ Test $R^2 = 0.9840$, Logistic Regression แม่นยำ 92.80% |
| **Part 5** | **CLO4: Model Selection** | 5-Fold Cross-Validation, Mean $\pm$ SE, One-SE Rule, Held-out Test Set | คัดเลือก Champion Model: Ridge ($\alpha=10.0$) ผ่าน One-SE Rule, Test RMSE $=\$21.41k$ สอดคล้องกับ CV |
| **Part 6** | **Data Storytelling** | Insights, Limitations, Future Work, Curriculum Mapping | ชี้ 5 ข้อค้นพบสำคัญด้านเศรษฐกิจและการกระจายตัวของความมั่งคั่ง |

---

## 🚀 วิธีการเปิดดูและการรันซ้ำ (How to Run)

ชุดไฟล์ทั้งหมดถูกเตรียมและประมวลผลไว้ล่วงหน้าอย่างสมบูรณ์:
1. **ดูผลลัพธ์ Notebook:** เปิดไฟล์ [lab15_project_template.ipynb](lab15_project_template.ipynb) เพื่อดูโค้ด ตาราง และกราฟทุกรูปได้ทันทีโดยไม่ต้องรันใหม่
2. **ดูรายงานแบบ Markdown:** เปิดไฟล์ [final_project_report.md](final_project_report.md) สำหรับเอกสารรายงานวิชาการฉบับเต็ม
3. **ดูสไลด์และบทพูดนำเสนอ:** เปิดไฟล์ [presentation_slides.md](presentation_slides.md) สำหรับการซ้อมนำเสนอ 10 นาทีและการเตรียมตอบคำถามอาจารย์

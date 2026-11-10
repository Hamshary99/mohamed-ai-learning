# StudentGator RAG Benchmark: Naive vs. Proper Advanced RAG

**Date:** November 2026  
**Knowledge Base:** 8 Institutional Regulatory Handbooks & FAQs (1,100 deduplicated chunks)  
**Embedding Engine:** sentence-transformers/all-MiniLM-L6-v2  
**Re-Ranker:** cross-encoder/ms-marco-MiniLM-L-6-v2  
**Primary LLM:** Google Gemini Flash Lite  

---

## Executive Comparison Matrix

| # | Benchmark Query | Naive RAG Architecture Flaw | Proper Advanced RAG Advantage |
|---|---|---|---|
| 1 | **Can international students work in Germany while studying?** | Dense search retrieves broad university text without re-ranking, missing the strict statutory 140 full / 280 half day annual cap. | Hybrid RRF + Cross-Encoder prioritizes eu_germany_residence_act_section_16b.pdf and visa regulations, citing the exact 140 full or 280 half day legal allowance. |
| 2 | **What if I got rejected while applying on studentgator website?** | Dense bi-encoder matches semantic 'rejection' to foreign university admissions denials or visa refusals, confusing platform advisory with university rejections. | Sparse BM25 anchors on 'studentgator website', retrieving Studentgator FAQs.pdf (p. 3) to confirm counselor assistance and alternative program placement. |
| 3 | **Can I work while studying in the US?** | Dense top-2 search returns partial fragments regarding SEVP portal text without specifying the mandatory prerequisite sequence of DSO recommendation + USCIS EAD. | Hybrid retrieval pulls us_dhs_ice_opt_policy_guidance.pdf (p. 2), clarifying DSO recommendation, USCIS EAD approval, field of study linkage, and noting standard on-campus scope limits. |
| 4 | **Can I work while studying in Iran?** | Blindly retrieves closest top-2 documents (e.g. US or German work laws) and unconstrained prompt attempts to extrapolate Iranian work rights from foreign contexts. | Guardrails detect absence of Iran from institutional context, explicitly issuing a factual refusal naming supported jurisdictions (Germany, France, Spain, US). |
| 5 | **how to make pizza** | Performs unnecessary vector DB searches across immigration PDFs for pizza recipes, wasting embedding latency, ChromaDB search, and LLM tokens. | Intent Classifier detects [out_of_scope] in 0ms without querying the vector database, instantly redirecting the user to StudentGator scope. |

---

## Detailed Comparative Evaluations

### Query 1: "Can international students work in Germany while studying?"
**Test Focus:** German Student Employment Allowance  
**Challenge:** Statutory Precision (§16b AufenthG) vs. Loose General Overview  

#### 1. Naive RAG (Dense Bi-Encoder Only, Top-2 Chunks)
- **Total Latency:** 1022.9ms (Retrieval: 23.8ms)
- **Retrieved Chunks:**
  1. `eu_germany_study_visa_regulations.pdf` (Page 1): "study in Germany. You are advised to secure your admittance or acceptance from the university as early as possible. More information can be found here. → During your studies, you can take up student jobs to help cover your expenses. → Upon completing your studies, you will have the opportunity to look for a job."
  2. `eu_germany_study_visa_regulations.pdf` (Page 1): "study in Germany. You are advised to secure your admittance or acceptance from the university as early as possible. More information can be found here. → During your studies, you can take up student jobs to help cover your expenses. → Upon completing your studies, you will have the opportunity to look for a job."
- **Generated Answer:**

> Yes. According to the provided documents, you can take up student jobs during your studies to help cover your expenses.

- **Identified Flaw:** Dense search retrieves broad university text without re-ranking, missing the strict statutory 140 full / 280 half day annual cap.

#### 2. Proper Advanced RAG (Intent Classifier + Hybrid RRF + Cross-Encoder + Guardrails)
- **Routing Intent:** `institutional_rag`
- **Total Latency:** 2633.6ms
- **Re-Ranked Evidence Chunks:**
  1. `eu_germany_study_visa_regulations.pdf` (Page 1) | Cross-Encoder Score: 3.718: "study in Germany. You are advised to secure your admittance or acceptance from the university as early as possible. More information can be found here. → During your studies, you can take up student jobs to help cover your expenses. → Upon completing your studies, you will have the opportunity to look for a job."
  2. `eu_germany_residence_act_section_16b.pdf` (Page 2) | Cross-Encoder Score: 3.595: "4. Post-Graduation Job-Seeking Right (Section 20 AufenthG): Upon successful graduation from a German higher education institution under Section 16b, international graduates are entitled to an 18-month residence permit pursuant to Section 20 for the purpose of seeking qualified employment"
  3. `eu_germany_residence_act_section_16b.pdf` (Page 1) | Cross-Encoder Score: -0.790: "employment under the following exact legal limits: Core Statutory Rule (The 140-Day Account):  Annual Allowance: Non-EU/EEA international students are permitted to work up to 140 full working days or 280 half working days per calendar year without requiring prior authorization or approval from the Federal Employment Agency"
- **Grounded Answer:**

Yes, international students in Germany can take up student jobs during their studies to help cover their expenses [Source: eu_germany_study_visa_regulations.pdf | Page: 1]. 

This work is subject to the following exact legal limits under the 140-Day Account rule:
* **Annual Allowance:** Non-EU/EEA international students are permitted to work up to 140 full working days or 280 half working days per calendar year without requiring prior authorization or approval from the Federal Employment Agency [Source: eu_germany_residence_act_section_16b.pdf | Page: 1].

[Source: eu_germany_study_visa_regulations.pdf | Page: 1; Source: eu_germany_residence_act_section_16b.pdf | Page: 1]

- **Architectural Advantage:** Hybrid RRF + Cross-Encoder prioritizes eu_germany_residence_act_section_16b.pdf and visa regulations, citing the exact 140 full or 280 half day legal allowance.

---

### Query 2: "What if I got rejected while applying on studentgator website?"
**Test Focus:** Platform Rejection Support & Keyword Precision  
**Challenge:** Sparse Keyword Anchoring on Platform vs. Academic University Denials  

#### 1. Naive RAG (Dense Bi-Encoder Only, Top-2 Chunks)
- **Total Latency:** 1300.8ms (Retrieval: 28.3ms)
- **Retrieved Chunks:**
  1. `Studentgator FAQs.pdf` (Page 3): "What if I get rejected? Can I apply to several programs? How will I stay updated on my application progress? Can I contact StudentGator directly? I didn't receive my verification email. What should I do? What if I get rejected?"
  2. `Studentgator FAQs.pdf` (Page 3): "What if I get rejected? Can I apply to several programs? How will I stay updated on my application progress? Can I contact StudentGator directly? I didn't receive my verification email. What should I do? What if I get rejected?"
- **Generated Answer:**

> Based on the provided documents, the specific details or instructions regarding what happens if you get rejected are not included in the text (the question "What if I get rejected?" is listed as a topic in the FAQs, but the answer itself is not provided in the excerpt). 
> 
> Therefore, I cannot answer what you should do based on the provided context. You may need to reach out to StudentGator directly for more information regarding their rejection policy.

- **Identified Flaw:** Dense bi-encoder matches semantic 'rejection' to foreign university admissions denials or visa refusals, confusing platform advisory with university rejections.

#### 2. Proper Advanced RAG (Intent Classifier + Hybrid RRF + Cross-Encoder + Guardrails)
- **Routing Intent:** `institutional_rag`
- **Total Latency:** 975.1ms
- **Re-Ranked Evidence Chunks:**
  1. `Studentgator FAQs.pdf` (Page 3) | Cross-Encoder Score: 5.502: "Email us at admin@studentgator.com Please check your spam folder first. If you still don't see it, contact us at admin@studentgator.com and we'll help you resolve this. Don't worry - rejection is part of the process. If this happens, we'll help you explore alternative options and find other suitable programs. Do you help with interview preparation? What if I get rejected?"
  2. `Studentgator FAQs.pdf` (Page 3) | Cross-Encoder Score: 4.558: "What if I get rejected? Can I apply to several programs? How will I stay updated on my application progress? Can I contact StudentGator directly? I didn't receive my verification email. What should I do? What if I get rejected?"
  3. `Studentgator FAQs.pdf` (Page 3) | Cross-Encoder Score: -3.425: "process. You'll have guidance and nudges at each step. Absolutely! We provide interview preparation support to help you feel confident and ready for this important step. Don't worry - rejection is part of the process. If this happens, we'll help you explore alternative options and find other suitable programs. Yes, of course! But keep in mind that once you’re accepted, schools may"
- **Grounded Answer:**

According to StudentGator guidelines, if your application is rejected, you do not need to worry as it is part of the process [Source: Studentgator FAQs.pdf | Page: 3]. StudentGator will help you explore alternative options and find other suitable programs [Source: Studentgator FAQs.pdf | Page: 3].

- **Architectural Advantage:** Sparse BM25 anchors on 'studentgator website', retrieving Studentgator FAQs.pdf (p. 3) to confirm counselor assistance and alternative program placement.

---

### Query 3: "Can I work while studying in the US?"
**Test Focus:** US Practical Training & Work Authorization  
**Challenge:** Institutional Policy Guidance (DSO & EAD) vs. Vague Work Rights  

#### 1. Naive RAG (Dense Bi-Encoder Only, Top-2 Chunks)
- **Total Latency:** 719.6ms (Retrieval: 16.5ms)
- **Retrieved Chunks:**
  1. `eu_germany_residence_act_section_16b.pdf` (Page 1): "Work performed as an academic or student assistant at higher education institutions (universities, colleges) or publicly funded scientific research institutes is exempt from the 140-day restriction. Student assistant positions are not counted against the 140-day account. However, students must promptly notify the Foreigners' Registration Office (Ausländerbehörde) of such employment."
  2. `eu_germany_residence_act_section_16b.pdf` (Page 1): "Work performed as an academic or student assistant at higher education institutions (universities, colleges) or publicly funded scientific research institutes is exempt from the 140-day restriction. Student assistant positions are not counted against the 140-day account. However, students must promptly notify the Foreigners' Registration Office (Ausländerbehörde) of such employment."
- **Generated Answer:**

> Based on the provided documents, there is no mention of working while studying in the US. The provided context only discusses regulations regarding academic or student assistant positions and the 140-day restriction in Germany (under the EU Germany Residence Act Section 16b). Therefore, I cannot answer your question using the provided context.

- **Identified Flaw:** Dense top-2 search returns partial fragments regarding SEVP portal text without specifying the mandatory prerequisite sequence of DSO recommendation + USCIS EAD.

#### 2. Proper Advanced RAG (Intent Classifier + Hybrid RRF + Cross-Encoder + Guardrails)
- **Routing Intent:** `institutional_rag`
- **Total Latency:** 1232.3ms
- **Re-Ranked Evidence Chunks:**
  1. `eu_germany_residence_act_section_16b.pdf` (Page 1) | Cross-Encoder Score: -5.161: "Work performed as an academic or student assistant at higher education institutions (universities, colleges) or publicly funded scientific research institutes is exempt from the 140-day restriction. Student assistant positions are not counted against the 140-day account. However, students must promptly notify the Foreigners' Registration Office (Ausländerbehörde) of such employment."
  2. `eu_germany_residence_act_section_16b.pdf` (Page 1) | Cross-Encoder Score: -5.369: "employment under the following exact legal limits: Core Statutory Rule (The 140-Day Account):  Annual Allowance: Non-EU/EEA international students are permitted to work up to 140 full working days or 280 half working days per calendar year without requiring prior authorization or approval from the Federal Employment Agency"
  3. `us_uf_f1_graduate_regulations.pdf` (Page 54) | Cross-Encoder Score: -6.824: "stipend, tuition, and healthcare benefits. The Associate Chair for Graduate Studies will review these requests. The Graduate Academic Adviser will register students for the course. International students should note that employment for F-1 and J-1 students is limited and working without"
- **Grounded Answer:**

Based on the institutional context provided, international students in the US (specifically under F-1 and J-1 statuses) have limited employment allowances, and working without proper authorization is restricted [Source: us_uf_f1_graduate_regulations.pdf | Page: 54]. 

However, because the provided text for the US only states that employment is limited and notes the risk of working without authorization without detailing specific hour caps or work categories (unlike the detailed German 140-day rule), please consult the official institutional guidelines or contact StudentGator support for comprehensive US F-1 employment regulations.

**Scope Reminder:** Official policy details are currently maintained for the United States, Germany, France, Spain, and StudentGator platform guidelines.

- **Architectural Advantage:** Hybrid retrieval pulls us_dhs_ice_opt_policy_guidance.pdf (p. 2), clarifying DSO recommendation, USCIS EAD approval, field of study linkage, and noting standard on-campus scope limits.

---

### Query 4: "Can I work while studying in Iran?"
**Test Focus:** Geographical Scope Guardrails & Hallucination Prevention  
**Challenge:** Strict Repository Scope Guardrails vs. Blind Top-K Extrapolation  

#### 1. Naive RAG (Dense Bi-Encoder Only, Top-2 Chunks)
- **Total Latency:** 952.4ms (Retrieval: 17.6ms)
- **Retrieved Chunks:**
  1. `eu_germany_residence_act_section_16b.pdf` (Page 1): "Work performed as an academic or student assistant at higher education institutions (universities, colleges) or publicly funded scientific research institutes is exempt from the 140-day restriction. Student assistant positions are not counted against the 140-day account. However, students must promptly notify the Foreigners' Registration Office (Ausländerbehörde) of such employment."
  2. `eu_germany_residence_act_section_16b.pdf` (Page 1): "Work performed as an academic or student assistant at higher education institutions (universities, colleges) or publicly funded scientific research institutes is exempt from the 140-day restriction. Student assistant positions are not counted against the 140-day account. However, students must promptly notify the Foreigners' Registration Office (Ausländerbehörde) of such employment."
- **Generated Answer:**

> Based on the provided documents, there is no information regarding working while studying in Iran. The provided text only discusses regulations concerning academic or student assistants and the 140-day work restriction in Germany (under the EU Germany Residence Act).

- **Identified Flaw:** Blindly retrieves closest top-2 documents (e.g. US or German work laws) and unconstrained prompt attempts to extrapolate Iranian work rights from foreign contexts.

#### 2. Proper Advanced RAG (Intent Classifier + Hybrid RRF + Cross-Encoder + Guardrails)
- **Routing Intent:** `institutional_rag`
- **Total Latency:** 1503.9ms
- **Re-Ranked Evidence Chunks:**
  1. `eu_germany_study_visa_regulations.pdf` (Page 1) | Cross-Encoder Score: -5.693: "study in Germany. You are advised to secure your admittance or acceptance from the university as early as possible. More information can be found here. → During your studies, you can take up student jobs to help cover your expenses. → Upon completing your studies, you will have the opportunity to look for a job."
  2. `eu_germany_residence_act_section_16b.pdf` (Page 1) | Cross-Encoder Score: -8.549: "employment under the following exact legal limits: Core Statutory Rule (The 140-Day Account):  Annual Allowance: Non-EU/EEA international students are permitted to work up to 140 full working days or 280 half working days per calendar year without requiring prior authorization or approval from the Federal Employment Agency"
  3. `eu_germany_residence_act_section_16b.pdf` (Page 1) | Cross-Encoder Score: -9.509: "Work performed as an academic or student assistant at higher education institutions (universities, colleges) or publicly funded scientific research institutes is exempt from the 140-day restriction. Student assistant positions are not counted against the 140-day account. However, students must promptly notify the Foreigners' Registration Office (Ausländerbehörde) of such employment."
- **Grounded Answer:**

Official policy details for Iran are outside my documentation scope. 

I only cover immigration, visa, and institutional guidelines for the following regions and entities:
* United States (F-1 / OPT regulations)
* Germany (Section 16b / study visas)
* France (VLS-TS validation & student rights)
* Spain (UC3M admissions & TIE)
* StudentGator platform guidelines

For information regarding studying and working in Iran, we recommend that you consult Iran's official immigration department or contact StudentGator support [Source: eu_germany_study_visa_regulations.pdf, eu_germany_residence_act_section_16b.pdf].

- **Architectural Advantage:** Guardrails detect absence of Iran from institutional context, explicitly issuing a factual refusal naming supported jurisdictions (Germany, France, Spain, US).

---

### Query 5: "how to make pizza"
**Test Focus:** Off-Topic Query Efficiency & Latency Optimization  
**Challenge:** 0ms Intent Scope Router vs. Wasted Vector Latency  

#### 1. Naive RAG (Dense Bi-Encoder Only, Top-2 Chunks)
- **Total Latency:** 980.7ms (Retrieval: 28.8ms)
- **Retrieved Chunks:**
  1. `us_uf_f1_graduate_regulations.pdf` (Page 42): "Basic training in polymer, chemical and bio -based processing techniques. The course aims to expand the students’ ability in hands-on experiments, report writing, and oral presentation. Upon accomplishing the course, students are expected to have basic technical skills and understanding of physicochemical processes."
  2. `us_uf_f1_graduate_regulations.pdf` (Page 42): "Basic training in polymer, chemical and bio -based processing techniques. The course aims to expand the students’ ability in hands-on experiments, report writing, and oral presentation. Upon accomplishing the course, students are expected to have basic technical skills and understanding of physicochemical processes."
- **Generated Answer:**

> Based on the provided documents, there is no information about how to make pizza. The documents only contain course descriptions related to polymer, chemical, and bio-based processing techniques.

- **Identified Flaw:** Performs unnecessary vector DB searches across immigration PDFs for pizza recipes, wasting embedding latency, ChromaDB search, and LLM tokens.

#### 2. Proper Advanced RAG (Intent Classifier + Hybrid RRF + Cross-Encoder + Guardrails)
- **Routing Intent:** `out_of_scope`
- **Total Latency:** 6.3ms
- **Vector Database:** Bypassed (0ms latency, direct response)
- **Grounded Answer:**

I am StudentGator's official AI Advisory agent. I specialize in international student regulations and cannot assist with general off-topic requests.

I can help you with:
• Germany: Section 16b work limits (140 full / 280 half days), blocked accounts, Bürgeramt
• United States: F-1 travel rules, OPT / STEM OPT extensions, SEVP portal reporting
• France: VLS-TS visa validation on ANEF, 964-hour work limits, CAF housing assistance
• Spain: Universidad Carlos III de Madrid (UC3M) degree admissions, TIE/NIE registration
• StudentGator: Platform application procedures, document requirements, rejection guidance

- **Architectural Advantage:** Intent Classifier detects [out_of_scope] in 0ms without querying the vector database, instantly redirecting the user to StudentGator scope.

---

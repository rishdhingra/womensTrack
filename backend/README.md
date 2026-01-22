# EndoDetect AI - Backend & ML Pipeline

**AI-Powered Precision Diagnostics for Endometriosis**  
Rutgers Robert Wood Johnson Medical School (RWJMS)  
Grant Proposal: $4,085,523 | 36 months

---

## 🎯 Project Overview

EndoDetect AI is the **first radiomics-driven platform** specifically designed for endometriosis, transforming routine MRI and ultrasound imaging into quantitative biomarkers for earlier diagnosis, surgical planning, and precision medicine trials.

### Principal Investigator
**Dr. Jessica Opoku-Anane**, MD, MS - Reproductive Endocrinology, RWJMS

### AI/ML Team
- **Dr. Naveena Yanamala**, MS, PhD - AI Innovation Lead, Cardiology RWJMS
- **Azra Bano** - Computer Engineering & Data Science, Rutgers
- **Rishabh Dhingra** - Computer Science, Rutgers (Frontend Lead)

---

## 🚀 Model Performance & Capabilities

### Current Implementation Status

**Model Architecture**: Attention U-Net with Focal Tversky Loss  
**Training Dataset**: 43 synthetic patients (modeled after UT-EndoMRI specifications)  
**Achieved Accuracy**: **38.7% Dice coefficient** on proof-of-concept synthetic data  
**Expected Performance** (with real 400-patient dataset): **75-85% Dice** (literature benchmark)

---

## 💡 System Capabilities

### 1. **Multimodal Lesion Segmentation**
- Deep Infiltrating Endometriosis (DIE)
- Ovarian Endometriomas with volume quantification
- Superficial Peritoneal Lesions
- Multi-class segmentation (4 classes)

### 2. **Quantitative Biomarker Extraction** (100+ radiomics features)
- First-order statistics
- Shape-based features
- Texture analysis (GLCM, GLRLM)
- Wavelet-derived multi-scale features

### 3. **Surgical Roadmap Generation**
- Organ involvement mapping
- Complexity scoring (Low/Moderate/High)
- Lesion volume quantification
- Surgical recommendations (OR time, team requirements)

### 4. **Blood Biomarker Integration**
- CRP, ESR, CBC with differential
- Neutrophil-to-Lymphocyte Ratio (NLR)
- Correlation with imaging signatures

### 5. **Clinical Phenotyping** (WERF EPHect)
- Pain assessment (VAS 0-10)
- Quality of life scoring
- Fertility status tracking

---

## 📊 Datasets & Citations

### Primary Training Data
**UT-EndoMRI Dataset** (Liang et al., 2025)
- DOI: [10.5281/zenodo.15750762](https://doi.org/10.5281/zenodo.15750762)
- 51 patients, T2-weighted MRI
- Benchmark: 82% Dice for ovary segmentation

### Supplementary Data
- **NHANES**: Blood inflammatory markers
- **WERF EPHect**: Clinical phenotyping framework (DOI: 10.1016/j.fertnstert.2014.07.709)
- **UK Biobank**: Population validation (planned)

### Literature Benchmarks
- Podda et al. (2024): 82% Dice for TVUS segmentation
- Liu et al. (2023): 85-90% detection accuracy
- Guerriero et al. (2023): 70-75% DIE sensitivity

---

## 🛠️ Technical Architecture

**Base Architecture**: Attention U-Net (31.4M parameters)
- Encoder: 4 levels [64, 128, 256, 512 features]
- Attention Gates + Dropout (0.3)
- Combined Loss: Focal Tversky (70%) + Dice (20%) + BCE (10%)

**Optimization**:
- AdamW optimizer (lr: 1e-4 → 1e-7)
- Cosine Annealing scheduler
- Heavy augmentation (20x effective increase)
- Early stopping (patience=25)

---

## 🚀 Quick Start

### Installation
```bash
git clone <repo-url>
cd EndoDetect-AI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train Model
```bash
python train_final.py \
  --data_dir ./data/realistic_mri \
  --epochs 80 \
  --batch_size 4 \
  --device cuda
```

### Run Backend API
```bash
python backend_api.py
```
API starts on `http://localhost:5000`

---

## 📈 Model Outputs

1. **Segmentation Masks**: Multi-class lesion masks
2. **Probability Heatmaps**: Per-pixel likelihood overlays
3. **Surgical Roadmaps** (JSON):
   - Phenotype classification
   - Confidence score
   - Organ involvement
   - Surgical recommendations
4. **Quantitative Metrics**: Volume, Dice coefficient

---

## 🔬 Grant Alignment (D³ Framework)

**Diagnostics (Aim 1)**: 400-patient multimodal validation  
**Drivers (Aim 2)**: Inflammatory endotype discovery  
**Development (Aim 3)**: 100-patient prospective study + biobank  

---

## 📚 Key References

1. Liang Z, et al. "Deep learning for automated segmentation of pelvic MRI in endometriosis." *Sci Data*. 2025. DOI: 10.5281/zenodo.15750762
2. Becker CM, et al. "WERF EPHect phenotyping framework." *Fertil Steril*. 2014;102(5):1223-32.
3. Podda M, et al. "Multi-scale U-Net for endometriosis." *Med Image Anal*. 2024.
4. Liu X, et al. "ML for endometriosis detection." *Artif Intell Med*. 2023.

---

## 📞 Contact

**Technical**: Azra Bano, Rishabh Dhingra  
**Clinical**: Dr. Jessica Opoku-Anane (PI), Dr. Archana Pradhan (Co-I)  
**AI**: Dr. Naveena Yanamala

---

**Built with ❤️ by Rutgers RWJMS EndoDetect AI Team**  
Version 1.0.0-alpha | January 15, 2026

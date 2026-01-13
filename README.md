# DisasTeller

🚨 **DisasTeller** is a multi-agent, Large Vision-Language Model (LVLM)-powered framework for **efficient post-disaster damage assessment, reporting, and response coordination**.

This repository provides the **official implementation** of the paper:

> **Integration of Large Vision-Language Models for Efficient Post-disaster Damage Assessment and Reporting**  
> *Nature Communications*, 2026. https://www.nature.com/articles/s41467-025-68216-z

---

## 📌 Overview

DisasTeller is an agentic, multi-LVLM (Large Vision-Language Model) framework for **automating key tasks in post-disaster response**, including on-site damage assessment, emergency alert generation, resource allocation, and recovery planning. The framework coordinates multiple specialised LVLM agents, with GPT-4o or Gemma3 at its core, to accelerate the disaster response workflow and structure information flow more efficiently than manual coordination. DisasTeller is intended to complement expert teams by rapidly organising visual and textual information, generating structured reports, and reducing human execution time for coordinated response activities.
<br><br>
<img width="940" height="619" alt="image" src="https://github.com/user-attachments/assets/70d35dd0-3603-4e8c-8a28-f213ce9e240e" />
<br><br>
<img width="940" height="559" alt="image" src="https://github.com/user-attachments/assets/4d6b4dac-7cea-44ae-bf5d-5757f537589b" />

---

## 🧠 Framework Highlights

- A multi-agent LVLM framework for structuring post-disaster information workflows  
- Vision-conditioned analysis to support damage interpretation from field images  
- Generation of structured emergency reports and response drafts  
- Modular design enabling flexible agent composition and task configuration   

---

## ⚙️ Environment Setup

The code has been successfully tested on:

- **Windows 10**
- **Linux**
- **macOS**

Please install the required environment before running the framework.

### Dependencies

- `Python == 3.12.4`
- `langchain == 0.2.7`
- `crewai == 0.36.0`
- `crewai-tools == 0.4.8`

You can install all dependencies using the provided environment file:

```bash
conda env create -f environment.yml
conda activate <ENV_NAME>
```

## 🚀 Quick Start

### 1. Set your OpenAI API key

Create or edit the following file:

```text
DisasTeller/.env
```
and add:
```
OPENAI_API_KEY=your_api_key_here
```
### 2. Run the framework
```
python DisasTeller/DisasterManagement_teamwork_simulation.py
```
---

## 📄 Citation

If you find this work useful, please cite the following article:

**Zhaohui, Chen, Elyas Asadi Shamsabadi, Sheng Jiang, Luming Shen, and Daniel Dias-da-Costa.**  
*Integration of large vision language models for efficient post-disaster damage assessment and reporting.*  
**Nature Communications**, 2026.  
https://doi.org/10.1038/s41467-025-68216-z

### BibTeX

```bibtex
@article{Chen2026DisasTeller,
  title   = {Integration of large vision language models for efficient post-disaster damage assessment and reporting},
  author  = {Zhaohui, Chen, Elyas Asadi Shamsabadi, Sheng Jiang, Luming Shen, and Daniel Dias-da-Costa},
  journal = {Nature Communications},
  year    = {2026},
  doi     = {10.1038/s41467-025-68216-z}
}

# Hierarchy Builder (Streamlit App)

This project is an end-to-end **hierarchy builder** designed for extremely wide and sparse master data (e.g., 20,000 rows × 8,000+ columns).  
It cleans data, generates category names, builds semantic and attribute-based clusters, and provides an interactive Streamlit interface for exploring and exporting enriched hierarchies. Example uses include product master data, customer master data, athlete performance data, text data, etc.

---

## Overview

Many real-world master datasets contain thousands of sparse attributes, inconsistent category labels, and missing structure.  
This application solves those problems by:

- Auto-generating or normalizing **category names**
- Creating multi-layer **semantic clusters** using text embeddings
- Building **attribute subclusters** using column sparsity patterns or column value patterns
- Allowing full **interactive editing** (move, merge, split, rename clusters)
- Providing a **downloadable enriched dataset** with hierarchy metadata
- Enabling **analysis** of attribute clusters via supporting scripts

---

# Features

## 1. Preprocessing Layer
Included in `src/preprocessing/numeric_units.py`:
- Extracts units of measurement from messy text (e.g., “234.9 mm”)
- Standardizes numeric values and units into structured columns
- Handles extremely sparse schemas with thousands of columns
- Normalizes measurement value formats

---

## 2. Category Layer
Provided by `src/core/category_layer.py`:
- Cleans inconsistent category names
- Detects missing categories and generates them automatically
- Uses text-based inference when possible
- Ensures every product has a valid `category_name`

---

## 3. Semantic Layer
Implemented in `src/core/semantic_layer.py`:
- Uses **SentenceTransformers** (MiniLM) to embed category names
- Runs KMeans to generate:
  - **Semantic Layer 0** (high-level groups)
  - **Semantic Layer 1** (medium-granularity groups)
- Names clusters using TF-IDF keyword extraction

---

## 4. Attribute Layer
Implemented in `src/core/attribute_layer.py`:
- Operates *within each category*
- Selects meaningful attribute columns (ignores metadata)
- Uses **sparsity signatures** or **column values** to group similar products
- Names clusters using:
  - Attribute purity
  - Fallback rules for unclear clusters

This layer often reveals meaningful hidden structure in product families.

---

## 5. Full Streamlit Application
Found in `src/app/main.py` and `src/app/wizard_pages/`.

Includes an interactive multi-step wizard:

### **Upload**
- Upload CSV files (large datasets supported)
- Automatically infers schema characteristics

### **Configure**
- Select category columns
- Configure preprocessing and layer options

### **Semantic Layers**
- Build Layer 1 and Layer 0 clusters
- Edit cluster names
- Rearrange cluster membership

### **Attribute Layer**
- Build clusters per category
- Examine attribute signatures
- Visualize product distributions

### **Explore**
- File-system–like hierarchy browser
- Click down levels:
  - Layer 0 → Layer 1 → Category → Attribute Cluster
- Inspect products and attributes

### **Download**
- Export:
  - Enriched master dataset
  - Cluster summaries
  - Optional PDF visualizations

---

# Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the App

Run Streamlit:

```bash
streamlit run src/app/main.py
```

Streamlit will open in your browser at:

```
http://localhost:8501
```

---

# Using the App

### 1. Upload Data
Load a CSV file from your machine. Use included synthetic examples if you want to test quickly.

### 2. Configure Options
Choose:
- Category column
- Whether to auto-generate category names
- Whether to preprocess numeric/units
- Layer-building settings

### 3. Build Semantic Layers
Two-stage semantic hierarchy:
- High-level groupings
- More granular groupings

### 4. Build Attribute Clusters
Cluster within categories based on attribute sparsity or value, depending on preference.

### 5. Explore Hierarchy
Browse using an intuitive explorer UI.

### 6. Export Results
Download enriched datasets and cluster summaries.

---

# Tech Stack

- **Python 3.10+**
- **Streamlit**
- **pandas / numpy**
- **scikit-learn**
- **sentence-transformers**
- **NLTK**
- **regex**
- **matplotlib**

---

# Future Enhancements

- Add caching for large datasets
- Allow full session save/load of edited hierarchies
- Add UI control for granularity of attribute clusters
- Add embeddings-based naming for attribute subclusters
- Add GPU support for embedding generation


---

# 💬 Contact

For questions or collaboration opportunities, feel free to reach out or open an issue on GitHub.

# Product Hierarchy Builder (Streamlit App)

This project is an end-to-end **product hierarchy builder** designed for extremely wide and sparse product master data (e.g., 20,000 rows × 8,000+ columns).  
It cleans data, generates category names, builds semantic and attribute-based clusters, and provides an interactive Streamlit interface for exploring and exporting enriched product hierarchies.

---

## 🚀 Overview

Many real-world product master datasets contain thousands of sparse attributes, inconsistent category labels, and missing structure.  
This application solves those problems by:

- Auto-generating or normalizing **category names**
- Creating multi-layer **semantic clusters** using text embeddings
- Building **attribute subclusters** using column sparsity patterns
- Allowing full **interactive editing** (move, merge, split, rename clusters)
- Providing a **downloadable enriched dataset** with hierarchy metadata
- Enabling **analysis** of attribute clusters via supporting scripts

This repository serves as a complete data-engineering + machine-learning + web-app portfolio project.

---

# 📁 Features

## 🧹 1. Preprocessing Layer
Included in `src/preprocessing/numeric_units.py`:
- Extracts units of measurement from messy text (e.g., “234.9 mm”)
- Standardizes numeric values and units into structured columns
- Handles extremely sparse schemas with thousands of columns
- Normalizes measurement value formats

---

## 🏷️ 2. Category Layer
Provided by `src/core/category_layer.py`:
- Cleans inconsistent category names
- Detects missing categories and generates them automatically
- Uses text-based inference when possible
- Ensures every product has a valid `category_name`

---

## 🧠 3. Semantic Layers (Layer 1 and Layer 0)
Implemented in `src/core/semantic_layer.py`:
- Uses **SentenceTransformers** (MiniLM) to embed category names
- Runs KMeans to generate:
  - **Semantic Layer 1** (high-level groups)
  - **Semantic Layer 0** (medium-granularity groups)
- Names clusters using TF-IDF keyword extraction (`tfidf_cluster_label`)

---

## 🧩 4. Attribute Layer (Subclusters)
Implemented in `src/core/attribute_layer.py`:
- Operates *within each category*
- Selects meaningful attribute columns (ignores metadata)
- Uses **sparsity signatures** to group similar products
- Names clusters using:
  - Attribute purity
  - TF-IDF weighting
  - Fallback rules for unclear clusters

This layer often reveals meaningful hidden structure in product families.

---

## 🖥️ 5. Full Streamlit Application
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
- Build subclusters per category
- Examine attribute signatures
- Visualize product distributions

### **Explore**
- File-system–like hierarchy browser
- Click down levels:
  - Layer 1 → Layer 0 → Category → Attribute Cluster
- Inspect products and attributes

### **Download**
- Export:
  - Enriched master dataset
  - Cluster summaries
  - Optional PDF visualizations

---

# 📂 Project Structure

```
hierarchy_app/
├── data/
│   ├── synthetic_product_master.csv
│   ├── synthetic_product_master_200_categories_literary.csv
│   ├── synthetic_product_master_200_categories_structured.csv
│   ├── raw/           # place your own raw product master files here
│   ├── interim/       # intermediate processed outputs
│   └── processed/     # final processed / enriched datasets
└── src/
    ├── app/
    │   ├── main.py
    │   ├── components/
    │   └── wizard_pages/
    ├── core/
    │   ├── hierarchy_engine.py
    │   ├── hierarchy.py
    │   ├── semantic_layer.py
    │   ├── attribute_layer.py
    │   ├── category_layer.py
    │   ├── text_utils.py
    │   └── naming_utils.py
    ├── preprocessing/
    │   └── numeric_units.py
    └── analysis/
        └── subcluster_analysis.py
```

---

# 🛠️ Installation

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

# ▶️ Running the App

Run Streamlit:

```bash
streamlit run src/app/main.py
```

Streamlit will open in your browser at:

```
http://localhost:8501
```

---

# 📊 Using the App

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

### 4. Build Attribute Subclusters
Cluster within categories based on attribute sparsity.

### 5. Explore Hierarchy
Browse using an intuitive explorer UI.

### 6. Export Results
Download enriched datasets and cluster summaries.

---

# 📚 Tech Stack

- **Python 3.10+**
- **Streamlit**
- **pandas / numpy**
- **scikit-learn**
- **sentence-transformers**
- **NLTK**
- **regex**
- **matplotlib**

---

# 🧭 Future Enhancements

- Add caching for large datasets
- Allow full session save/load of edited hierarchies
- Add UI control for granularity of attribute clusters
- Add embeddings-based naming for attribute subclusters
- Add GPU support for embedding generation

---

# 📝 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

---

# 🙌 Contributions

Pull requests are welcome!  
Feel free to:
- Add new preprocessing modules  
- Extend semantic-layer logic  
- Add new visualization components  
- Improve naming heuristics  

---

# 💬 Contact

For questions or collaboration opportunities, feel free to reach out or open an issue on GitHub.

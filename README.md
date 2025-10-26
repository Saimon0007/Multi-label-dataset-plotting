# Multi-label Dataset Plotting

A small collection of Jupyter Notebook tools and examples for exploring and visualizing multi-label datasets.  
This repository helps you quickly understand label distributions, label co-occurrence, sample-level label counts, and low-dimensional embeddings (t-SNE / UMAP) of multi-label data.

Why use this repo?
- Quickly generate informative visualizations for multi-label classification datasets.
- Inspect label imbalance and common co-occurrences.
- Visualize sample relationships in 2D while preserving multi-label information.
- Serve as handy starter notebooks for dataset EDA before modelling.

---

## Contents

- Jupyter notebooks (main work of the repo)
  - Notebooks demonstrate:
    - Label distribution bar charts
    - Label co-occurrence heatmaps
    - Histogram of label counts per sample
    - t-SNE / UMAP embeddings colored by label (strategies for handling multiple labels)
    - Example pipeline from CSV -> binarized label matrix -> visualizations

(If you'd like, I can add a short index of existing notebooks in the repo — tell me and I'll list them.)

---

## Quickstart

1. Clone the repository:
   git clone https://github.com/Saimon0007/Multi-label-dataset-plotting.git
   cd Multi-label-dataset-plotting

2. Create and activate a Python environment (recommended):
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate     # Windows

3. Install dependencies:
   pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn umap-learn plotly

   - Optional extras depending on notebooks: scikit-multilearn, plotly for interactive plots

4. Start Jupyter and open the notebooks:
   jupyter lab
   or
   jupyter notebook

5. Run the cells in the notebook(s). Each notebook contains explanatory text and runnable code blocks.

---

## Expected dataset formats

This repository expects multi-label datasets in one of the common formats:

1. CSV with a single "labels" column
   - Example row:
     id,text,labels
     1,"sample text","cat|outdoor|pet"
   - Specify the delimiter used in the labels field (e.g. `|`, `;`, `,`) when loading.

2. CSV with one-hot / binary label columns
   - Example columns:
     id,text,label_cat,label_outdoor,label_pet
   - Values: 0/1 indicating presence/absence.

The notebooks include examples that:
- Parse a delimited labels string and convert to a binary indicator matrix (pandas + sklearn.preprocessing.MultiLabelBinarizer)
- Work with datasets that already have binary label columns

---

## Visualizations included (high level)

- Label frequency bar chart
  - Shows how often each label appears across the dataset.
  - Useful to spot class imbalance.

- Label counts per sample histogram
  - Shows how many labels a typical sample has.

- Label co-occurrence (correlation) heatmap
  - Visualizes pairs of labels that commonly appear together.
  - Uses normalized co-occurrence counts or phi-coefficient / Jaccard Index.

- t-SNE / UMAP embedding with label overlays
  - Projects high-dimensional feature vectors or TF-IDF representations down to 2D.
  - Several strategies for multi-label visualization:
    - Color by a single chosen label (show presence/absence)
    - Show separate scatter panels for top-k labels
    - Use marker shapes / sizes to indicate other labels
    - Interactive tooltips (via plotly) that list all labels for each sample

- Pairwise label scatter / UpSet plots (recommendation)
  - UpSet plots are recommended for complex co-occurrence exploration (not always present by default).

---

## Example code snippets

- Convert delimited label column to binary matrix (pandas + sklearn):
```python
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

df = pd.read_csv("data.csv")
df["labels_list"] = df["labels"].str.split("|")  # change delimiter if needed

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["labels_list"])
label_names = mlb.classes_
```

- Simple label frequency plot (matplotlib / seaborn):
```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

counts = Y.sum(axis=0)
order = np.argsort(counts)[::-1]
sns.barplot(x=counts[order], y=np.array(label_names)[order])
plt.xlabel("Label count")
plt.ylabel("Label")
plt.title("Label frequency")
```

---

## Tips & best practices

- Pre-cook large datasets into smaller samples for exploratory plotting (faster rendering).
- For co-occurrence heatmaps, normalize counts so very frequent labels don't drown out rarer but important pairings.
- When using t-SNE/UMAP, use deterministic random_state for reproducibility and scale/standardize features first.
- Consider UpSet plots for high-cardinality label-set discovery (libraries like upsetplot exist).
- If labels are many (hundreds), focus on top-K most frequent labels for interpretable plots.

---

## Extending this repo

- Add notebooks for:
  - Interactive dashboards (Voila / Streamlit)
  - Automatic report generation (nbconvert) for dataset EDA
  - UpSet and network graphs of label co-occurrence
- Add a requirements.txt and a Binder/Colab badge so users can launch notebooks quickly.
- Add real example datasets (or links) to demonstrate the notebooks.

---

## Contributing

Contributions are welcome. Good ways to contribute:
- Add new plotting notebooks or enhancements
- Add unit tests or example datasets
- Improve readme with concrete notebook index and rendered GIFs/screenshots

Please open an issue or a pull request; label your PR with a brief description of what changed.

---

## License

MIT License — see the LICENSE file (or tell me if you want me to add one and I can create it).

---

## Contact

Maintainer: @Saimon0007 (GitHub) 

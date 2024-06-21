# PCA and ICA Fusion Applied in Speech Emotion Recognition (SER)

This repository demonstrates the application of Principal Component Analysis (PCA) and Independent Component Analysis (ICA) in Speech Emotion Recognition (SER).

## Datasets

To test these scripts, you will need to download the following datasets:

1. **SAVEE** 
2. **RAVDESS**
3. **Berlin EmoDB** - Can be downloaded using [audb](https://github.com/audeering/emodb?tab=readme-ov-file).

## How to Use

### 1. Generate Files

Run the following scripts to create `.xlsx` and `.csv` files:

- `<database_name>_pca_ica_components.py`
- `<database_name>_pca_ica_components_kw.py`

Replace `<database_name>` with the name of the dataset you are using see name of the files.

### 2. Generate Graphs

After generating the necessary files, you can create graphs by running the `plot.py` scripts.

## Example

To create PCA and ICA components for the SAVEE dataset:
```sh
python SAVEE_pca_ica_components.py
python SAVEE_pca_ica_components_kw.py
```

To plot the results:
```sh
python plot_emodb.py
python plot_savee.py
python ravdess_graph_plot.py

```

## Notes

- Ensure that the required datasets are downloaded and properly set up before running the scripts.
- The scripts assume that the datasets are organized in their respective directories.

## Acknowledgements

- **SAVEE** - Surrey Audio-Visual Expressed Emotion Database
- **RAVDESS** - Ryerson Audio-Visual Database of Emotional Speech and Song
- **Berlin EmoDB** - Berlin Database of Emotional Speech

Feel free to reach out if you encounter any issues or have questions.



---



---


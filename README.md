# PCA and ICA Fusion Applied in Speech Emotion Recognition (SER)

This repository demonstrates the application of Principal Component Analysis (PCA) and Independent Component Analysis (ICA) in Speech Emotion Recognition (SER).

## Datasets

To test these scripts, you will need to download the following datasets:

1. **SAVEE** - Can be downloaded in https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
2. **RAVDESS** Can be downloaded in https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
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
## Citation

If you find this repository useful for your research or work, please consider citing the associated article:

> Kingeski, R.; Henning, E.; Paterno, A.S. Fusion of PCA and ICA in Statistical Subset Analysis for Speech Emotion Recognition. Sensors 2024, 24, 5704. https://doi.org/10.3390/s24175704


---


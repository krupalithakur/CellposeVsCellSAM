# 🔬 Cellpose vs CellSAM: Microscopy Segmentation Benchmark

This repository contains the initial setup and code for benchmarking **Cellpose** and **CellSAM** on microscopy image datasets.

The aim is to evaluate how well each model performs for cell segmentation tasks using ground truth annotations, focusing on real-world datasets like **LIVECell** and **S-BIAD634**.

## 📁 Project Structure

- `images/` – Raw microscopy images  
- `masks/` – Ground truth segmentation masks  
- `cellpose_results/` – Outputs from Cellpose  
- `cellsam_results/` – Outputs from CellSAM  
- `notebooks/` – Jupyter/Colab notebooks for inference and comparison  
- `utils/` – Helper functions for visualisation and evaluation

## 🛠️ Tools & Libraries

- Python, OpenCV, NumPy, Matplotlib  
- Cellpose (`pip install cellpose`)  
- Segment Anything / CellSAM  
- Google Colab (recommended for GPU-based runs)

## 🚀 Current Focus

- Run baseline inference with both models  
- Visualise and overlay predictions  
- Prepare data for evaluation

## 🧠 Key References

- 🔹 **Cellpose: A Generalist Algorithm for Cellular Segmentation**  
  Stringer et al., 2021 – [https://doi.org/10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x)

- 🔹 **Segment Anything**  
  Kirillov et al., Meta AI, 2023 – [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)

- 🔹 **CellSAM: Segment Anything in Microscopy**  
  S. Ji et al., 2023 – [https://arxiv.org/abs/2306.00989](https://arxiv.org/abs/2306.00989)

- 🔹 **Cellpose 2.0: How to Train Your Own Model**  
  Pachitariu et al., 2022 – [https://doi.org/10.1101/2022.04.01.486764](https://doi.org/10.1101/2022.04.01.486764)

- 🔹 **SAM-Cell: A Microscopy-Tuned Version of SAM (Community Repo)**  
  [https://github.com/saikat2019/SAM-Cell](https://github.com/saikat2019/SAM-Cell)

---

> 📌 *This project is part of my MSc dissertation work at Queen’s University Belfast.*


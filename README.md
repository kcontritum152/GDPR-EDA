# GDPR-EDA

Exploratory Data Analysis (EDA) and visualization related to GDPR text similarity and insights.

This repository contains Python scripts for analyzing and visualizing aspects of GDPR-related data.

## Overview

The goal of this project is to explore patterns and similarities within GDPR text or related datasets, and to produce visualizations that help reveal structure, clusters, or relationships in the data.

## Repository Contents

- **`gdpr_similarity.py`** — Scrapes GDPR articles from gdpr-info.eu, generates semantic embeddings for each article, computes cosine similarity between articles, and evaluates how well embedding-based similarity matches the site’s ground-truth related-article references.
- **`gdpr_vis2.py`** — Extends GDPR article similarity analysis by improving ground-truth extraction and producing multiple visualizations—heatmaps, network graphs, and 2D embedding projections—to explore and validate semantic relationships between articles.
- **`Outputs/`** — Directory containing generated visualizations, plots, and files produced by the analysis.


## Features

- Compute similarity or distance between GDPR text samples
- Perform exploratory analysis on GDPR-related dataset(s)
- Produce visualizations to support data insights

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/kcontritum152/GDPR-EDA.git
cd GDPR-EDA

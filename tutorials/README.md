---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
---

We use jupytext to keep copies of notebook in sync with Markdown equivalent.

### Adding a new notebook

```
jupytext --set-formats ipynb,md:myst path/to/the/notebook.ipynb
```

### Syncing Notebooks

After editing either the ipynb or md versions of the notebooks, you can sync the two versions using jupytext by running:

```
jupytext --sync docs/notebooks/*
```

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

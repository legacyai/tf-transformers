def add_new_files_to_jupytext():
    """This function will check all .ipynb and .md files.
    If a new .ipynb is present, it will convert to equivalent .md
    """
    import glob
    import subprocess

    all_ipynb = glob.glob('tutorials/*.ipynb')
    all_md = glob.glob('tutorials/*.md')

    all_ipynb = [name.split('.ipynb')[0] for name in all_ipynb]
    all_md = [name.split('.md')[0] for name in all_md]

    notebook_list = []
    for notebook_name in all_ipynb:
        if notebook_name not in all_md:
            notebook_list.append(notebook_name + '.ipynb')

    if notebook_list != []:
        for notebook in notebook_list:
            subprocess.run(["jupytext --set-formats ipynb,md:myst", notebook])


def move_to_docs():
    "Move tutorals to docs"
    import shutil

    shutil.copytree("tutorials", "docs/source/tutorials", dirs_exist_ok=True)


if __name__ == '__main__':
    add_new_files_to_jupytext() # Convert new notebooks to md using jupytext
    move_to_docs # Move new tuorials to docs

WikiSection dataset
====================

This folder is intended to hold a local copy of the WikiSection dataset used by the project. The repository does not include the dataset by default, so you must download or clone it and place the dataset files here before running the experiments in `src/`.

Source
------
Official repository: https://github.com/sebastianarnold/WikiSection

How to obtain the dataset
-------------------------
Choose one of the options below and make sure the dataset files (the contents of the WikiSection repository or the specific dataset files you need) are placed directly inside this folder (`data/wikisection_dataset/`). The scripts in `src/` expect to find the dataset under this path.

1) Clone the repository (recommended if you want the full source and examples):

```bash
# clone the repo and move its contents into this folder
git clone https://github.com/sebastianarnold/WikiSection.git /tmp/WikiSection
# copy or move the dataset files into this folder
cp -r /tmp/WikiSection/* .
rm -rf /tmp/WikiSection
```

2) Download the ZIP from GitHub and unzip into this folder.

Note about the subset required for this project
------------------------------------------------
This project only needs the `en_city` subset of the WikiSection dataset. If you download the full dataset, you can remove other subsets to save space, but make sure the `en_city` JSON files remain in `data/wikisection_dataset/`.

Unpacking the JSON tarball
-------------------------
If you receive the dataset as the JSON tarball named `wikisection_dataset_json.tar.gz`, extract it into this folder and ensure the JSON files end up directly in `data/wikisection_dataset/`. Example commands (adjust paths as needed):

```bash
# move the tarball into the dataset folder (optional)
mv ~/Downloads/wikisection_dataset_json.tar.gz data/wikisection_dataset/

cd data/wikisection_dataset

# extract the tarball here
tar -xzf wikisection_dataset_json.tar.gz

# if extraction creates a nested directory like "wikisection_dataset_json/", move its contents up
if [ -d "wikisection_dataset_json" ]; then
	mv wikisection_dataset_json/* .
	rmdir wikisection_dataset_json
fi

# remove the tarball if you don't need it
rm -f wikisection_dataset_json.tar.gz
```

Expected usage in this repo
---------------------------
- The code in `src/` expects the dataset to be available at `data/wikisection_dataset/` relative to the project root.
- If you keep the dataset elsewhere, update the code or create a symlink. Example to create a symlink from another location:

```bash
ln -s /path/to/your/WikiSection data/wikisection_dataset
```

License & citation
-------------------
Please check the original WikiSection repository for licensing, citation, and attribution information before using or publishing results based on the data.

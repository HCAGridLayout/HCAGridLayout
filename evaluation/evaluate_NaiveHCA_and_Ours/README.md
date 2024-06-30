Compile
----------
There are several packages that need to be compiled or installed manually.

\
Color Package:

Please come to [this repository](https://github.com/Dynamic-Color/Dynamic-Color) to download and compile color packages.

Then move the compiled files (such as .pyd or .so) to backend/application/data/color/.

\
Linear Assignment Package:
```
cd backend
cd application
cd grid
cd linear_assignment
python setup.py install
```

\
Grid Layout Utils Package:
```
cd backend
cd application
cd grid
cd "c module_now"
python setup.py build_ext --inplace    (For MSVC environment in Windows, please use setup_win.py)
```
Then move the compiled files (such as .pyd or .so) to backend/application/grid/.

\
Incremental T-SNE Package:

Please come to backend/application/data/incremental_tsne/ to compile packages with cmake.

Then move the compiled files (such as .dll or .so) to backend/application/data/.

\
If you encounter issues while compiling, please try adjusting the line endings to LF or CRLF according to your system environment.

Datasets
----------
Please download datasets from [google drive](https://drive.google.com/drive/folders/15R0ghoW9YkYbnDaU8NXQy6IqdnKPoLYm) or [osf](https://osf.io/a8epu/?view_only=fac7bd5cbfc149fbb373df3e0eb5810f) and move the directory to backend/datasets/.

For example, backend/datasets/cifar100/.

Then unzip the .zip files in the folder, and run calc_conf.py and multiload.py (if they exist) to preprocess.

For your dataset, please:
1. prepare the xxx.json, xxx_features.npy, xxx_labels.npy, xxx_labels_gt.npy, xxx_predict_confs.npy and xxx_images folder, xxx can be the name of your dataset. The details of these files can be seen in README.md in [google drive](https://drive.google.com/drive/folders/15R0ghoW9YkYbnDaU8NXQy6IqdnKPoLYm) or [osf](https://osf.io/a8epu/?view_only=fac7bd5cbfc149fbb373df3e0eb5810f).
2. Then run calc_conf.py to preprocess confidence like existing datasets.
3. For large dataset, please run multiload.py to preprocess incremental loading.
4. Add your dataset to function "load" of LabelHierarchy in backend/application/data/LabelHierarchy.py.
5. Add your dataset to data "options", "options2" and "DatasetModels" in vis/src/components/ControlView.vue.

For the evaluation with dendromap, we use mini subsets of each dataset because dendromap cannot support datasets that are too large.

If you want to evaluate with dendromap, please also download exploration traces traces.zip in [google drive](https://drive.google.com/drive/folders/15R0ghoW9YkYbnDaU8NXQy6IqdnKPoLYm) or [osf](https://osf.io/a8epu/?view_only=fac7bd5cbfc149fbb373df3e0eb5810f) which are generated in dendromap system by manual zoom-in operations, and then unzip it in the backend folder. For example, backend/dendromap_step_cifar100_16px.json.

Run Evaluation
----------
Run eva.py to start evaluation.
```
cd backend
python eva.py
```

Then run eva2.py to calculate average scores.
```
python eva2.py
```

Run Evaluation With DendroMap
----------
To evaluate our method compared with dendromap, run new_eva_from_dendromap.py
```
cd backend
python new_eva_from_dendromap.py
```

Then run new_eva_from_dendromap2.py to calculate average scores.
```
python new_eva_from_dendromap2.py
```


## Contact
If you have any problem about our code, feel free to contact
- HCAGridlayout@gmail.com

or describe your problem in Issues.

Hierarchical Cluster-Aware Grid Layout Algorithm
===========================

=======================================

Codes for grid layout algorithm described in our paper ["Hierarchical Cluster-Aware Grid Layout"](https://xxxx) (xxxx).

Note
----------
Tested on python 3.8.

This method is best used in a Windows or Mac environment; otherwise, the parallelism of QAP solver may not lead to efficiency improvements, resulting in longer runtime.

Setup Environment
----------
You can use pip or other tools to setup environment.

For Example:
```
pip install -r requirements.txt
```



Compile
----------
There are several packages that need to be compiled or installed manually.

\
Color Package:

Please come to [this repository](https://xxxx) to download and compile color packages.

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

Datasets
----------
Please download datasets from [here](https://drive.google.com/drive/folders/15R0ghoW9YkYbnDaU8NXQy6IqdnKPoLYm) and move the directory to backend/datasets/.

For example, backend/datasets/cifar100/.

Then run calc_conf.py and multiload.py to preprocess.

For your dataset, please:
1. prepare the xxx.json, xxx_features.npy, xxx_labels.npy, xxx_labels_gt.npy, xxx_predict_confs.npy and xxx_images folder, xxx can be the name of your dataset. The details of these files can be seen in README.md [here](https://drive.google.com/drive/folders/15R0ghoW9YkYbnDaU8NXQy6IqdnKPoLYm). 
2. Then run calc_conf.py to preprocess confidence like existing datasets.
3. For large dataset, please run multiload.py to preprocess incremental loading.
4. Add your dataset to backend/application/data/LabelHierarchy.py, load function.
5. Add your dataset to vis/src/components/ControlView.vue, options, options2 and DatasetModels.

Run
----------
For Backend:

Modify the "port = Port(xxx)" and "port.load_dataset(xxx)" statements in server.py according to your requirements.
```
cd backend
bash run.sh    (or directly runing server.py)
```

For Frontend:
```
cd vis
yarn
yarn start
```


## Contact
If you have any problem about our code, feel free to contact
- HCAGridlayout@gmail.com

or describe your problem in Issues.

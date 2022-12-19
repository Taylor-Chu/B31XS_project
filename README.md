# B31XS project
### Replication of [Burke et al. 2019, MNRAS, 490 3952.](http://adsabs.harvard.edu/doi/10.1093/mnras/stz2845)
GitHub repositories used:
- https://github.com/burke86/astro_rcnn/
- https://github.com/burke86/astrodet/
- https://github.com/facebookresearch/detectron2/

**Python 3.8.0** should be used in accordance with packages listed in `requirements.txt` for the codes to work.

Datasets for training, validation as well as testing can be downloaded from instructions in https://github.com/burke86/astro_rcnn/. Dataset folder should be renamed `train`, `val` and `test` respectively for codes to work seamlessly.

New models can be trained by running `train.py`, initial weights and trained models can be downloaded from [this dropbox link](https://www.dropbox.com/sh/jphwb6aunju34d2/AADQvXnmuXiy3skyUi_Vc7vSa?dl=0).
Models can be tested by running `test.py`.

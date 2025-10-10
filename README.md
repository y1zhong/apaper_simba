# apaper_simba
This repository maintains the code and scripts for "SIMBA: Scalable Image Modeling using a Bayesian Approach, A Consistent Framework for Including Spatial Dependencies in fMRI Studies".

The data applied to this code is stored at the [OSF project](https://osf.io/2d6t8/?view_only=933dc9343ebb4a13b9ee5b921bf46279).


Notes on running the Jupyter notebooks for the SIMBA code.


Directory setup
---------------

To run the examples here, the user's directory should be set up as
follows:

  apaper_simba/    : copy of GitHub repository of SIMBA code and notebooks
  data/            : directory with data for examples
  |-- NARPS        : dir with input data pickles and NIFTI data for Ex. 2
  `-- simulation   : dir with input data pickles for Ex. 1

The simulation demo (Ex. 1) uses the mask dataset in the NARPS
directory when generating baseline inputs.

Note that Ex. 3 uses ABCD which is not publicly shareable in the same
way as the NARPS and simulation data that are openly included here.
Users can download that separately to run the Ex. 3 notebook.


Environment/dependencies
------------------------

The Python module and other dependencies needed to run these data
projects are listed in the `import ..` lines of the repository
notebooks.  Additionally, the code repository's
environment_apaper_simba.yml text file contains a list of all
dependencies across the notebook examples.

To use conda to install the dependencies, one can install the
program (using, say, Miniconda) and then execute the following to 
build it:

  conda env create -f environment_apaper_simba.yml

Then, users can activate the environment with:

  conda activate apaper_simba


Running the example notebooks
-----------------------------

The following Jupyter notebooks are included in this code repository:

  abcd_analysis.ipynb        : notebook for Ex. 3 (ABCD surface data)
  narps_analysis.ipynb       : notebook for Ex. 2 (NARPS volumetric analysis)
  simulation_analysis.ipynb  : notebook for Ex. 1 (simulated data)

The directories of data for `narps*.ipynb` and `sim*.ipynb` can be
downloaded directly from OSF.  As noted above, data for `abcd*.ipynb`
must be obtained separately.

To run any simulation, first organize the code and data set shown
above.  Then, start a Jupyter interface (with the appropriate
dependencies available; see previous section) from the code directory,
such as by running:

  cd apaper_simba
  jupyter-notebook

Finally, select the appropriate notebook file, and execute the cells
within it.
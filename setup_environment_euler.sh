#For use on the Euler cluster: this bash script sets up an evironment called 'pylbmforsolids',
#simply running this script i.e.:  
#
#   bash setup_environment_euler.sh
#
#from your home directory creates and activates the environment and installs our pylbm fork.
#It also links this environment to Jupyterhub. 
#Generally, the environment can then be activated and deactivated with
#
#   source pylbmforsolids/bin/activate
#
#and
#
#   deactivate
#
#The modules in 'module load' should be loaded for the environment to work correctly.

module load gcc/8.2.0 openmpi/4.1.4 hdf5/1.10.1 python/3.10.4

python -m venv pylbmforsolids

source pylbmforsolids/bin/activate

pip install 'pylbm @ git+https://github.com/bucphili/pylbm'

pip install ipykernel

python -m ipykernel install --user --name=pylbmforsolids

pip uninstall cython

pip install cython==0.29.36
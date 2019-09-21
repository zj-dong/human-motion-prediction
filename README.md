You could find the report and code for the project 4 "human motion prediction" below.
After downloading the code and unzip it, you should set the environment and related requirement as described on the webpage https://machine-perception.ait.inf.ethz.ch/project4/#description:
    
    module load python_gpu/3.6.4 hdf5 eth_proxy
    
    module load cudnn/7.2

Install virtualenvwrapper if you haven't done so yet:
    
    pip install virtualenvwrapper
    
    source $HOME/.local/bin/virtualenvwrapper.sh
    
Create a virtual environment:

    mkvirtualenv "env-name"

The virtual environment is by default activated. You can disable or enable it by using

    deactivate
    workon "env-name"

Go to you project folder and run the following to install the required dependencies:
    
    python setup.py install

To train the model, run the following commend:

    bsub -n 6 -W 4:00 -R "rusage[mem=10000, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp19/project4 --save_dir ./experiments --experiment_name sample


the training process requires enough space and should takes about four hours. And to evaluate the result:

    bsub -n 6 -W 4:00 -R "rusage[mem=10000, ngpus_excl_p=1]" python evaluate_test.py --data_dir /cluster/project/infk/hilliges/lectures/mp19/project4 --save_dir ./experiments --model_id model_id --export
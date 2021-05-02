## Semantic-Image-Manipulation (simsg)

<p align="right", style="font-size:30px"><b>Authors:</b><br />Jitong Ding<br />Eashan Adhikarla</p>

### Reproducing Paper with Improvements
Paper: [Semantic Image Manipulation Using Scene Graphs](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dhamo_Semantic_Image_Manipulation_Using_Scene_Graphs_CVPR_2020_paper.pdf)
Dhamo et al., CVPR 2020 

This is a reproduction of the paper with minor improvements in architecture and loss functions.
We have improved the training time of the model by reducing the code structure complexity.

For the code setup we use the anaconda environment. Please follow the setup below to
reproduce the environment and run code.
Note: Python version 3.7 is must.

### Code Setup:

```
# Step 1. Cloning the repository
git clone https://github.com/eashanadhikarla/simsg.git

# Step 2. Go into the Directory
cd simsg

# Step 3. Create a new conda environment
conda create --name sim python=3.7 --file requirements.txt

# Step 4. Activate the virtual environment
conda activate sim

# Step 5. If using CUDA use cudatookkit (else remove `cudatoolkit` from the below cmd)
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# Step 6. Install Pip requirements that were missing in Conda
python3.7 -m pip install opencv-python tensorboardx grave addict

# Step 7. 
which python ( >> copy the output path minus `/bin/python` )

# Step 8.
echo $PWD > <paste_the_copied_path_from_step_7>/lib/python3.7/site-packages/simsg.pth

```

We have the dataset and the models already preprocessed. 
Below is the dropbox link for download.

```
datasets : https://www.dropbox.com/sh/03u4d3dhifdywz8/AABjone998j6bgb8G0EYh5g5a?dl=0
experiments : https://www.dropbox.com/sh/w001qymczovq5tn/AACHQN3C2JlBFAJR3xMTFlDea?dl=0
```

### Overall Folder Structure:

- root
	- dataset (download from the link above)
	- experiments ((download from the link above))
	- PerceptualSimilarity
	- preprocess-data
	- scripts
	- simsg
	- requirements.txt


## Run Code:
### Train:

python scripts/train_spade.py 
python scripts/train_crn.py

Note: (To change the input size, you can change input_size and batch_size in train_spade.py/train_crn.py)

### Evaluation:

python scripts/evaluate_clevr.py \
--exp_dir ./experiments/clevr_models \
--experiment spade_64_clevr_model_final




































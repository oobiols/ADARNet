## PYTHON >=3 AND PIP >= 19.x ARE ASSUMED. 
## PLEASE GET ALL REQUIRED LIBRARIES FIRST:

pip install -r requirements.txt

## DOWNLOAD THE DATASET

cd datasets
bash download.sh

### CODE:

There are four main files: amr.py , ./src/NS_amr.py , postprocess.py , ./src/PostProcess.py

- amr.py: load the data, call the model, train it, and save it
- ./src/NS_amr.py: builds the model and sets up the training steps
- postprocess.py: loads the saved model, makes the inference, and writes the results
- ./src/PostProcess.py: main class for postprocess.py. Has all the different postprocessing functions (drawing the refinement levels, writing OpenFOAM files)





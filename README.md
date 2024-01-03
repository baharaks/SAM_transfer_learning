Solar Panel Image Segmentation


Overview
This repository contains a collection of Python scripts and utilities designed for the analysis and processing of solar panel images. 

The toolkit focuses on tasks such as image segmentation, defect detection, model fine-tuning, and general image analysis in the context of solar energy technology.


Contents


solar_panel_roboflow.py: Processes and analyzes solar panel images, potentially for defect detection or segmentation.
sam_segment.py: Performs automatic image segmentation, segment anything by meta model.
sam_fine_tune_solar_refactor.py: Focuses on fine-tuning segmentation models specifically for solar panel imagery.
utils.py: A utility module supporting various operations including image processing, neural network operations, and data handling.


Setup


To use these scripts, you need to install the necessary Python libraries. 
Each script has its own requirements.txt file that lists the required libraries.

Clone the repository:

git clone https://github.com/baharaks/SAM_transfer_learning.git

cd SAM_transfer_learning

Install the dependencies for each script:

pip install -r requirements.txt

To get your API_KEY from Roboflow platform follow this link: https://docs.roboflow.com/api-reference/authentication

Usage

Each script can be run independently based on the specific task at hand.

How to use the codes:
1. Run sam_transfer_learning.py for transfer learning on SAM model
2. Run sam_segment.py to test SAM model. Download the fine tuned model from https://drive.google.com/file/d/1UgthKtmwuLhktC40LNUWIU_w7W8evkUA/view?usp=sharing 
4. Run solar_panel_roboflow.py for combination of roboflow model and SAM

License
[Specify your license here]


Solar Panel Image Analysis Toolkit
Overview
This repository contains a collection of Python scripts and utilities designed for the analysis and processing of solar panel images. The toolkit focuses on tasks such as image segmentation, defect detection, model fine-tuning, and general image analysis in the context of solar energy technology.

Contents
solar_panel_roboflow.py: Processes and analyzes solar panel images, potentially for defect detection or segmentation.
sam_segment.py: Performs automatic image segmentation, leveraging deep learning models.
sam_fine_tune_solar_refactor.py: Focuses on fine-tuning segmentation models specifically for solar panel imagery.
utils.py: A utility module supporting various operations including image processing, neural network operations, and data handling.
Setup
To use these scripts, you need to install the necessary Python libraries. Each script has its own requirements.txt file that lists the required libraries.

Clone the repository:

git clone [Your Repository URL]

cd [Your Repository Folder]

Install the dependencies for each script:

pip install -r requirements.txt

Note: The utils.py file does not require a separate requirements file as it is a utility module.
Usage

Each script can be run independently based on the specific task at hand:

License
[Specify your license here]

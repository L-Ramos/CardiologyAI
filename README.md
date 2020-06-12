# CardiologyAI
Repository for the AI code developed for the PLN project (Amsterdam UMC)

It contains code for cleaning and pre processing data, as well as code for creating and optimizing models.

Current content:


pre_process_data.py: file for reading and converting ecg signals in .xml format

find_ecg_data: main file with all functions needed to clean and select patient data. Uses aux_functions.py.

main: main code to load the data and train the models

clean_data: used by main to organize the beats in the right format

models: used to define the models to be trained, it uses the output from create_json to generate the models defined there

create_json: saves each ditcionary of parameters into a json to be loaded as an individual model

grad_cam: used for visualization, it has an adaptation for 1DCNN

visualization: loads trained models and compute relevant regions 


















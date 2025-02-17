RT-DETR-Nodule
This is the source code for our work. You can train our model by running train.py. Please note that the default output format does not include the FROC curve.

To compute and generate the FROC curve, you can use the conversion script froc.py, which processes the output from the training run to produce the desired FROC curve.

Usage Instructions
Train the Model
Run the following command to start training the model:

bash
python train.py  

Generate FROC Curve
Once training is complete, use the following command to generate the FROC curve:

bash
python froc.py  

Make sure to install all necessary dependencies before running these scripts. For more information, please refer to the project documentation or contact team members.

Thank you for your attention and support!

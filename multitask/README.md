## Multitask Learning

The notebooks in this folder are used to train and evaluate our multitask models.
- `QM9GNN2_Multitask.ipynb` has the appropriate modifications to our graph convolutional network code to allow for training and saving of models using hard or soft parameter sharing. This file also contains the methods defining model architecture and storage of multitask models. 
- `Calculate_Transfer_Coeffs.ipynb` was used to train single-task models in order to calculate transfer coefficients, which measure the effectiveness of transfer learning between tasks as a proxy for task similarity. We used these transfer coefficients to motivate our task clustering when we trained our multitask model.
- `Demo.ipynb` contains an interactive demonstration file that let's the user query properties from the QM9 dataset and compare the runtimes for DFT and neural methods.
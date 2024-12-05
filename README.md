### **Directory Breakdown**
1. **checkpoints/**
   - Stores trained model weights.
   - Files like `best_autoencoder.pth`, `best_nf_vae.pth`, and `best_vae.pth` indicate that the best-performing models for autoencoders, VAEs, and NF-VAEs are saved here.

2. **data/**
   - Contains datasets and any associated files.
   - `cifar-10-batches-py/` and `cifar-10-python.tar.gz` suggest that the CIFAR-10 dataset is being used for experiments.

3. **experiments/**
   - Houses Jupyter notebooks for running and testing models:
     - `autoencoder.ipynb`: Likely contains experiments for training and testing autoencoders.
     - `generative_adversarial_net.ipynb`: Focused on GAN experiments.
     - `nf_variational_autoencoder.ipynb`: Dedicated to normalizing flow VAEs.
     - `variational_autoencoder.ipynb`: Focused on VAEs.

4. **models/**
   - Contains Python scripts defining model architectures:
     - `ae.py`: Autoencoder implementation.
     - `vae.py`: Variational Autoencoder implementation.
     - `nf_vae.py`: Normalizing Flow Variational Autoencoder implementation.
     - `gan.py`: GAN architecture implementation.

5. **runs/**
   - Stores logs, metrics, and other output generated during model training (TensorBoard logs).

6. **README.md**
   - Provides project documentation, describing its purpose, usage, and structure.

7. **requirements.txt**
   - Lists the Python libraries and dependencies required to run the project.
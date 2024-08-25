# Colouring-B-W-Hubble-Telescope-Images-using-Pix2Pix-and-Cycle-GAN

This report discusses how the Pix2Pix Generative Adversarial Network (GAN) is used to add color to black-and-white images taken by the Hubble Space Telescope. Pix2Pix is a type of neural network made for tasks that involve translating images, and it requires matched datasets. We also look at CycleGAN, which can work without paired datasets and uses two generators and two discriminators to translate images in both directions, ensuring the images stay consistent through cycle consistency loss.

# Project Abstract

## Features

- Colourizing Black and White Astronomical images.


## Tech-Stack

- Streamlit for front end.
- Pix2Pix Gan for training the model.
- Cycle Gan for training the model.


## Deployment

To deploy this project,

1. Open the command prompt and run the following command

```bash
  pip install -r requirements.txt
```
2. On the command prompt go to the folder where you have saved the Pix2Pix and Cyclegan trained models.

3. Replace both the generator paths for the model.

3. Then run the following

```bash
  Streamlit run name_of_your_file.py

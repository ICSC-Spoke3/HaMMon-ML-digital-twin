
# Inference

Training for HaMMon was performed on the [FloodNet](https://arxiv.org/abs/2012.02951?utm_source=chatgpt.com) and [RescueNet](https://arxiv.org/abs/2202.12361?utm_source=chatgpt.com) datasets, using multiple transfer-learning and fine-tuning configurations. Full details are reported in this [technical report](https://www.openaccessrepository.it/doi/10.15161/oar.it/crt8n-kxw11).

The same applies to HaMMon-EQ, which used a combination of public datasets and original project datasets. Training settings and results are documented in this [technical report](https://www.openaccessrepository.it/doi/10.15161/oar.it/feqqg-49r54).

The trained weights are stored in the project’s internal storage. 

## Inference setup

To execute inference, download the network weights from this [google drive link](https://drive.google.com/drive/folders/1BUIZMuHx27N04ZauDcqr1QqhPEvQASp8?usp=drive_link) (if you have permissions) and place them under a `.weights` folder in the project root.


## Inference examples

Examples and code snippets for inference are available in the following notebooks:

- `notebooks/inference.ipynb`
- `notebooks/inference_patcher.ipynb`

A batch inference script is available in `utils/batch_inference.py`.  

To change model and weights, you need to modify the code directly in the script or the notebooks.


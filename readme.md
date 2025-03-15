## Disclaimer

This code is currently being migrated from a private repository and is still a work in progress. It requires cleanup and refinement.

## Project Description

This project is based on the original model described in the paper:

Jégou, S., Drozdzal, M., Vazquez, D., Romero, A., & Bengio, Y. (2017). "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*. Available at [arXiv:1611.09326](https://arxiv.org/abs/1611.09326).

The base model and several training utilities have been derived from the [PyTorch Tiramisu repository](https://github.com/bfortuner/pytorch_tiramisu). Many thanks to the authors for their contributions.

The model has been applied to two UAV image datasets related to natural hazard scenarios, [FloodNet](https://arxiv.org/abs/2012.02951) and [RescueNet](https://www.nature.com/articles/s41597-023-02799-4), and has been further adapted into a one-dimensional version for a dataset focused on detecting cracks in concrete surfaces.

Examples of training and inference tasks can be found in the /notebooks folder.


## License

The scripts in this repository are distributed under the terms of the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

This work is supported by Italian Research Center on High Performance Computing Big Data and Quantum Computing (ICSC), project funded by European Union - NextGenerationEU - and National Recovery and Resilience Plan (NRRP) - Mission 4 Component 2 within the activities of Spoke 3 (Astrophysics and Cosmos Observations





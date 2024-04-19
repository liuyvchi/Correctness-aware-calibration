# Correctness-aware Calibration

This repository contains the source code for the paper "Optimizing Calibration by Gaining Aware of Prediction Correctness." 

![Figure 1](./comparison.png)

## Prerequisites

Before you can run the scripts, you need to have Python installed along with the following packages:
- PyTorch
- torchvision

You can install these packages using pip:

```bash
pip install torch torchvision
```

## Data Preparation

To use the provided models and scripts, you will need to prepare your data according to the following directory structure:

```
├── data
│   ├── ImageNet-Val
│   ├── ImageNet-A
│   ├── ImageNet-R
│   ├── ImageNet-S
│   └── ObjectNet
└── modelOutput
    ├── imagenet_a_out_colorjitter
    │   └── tv_reesnet152.npy
    ├── imagenet_a_out_grey
    ├── imagenet_a_out_colorjitter
    └── ...
|── ckpt
└── src
```

### Directory Structure Details

- **data/**: This directory should contain all the datasets used for model calibration and testing.
- **modelOutput/**: This directory should be used to store model predictions (prediction logits). Different subdirectories within `modelOutput` can be used for different experimental configurations.
- **ckpt/**: Saaves trained calibrators.
- **src/**: Contains all the source code.

## Configuration

If you wish to run the scripts on your own data, you should modify the `option.py` file to update your configuration settings according to your local environment and data paths.

## How to Run

To start the calibration process, simply run the following command from the terminal:

```bash
python run_cal.py
```

## Figures

The repository includes four figures which are vital for understanding the output and operation of the algorithms used. Make sure to review these for better insights into the calibration results.

## Contributing

Contributions to the project are welcome. Please fork the repository, make your changes, and submit a pull request for review.

## License

This project is open source and available under the [MIT License](LICENSE.md).

## Citation

If you use this code or our results in your research, please cite as:

```bibtex
@article{YourPaper2024,
  title={Optimizing Calibration by Gaining Aware of Prediction Correctness},
  author={Author Names},
  journal={Journal Name},
  year={2024},
  publisher={Publisher}
}
```

---

This README is designed to be comprehensive, guiding users through installation, configuration, and usage, making it easier for them to get started with your project. Adjust the details as necessary based on your specific project requirements and structure.
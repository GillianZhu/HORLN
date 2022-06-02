# Hybrid-Order Representation Learning for Electricity Theft Detection
Electricity theft is the primary cause of electrical losses in power systems, existing methods usually detect anomalies in electricity consumption data on the first-order information and ignore the second-order representation learning that can efficiently model global temporal dependency and facilitates discriminative representation learning of electricity consumption data.

We propose a novel and lightweight end-to-end **Hybrid-Order Representation Learning Network (HORLN)** to identify electricity thieves. To the best of our knowledge, this is the first attempt to incorporate second-order information with regular first-order based deep architectures for electricity theft detection.

For more details, please refer to our paper [Hybrid-Order Representation Learning for Electricity Theft Detection](https://ieeexplore.ieee.org/document/9785914).


### Requirements
- python3.7
- numpy
- pytorch
- scikit_learn  (for implementation of the comparing method `RF`)
- xgboost  (for implementation of the comparing method `XGB`)
- [tsai](https://github.com/timeseriesAI/tsai) (for implementation of the comparing method `MiniRocket`)

### Dataset
We conducted our experiments on a public real-world dataset ([link](https://github.com/henryRDlab/ElectricityTheftDetection/)). We preprocessed the dataset and randomly split
it into three sets for training, validation, and testing. The preprocessed data has been saved in the `datasets/electricity.zip` file, please unzip the file and put it to the `datasets/electricity` folder.

## Train
The command to train our HORLN has been written in `Ours/train.sh`. Please run the bash file and the HORLN will be trained for 200 epochs. The parameters of the models will be saved in the `checkpoints/electricity_elec_horln` folder.
```
cd Ours
bash train_horln.sh
```

## Validate and Test
After training, the saved models of each epoch could be validated and tested by the following commands, respectively. The prediction results will be saved in the `results/electricity_elec_horln` directory, and two summary tables named `electricity_elec_horln_validation.csv` and `electricity_elec_horln_evaluation.csv` will also be generated.
```
cd Ours
bash val_horln.sh
```
```
cd Ours
bash test_horln.sh
```
## Evaluate
By validating the trained model in the validation set, the optimal value of the threshold for F1 score calculation will be automatically calculated for each epoch. Please find the threshold of the epoch you would like to evaluate (e.g., the epoch with the largest F1 score in the validation set) in the `electricity_elec_horln_validation.csv` file and revise the value of the `epoch` argument and the `best_threshold` argument in `eval.sh` accordingly. With the following command, the performance of the trained model on the testing set with the threshold determined on the validation set could be evaluated. 
```
cd Ours
bash eval_horln.sh
```

## Implementation of State-of-the-art Methods
For `RF`, `XGB` and `MiniRocket`, we adopt implementations from open-source packages and select the parameter settings based on our dataset. Please use `cd` command and change the working directory to the corresponding folders, and then run the bash file to realize training and evaluation in one step.
For `CNN` and `PFSC`, we implement the structures in previous works and revise some parameters for fair comparison; for `Wide&Deep`, and `HybridAttention`, we reimplement the official code. Please change the working directory to the corresponding folders, and then run the corresponding bash files in the order of train->val->test->eval, which is the same as running our HORLN.

## Citation
If you use this code for your research, please cite our paper.
```
@ARTICLE{HORLN,  
        author={Zhu, Yuying and Zhang, Yang and Liu, Lingbo and Liu, Yang and Li, Guan bin and Mao, Mingzhi and Lin, Liang},  
        journal={IEEE Transactions on Industrial Informatics},   
        title={Hybrid-Order Representation Learning for Electricity Theft Detection},   
        year={2022},  volume={},  number={},  pages={1-1},  
        doi={10.1109/TII.2022.3179243}}
```
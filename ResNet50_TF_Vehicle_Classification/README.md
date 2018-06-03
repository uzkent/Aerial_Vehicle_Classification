# Train the ResNet50 Model on the DIRSIG dataset

The DIRSIG vehicle classification dataset is trained on the ResNet50 model to perform binary classification. The tensorflow implementation of the ResNet50 model is slightly different
than the original model in a way that the input image is resized to `56x56` pixels and `7x7`
convolutional filter is applied on it. Also, `max pooling` step is skipped before proceeding to the
first ResNet block as it expects `56x56` pixels size feature maps. The rest is similar to the
original network.

## Execute training step

To train the model, you need to provide input to the several command line arguments in the
`resnet50_vehicle_classification.py` script. Those arguments are listed below.
1. batch_size
2. learning_rate
3. test_frequency
4. train_dir (should point to the `dirsig_train` folder in the dataset)
5. test_dir (should point to the `validation_wami` folder in the dataset)

You can execute training with the following command.
```shell
python resnet50_vehicle_classification.py --train_dir ~/GITHUB/Neural_Network_Coding_Examples/vehicle_dataset/train_dirsig/ --test_dir ~/GITHUB/Neural_Network_Coding_Examples/vehicle_dataset/validation_wami
--batch_size 32
--learning_rate 0.0001
--test_frequency 100
```
Finally, you can visualize the training stage in tensorboard with the following command.
```shell
tensorboard --logdir=./
```

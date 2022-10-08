## I2U-Net: A Novel U-Net with Rich Information Interaction for Medical Image Segmentation 

## skin lesion segmentation 
```
We take skin disease segmentation as an example to introduce the use of our model.
```

### Data preparation
resize datasets to 224*224 and saved them in npy format.
```
python data_preprocess.py
```

### Train and Test

Our method is easy to train and test,  just need to run "train_and_test_isic.py". 

```
python train_and_test_isic.py
```




README 
FMCG Product Classification with 100 labeled images


Problem Statement
In FMCG and retail computer vision (shelf auditing, planogram compliance), annotating thousands of images per new product SKU is slow and expensive. This project demonstrates how to hit production-grade accuracy with a fraction of the usual labeling effort.


Technique Used


Transfer Learning- Used model EfficientNet-B2 which is pretrained on ImageNet. It already knows shapes, textures, edges 


Selective Layer Freezing- Froze all the layers except the last convolutional block + head are trained. It prevents overfitting on small data.
 
Heavy Data Augmentation- It includes random crop, flip, rotate, colour jitter, perspective warp, random erase in this way 100 images act like thousands.


Repeat Sampling (15×)-  Each image seen 15 times per epoch with a different augmentation each time.


Mixup Augmentation- It blends pairs of images during training that results in smooth decision boundaries.


Two-Phase Training- Freeze backbone → warm up head → unfreeze and fine-tune at low LR 


Label Smoothing- It prevents overconfidence on small datasets 


Result


Training Accuracy                      ~96%
Final Test Accuracy                 ~98%
Labeled Images Used             100 (20 per class)
Model                                       EfficientNet-B2 (Transfer Learning) 
Target accuracy was                       ≥ 95% 


Dataset
Used Food101 a standard dataset built into PyTorch, easy to download.


Classes selected:
1. chocolate_mouuse
2. waffles
3. hot_and_sour_soup
4. spaghetti_carbonara
5. edamame


How to Run
Download FMCG_classifier.ipynb from the github repo(https://github.com/AYUSHI06-byte/FMCG_Classifier)
1. Open Google Colab
2. Click File → Upload notebook and upload FMCG_classifier.ipynb
3. Go to Runtime → Change runtime type → T4 GPU
4. Click Runtime → Run all


Dependencies


torch >= 2.0
torchvision >= 0.15
timm >= 0.9
scikit-learn >= 1.3
matplotlib >= 3.7
seaborn >= 0.12
numpy >= 1.24


Output Files


After running the notebook you will find:
training_curves.png     Accuracy and loss over epochs
confusion_matrix.png  Per-class prediction breakdown 
results.json                    Final accuracy metrics

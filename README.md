# GCLNSTDA
We propose a computational framework (GCLNSTDA) to predict tsRNA-disease associations. This is the implementation of GCLNSTDA: 
```
GCLNSTDA: Predicting tsRNA-Disease Association Based on Contrastive Learning and Negative Sampling.

```

# Environment Requirement
+ python == 3.9
+ torch == 1.12.0
+ numpy == 1.26.0
+ pandas == 2.1.0
+ scipy == 1.11.2


# Model
+ main.py: This file contains the main function. 
+ model.py: This file contains model building.
+ parser1.py: The paramaters of GCLNSTDA are adjusted in this file.
+ preprocessing.py: This file contains data reading.
+ utils.py: This file contains some data processing.
+  sortscore.py: The prediction score of each tsRNA-disease pair is sorted in this file before computing the AUC/AUPR.
+  ExtractFeature.py: This file contains the tsRNA-disease association prediction score calculation.

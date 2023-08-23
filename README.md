# PCDA-HNMP
  A new model, named PCDA-HNMP, is designed to identify circRNA-disease associations. The model constructs a heterogeneous network containing circRNAs, diseases, miRNAs and mRNAs as well as the associations of above four objects. The meta-paths for circRNAs, diseases and miRNAs are extracted from the heterogeneous network, from which meta-path-induced networks of circRNAs, diseases and circRNA-miRNAs are set up. From these networks, circRNA, disease, miRNA features are generated via mashup. The features are fed into XGBoost to build the model.  
![image](https://github.com/Zxy-zxy0/PCDA-HNMP/blob/master/img/FlowChart.png)
# Requirements
* python==3.8.5   
* sklearn==0.0  
* numpy==1.19.5  
* pandas==1.4.4  
* matplotlib==3.5.3  
* xgboost==1.6.1  
# Usage
Run the code in classifier to get the results of different classifiers.Train XGB_train.py to get the highest AUC.  
The features of each dimension are extracted by the mashup algorithm (http://mashup.csail.mit.edu).

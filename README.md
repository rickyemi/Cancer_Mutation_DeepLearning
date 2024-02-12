# Cancer_Mutation_DeepLearning

#### Disclaimer: Model was originally developed in June of 2018 

<center><img src="https://www.singerinstruments.com/wp-content/uploads/2015/03/what-are-mutation-1.jpg" width="1200" height="600"></center>

<b><h2><center> Deep learning for Cancer Mutation </center></h2></b>


### Overview 
- Identification of mutations contributing to cancer development is cardinal to our understanding of tumor proliferation and developing targeted therapies. Driver mutations are defined as mutations that provide a selective growth advantage, and thus promote cancer development, whereas those that do not are termed passenger mutations. Genomic instability and high mutation rates cause cancer to acquire numerous mutations and chromosomal alterations during its somatic evolution; most are termed passengers because they do not confer cancer phenotypes. Studies suggest that mildly deleterious passengers accumulate and can collectively slow cancer progression. Clinical data also suggest an association between passenger load and response to therapeutics, yet no causal link between the effects of passengers and cancer progression has been established.

### Objective

- Develop a deep learning model to(1) identify and predict driver mutation

### Data Source 
- TCGA PanCancer  using the the following features - `oncogenes` and `Tumor suppressors`, `FATHMM(missense)`, `Mutation Assessor`, `Mutation Taster`, `Polyphen-2 (v2.2.2)`,`CHASM`,`CanDrAv1.0`, `SIFT` scores etc

#### Libraries 

- Pandas
- Keras
- PyTorch
- Matplotlib and seaborn
- Sckitlearn

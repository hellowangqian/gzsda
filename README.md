# Generalized Zero-Shot Domain Adaptation via Coupled Conditional Variational Autoencoders
## Abstract
Domain adaptation aims to exploit useful information from the source domain where annotated training data are easier to obtain to address a learning problem in the target domain where only limited or even no annotated data are available. In classification problems, domain adaptation has been studied under the assumption all classes are available in the target domain regardless of the annotations. However, a common situation where only a subset of classes in the target domain are available has not attracted much attention. In this paper, we formulate this particular domain adaptation problem within a generalized zero-shot learning framework by treating the labelled source-domain samples as semantic representations for zero-shot learning. For this novel problem, neither conventional domain adaptation approaches nor zero-shot learning algorithms directly apply. To solve this problem, we present a novel Coupled Conditional Variational  Autoencoder (CCVAE) which can generate synthetic target-domain image features for unseen classes from real images in the source domain. Extensive experiments have been conducted on three domain adaptation datasets including a bespoke X-ray security checkpoint dataset to simulate a real-world application in aviation security. The results demonstrate the effectiveness of our proposed approach both against established benchmarks and in terms of real-world applicability.

## Data
All data used in our work are publicly available. \
Image features for OfficeHome, Office31 and BaggageXray20 are available from Baidu yun: \
link：https://pan.baidu.com/s/1ldG6eirNNOZtyaRAYz2u5g?pwd=nf4i \
code：nf4i\
If the raw images are needed, one can download them from https://collections.durham.ac.uk/files/r1c534fn98x \
or Baidu yun：https://pan.baidu.com/s/1voEObYqFjxaHqb5HrpeQYQ?pwd=0zdf \
code：0zdf

## How to run
One can run the command in run_xray.sh for the basic experiments on the BaggageXray20 dataset.

## Citation
@article{wang2023data,\
  title={Generalized Zero-Shot Domain Adaptation via Coupled Conditional Variational Autoencoders},\
  author={Wang, Qian and Breckon, Toby P},\
  journal={Neural Networks},\
  year={2023},\
  publisher={Elsevier}\
}

## Contact
qian.wang173@hotmail.com

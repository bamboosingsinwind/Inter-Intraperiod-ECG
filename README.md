# Inter-Intraperiod
================================
## Setup
1. Linux Operating System
2. NVIDIA GTX3090 GPU
3. Python 3.7 and packages in requirements.txt

## Data (ECG Signals)
1. BTCH (GE Marquette device). Details in "GE ECG Analysis.pdf"
2. CinC2017 https://www.cinc.org/archives/2017/pdf/065-469.pdf
3. CPSC2021 https://physionet.org/content/cpsc2021/1.0.0/

Data analysis and denoising are in "fft_denoising.py"
## Pre-training
To capture effective feature representations for AF detection from large-scale unlabeled ECG data, we propose
an inter-intra period-aware ECG pre-training method. Based on medical prior knowledge that
ECGs of atrial fibrillation patients exhibit absolute irregularity in RR intervals and the absence of P-waves, we propose
our method to capture the stable morphology representation
within single-period and valuable representation across periods by exploring interperiod and intraperiod representation,
respectively.

Detials in "pretrain.ipynb"

## Fine-tuning
During the pre-training, the model’s encoders gradually
absorb prior knowledge, gaining a comprehensive understanding of ECG from both interperiod and intraperiod
perspectives. Subsequently, the knowledge embedded in the
encoder of the pre-trained model is transferred to the downstream model by weight sharing. As illustrated in Fig. 5,
ECG signals are processed through the encoder and produce refined ECG representations. When available, these
representations are concatenated with relevant physiological features, specifically sex and age in our experiments. Finally,
they are fed into a classifier consisting of a fully-connected
(FC) layer and a Softmax layer for AF detection. When
assessing the representational capability of the SSL encoder
in practical scenarios, two approaches are considered: one
involves freezing the weights of the encoder and only updating the weights of the classifier for linear evaluation, while
the other entails updating all parameters for full fine-tuning.
In our experiments, we exclusively employ the multi-period
encoder for both evaluations, highlighting its effectiveness
and robustness in learning ECG representation. It is worth
mentioning that the single-period encoder is valuable even
for datasets consisting solely of single-period data, which
enhances the model’s scalability and applicability.

The pre trained weights are loaded for fine-tuning the atrial fibrillation detection task. Details in "finetune.py"

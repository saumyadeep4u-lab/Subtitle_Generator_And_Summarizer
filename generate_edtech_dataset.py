import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config_loader import get_logger, load_config, set_deterministic_seed

LECTURES: List[Dict[str, object]] = [
    {
        "video_id": "lecture_01_neural_networks",
        "title": "Deep Learning Fundamentals: Backpropagation & Gradient Descent",
        "split": "dev",
        "reference_summary": (
            "This lecture examines multilayer perceptrons, activation functions, and loss optimization in deep learning. "
            "It explains how forward propagation computes predictions while backpropagation applies the chain rule to calculate weight gradients. "
            "Stochastic gradient descent and adaptive optimizers like Adam adjust parameters using calibrated learning rates to minimize empirical loss. "
            "Finally, dropout regularization and batch normalization stabilize gradient flow and prevent overfitting on high-dimensional datasets."
        ),
        "sections": [
            (
                "Welcome to lecture one on deep learning fundamentals and neural network optimization. "
                "In modern machine learning, artificial neural networks model complex nonlinear relationships between input features and target labels. "
                "A multilayer perceptron consists of an input layer, multiple hidden layers, and an output layer connected by learnable weight matrices. "
                "Each artificial neuron computes a weighted linear combination of its inputs, adds a scalar bias term, and applies a nonlinear activation function. "
                "Without nonlinear activation functions such as ReLU, sigmoid, or hyperbolic tangent, a deep stack of linear layers would collapse into a single linear transformation. "
                "The rectified linear unit, or ReLU, outputs zero for negative inputs and passes positive values unchanged, which mitigates the vanishing gradient problem in deep architectures. "
                "During forward propagation, mini-batches of feature vectors pass sequentially through the network layers to produce raw logits or probability distributions."
            ),
            (
                "Moving to our second topic, we quantify prediction error using a differentiable objective or loss function. "
                "For continuous regression tasks, mean squared error penalizes large deviations between predicted and observed values. "
                "For multi-class classification, categorical cross-entropy measures the Kullback-Leibler divergence between the true label distribution and softmax probabilities. "
                "Once the forward pass computes the scalar loss, backpropagation computes the partial derivative of the loss with respect to every weight and bias in the network. "
                "Backpropagation relies on the multivariable chain rule from calculus, propagating error signals backward from the output layer toward the earliest hidden layers. "
                "By caching intermediate activations during the forward pass, automatic differentiation frameworks like PyTorch compute exact gradients in linear time relative to network depth. "
                "However, very deep networks can still suffer from exploding gradients if weight matrices have singular values significantly greater than one."
            ),
            (
                "In our third section, we explore gradient descent optimization algorithms and learning rate schedules. "
                "Full-batch gradient descent computes exact gradients over the entire training dataset before making a single parameter update, which is computationally expensive for millions of samples. "
                "Stochastic gradient descent, or SGD, approximates the true gradient using small randomized mini-batches of thirty-two to two hundred fifty-six examples. "
                "The learning rate hyperparameter controls the step size taken along the negative gradient direction at each iteration. "
                "If the learning rate is too large, optimization diverges across the loss landscape; if it is too small, convergence stalls in suboptimal local minima or saddle points. "
                "Momentum accelerates convergence by accumulating an exponentially decaying moving average of past gradients, dampening oscillations in steep ravines. "
                "Adaptive optimizers such as Adam and AdamW combine momentum with per-parameter second-moment scaling and decoupled weight decay."
            ),
            (
                "In our fourth section, we examine regularization strategies and batch normalization for generalization on unseen test data. "
                "Overfitting occurs when a high-capacity neural network memorizes training noise rather than learning underlying statistical structure. "
                "L2 weight decay adds a quadratic penalty on parameter magnitudes directly to the loss function, encouraging smoother decision boundaries. "
                "Dropout randomly zeroes out a fraction of hidden neuron activations during each training step, forcing the network to learn redundant distributed representations. "
                "At inference time, dropout is disabled and weights are scaled proportionally to preserve expected activation magnitudes. "
                "Batch normalization normalizes intermediate layer activations across each mini-batch to zero mean and unit variance, followed by learnable scale and shift parameters. "
                "Together, proper weight initialization, batch normalization, dropout, and cosine annealing learning rate schedules form the foundation of modern neural network training."
            ),
            (
                "In section five, let us conduct a detailed mathematical walkthrough of how matrix dimensions and tensor shapes interact during mini-batch training. "
                "Suppose we process a mini-batch of sixty-four examples, where each example is represented by a five-hundred-and-twelve dimensional input feature vector. "
                "The input tensor therefore has shape sixty-four by five hundred and twelve. "
                "If our first hidden layer contains two hundred and fifty-six neurons, the weight matrix connecting the input layer to the first hidden layer has shape five hundred and twelve by two hundred and fifty-six, accompanied by a bias vector of length two hundred and fifty-six. "
                "During the forward pass, broadcasting adds the bias vector to every row of the matrix product, yielding an intermediate representation of shape sixty-four by two hundred and fifty-six. "
                "When computing the backward pass, the upstream gradient arriving at this layer also has shape sixty-four by two hundred and fifty-six. "
                "To compute the gradient with respect to the weight matrix, we multiply the transposed input tensor of shape five hundred and twelve by sixty-four against the upstream gradient of shape sixty-four by two hundred and fifty-six, producing a weight gradient tensor that matches the exact dimensions of the parameter matrix."
            ),
            (
                "In section six, we analyze the geometric intuition behind learning rate schedules and momentum in high-dimensional non-convex optimization. "
                "In a parameter space containing millions of dimensions, local minima are statistically rare compared to saddle points, which are regions where the Hessian matrix exhibits both positive and negative eigenvalues. "
                "Standard gradient descent without momentum can stall near flat plateaus because the gradient magnitude approaches zero in all directions simultaneously. "
                "Classical Polyak momentum and Nesterov accelerated gradient overcome this limitation by accumulating velocity along consistent descent directions while dampening oscillations across high-curvature walls. "
                "When paired with a linear warmup phase during the first few hundred iterations, adaptive optimizers prevent early gradient spikes from distorting the second-moment variance estimates before the network weights have settled into a stable basin of attraction."
            ),
            (
                "In section seven, we review the role of activation function geometry in modern neural architecture design. "
                "While the classic rectified linear unit outputs zero for all negative inputs and identity for positive inputs, it suffers from the dying ReLU phenomenon whenever a large gradient update pushes a neuron bias strongly negative, causing it to output zero for all subsequent training instances. "
                "Leaky ReLU, Parametric ReLU, and Exponential Linear Units mitigate this issue by preserving a small non-zero slope in the negative regime. "
                "Even more effective in modern transformer and convolutional backbones are smooth gating activations such as Swish, SiLU, and Gaussian Error Linear Units, which weight inputs by their cumulative Gaussian distribution value, providing smooth derivatives everywhere and improving empirical convergence across deep architectures."
            ),
            (
                "In our eighth and concluding section, we synthesize these foundational deep learning principles into a structured engineering checklist for model development. "
                "First, always verify data normalization so input features have zero mean and unit variance before entering the first linear layer. "
                "Second, overfit a single mini-batch of eight to sixteen samples to zero training loss as a sanity check to confirm that gradient flow and loss computation are bug-free. "
                "Third, monitor layer-wise activation histograms and gradient norms throughout training to detect saturation or numerical instability early. "
                "By combining mathematical rigor in backpropagation with disciplined weight initialization, layer normalization, and adaptive optimization, engineers build resilient neural architectures capable of scaling to complex real-world problems."
            ),
        ],
    },
    {
        "video_id": "lecture_02_nlp_transformers",
        "title": "Natural Language Processing: Self-Attention & Transformer Models",
        "split": "dev",
        "reference_summary": (
            "This lecture traces the evolution of natural language processing from recurrent networks to the Transformer architecture. "
            "It details how scaled dot-product self-attention computes contextual token representations in parallel using query, key, and value matrices. "
            "Multi-head attention and positional encodings capture both syntactic dependencies and word order across long sequences. "
            "Finally, encoder-only models like BERT and encoder-decoder models like BART and T5 power modern classification and summarization pipelines."
        ),
        "sections": [
            (
                "Welcome to lecture two on natural language processing and the Transformer architecture. "
                "Historically, sequential text modeling relied on recurrent neural networks, long short-term memory cells, and gated recurrent units. "
                "Recurrent architectures process tokens one step at a time from left to right, compressing prior context into a fixed-size hidden state vector. "
                "This sequential dependency creates two severe bottlenecks: training cannot be parallelized across sequence positions, and long-range dependencies degrade over hundreds of tokens. "
                "Subword tokenization algorithms such as Byte-Pair Encoding and WordPiece first segment raw text into frequent subword units to handle rare words and morphology without out-of-vocabulary errors. "
                "Each subword token index is mapped to a dense continuous vector via a learnable embedding lookup table. "
                "Because the Transformer removes recurrence entirely, sinusoidal or learned positional encodings are added to token embeddings so the model can distinguish word order."
            ),
            (
                "In section two, we analyze scaled dot-product self-attention, the core computational mechanism of the Transformer. "
                "For every token in an input sequence, the model projects its embedding into three distinct vectors: a Query vector, a Key vector, and a Value vector. "
                "To determine how much attention token A should pay to token B, we compute the dot product between the Query of token A and the Key of token B. "
                "These raw compatibility scores are divided by the square root of the key dimension to prevent large dot products from pushing the softmax function into regions with tiny gradients. "
                "Applying the softmax operation across all sequence positions yields normalized attention weights that sum to one. "
                "Each token then computes a weighted sum of Value vectors, allowing direct contextual information exchange between any pair of words regardless of distance. "
                "Multi-head attention runs several self-attention projections in parallel so different heads can attend to syntactic roles, coreference, and semantic relations simultaneously."
            ),
            (
                "In section three, we examine the internal block structure of Transformer encoders and decoders. "
                "Each Transformer encoder layer wraps the multi-head self-attention module and a position-wise feed-forward network inside residual skip connections and layer normalization. "
                "Residual connections allow gradients to flow directly through deep stacks of twelve to ninety-six layers without degradation. "
                "In the decoder stack, causal masking sets upper-triangular attention logits to negative infinity so autoregressive generation cannot peek at future tokens. "
                "Cross-attention layers in encoder-decoder architectures allow the decoder queries to attend directly over the contextualized representations produced by the encoder. "
                "The computational complexity of standard self-attention scales quadratically with sequence length, motivating sliding-window and flash-attention optimizations for long documents."
            ),
            (
                "In section four, we compare pretrained Transformer paradigms including BERT, BART, and FLAN-T5. "
                "BERT is a bidirectional encoder pretrained using masked language modeling, making it ideal for token classification, named entity recognition, and semantic similarity. "
                "BART combines a bidirectional encoder with an autoregressive left-to-right decoder, pretrained as a denoising autoencoder by reconstructing corrupted text spans. "
                "This denoising objective makes BART exceptionally effective for abstractive document summarization and text generation. "
                "T5, the Text-to-Text Transfer Transformer, reframes every natural language processing task as sequence-to-sequence generation using task-specific text prefixes. "
                "Instruction-tuned variants like FLAN-T5 generalize across unseen summarization and question-answering tasks by training on thousands of diverse instruction templates."
            ),
            (
                "In section five, we examine subword tokenization algorithms in detail and their impact on vocabulary efficiency and sequence length. "
                "Word-level vocabularies struggle with out-of-vocabulary terms, morphological variations, and typographical errors, whereas character-level tokenization produces excessively long sequences that strain quadratic self-attention memory limits. "
                "Modern transformers resolve this trade-off using subword tokenization methods including Byte-Pair Encoding, WordPiece, and Unigram SentencePiece. "
                "Byte-Pair Encoding iteratively merges the most frequent pairs of adjacent bytes or characters until a target vocabulary size between thirty thousand and fifty thousand subword tokens is reached. "
                "Common words remain single tokens, while rare technical terms such as backpropagation or heteroskedasticity are decomposed into meaningful root and suffix subwords, eliminating unknown token errors entirely."
            ),
            (
                "In section six, we analyze the computational complexity of self-attention and how hierarchical segmentation handles long contexts. "
                "When an encoder processes a sequence of length N with hidden dimension D, computing the pairwise Query-Key attention matrix requires order N squared times D floating-point operations and order N squared memory to store attention weights. "
                "Doubling the input sequence length quadruples the memory footprint of the attention map. "
                "Standard pretrained models enforce strict positional embedding limits of five hundred and twelve or one thousand and twenty-four tokens. "
                "By combining semantic topic segmentation with bounded five-hundred-and-twelve token windows and a sliding sentence overlap, hierarchical summarization pipelines keep memory consumption linear with respect to total lecture duration while preserving cross-boundary semantic continuity."
            ),
            (
                "In section seven, we compare extractive summarization baselines against abstractive sequence-to-sequence models in educational applications. "
                "Graph-based extractive algorithms such as LexRank and TextRank represent sentences as nodes in a similarity graph, where edge weights reflect cosine similarity between term-frequency vectors, and apply eigenvector centrality to select the most salient verbatim sentences. "
                "While extractive summaries guarantee that individual phrases are copied directly from the transcript, they often suffer from dangling pronouns, abrupt topic transitions, and conversational disfluencies inherited from spoken lectures. "
                "Abstractive models like BART-Large-CNN and FLAN-T5 rewrite and compress multi-sentence explanations into cohesive, grammatically polished academic prose that reads like human-authored study notes."
            ),
            (
                "In our eighth and final section, we review the mathematical formulation of ROUGE and BERTScore for evaluating generated summaries. "
                "ROUGE-1 and ROUGE-2 measure unigram and bigram lexical overlap between the candidate summary and reference summary, whereas ROUGE-L computes the longest common subsequence to capture sentence-level word order. "
                "Because lexical overlap metrics penalize valid paraphrases that use synonyms, we complement ROUGE with BERTScore, which computes pairwise cosine similarity between contextualized token embeddings of the reference and candidate texts. "
                "Evaluating both lexical overlap and semantic embedding alignment provides a rigorous, multi-dimensional assessment of summarization quality across educational lectures."
            ),
        ],
    },
    {
        "video_id": "lecture_03_supervised_learning",
        "title": "Supervised Learning: Bias-Variance Tradeoff & Model Evaluation",
        "split": "test",
        "reference_summary": (
            "This lecture covers supervised machine learning paradigms, decomposing generalization error into bias, variance, and irreducible noise. "
            "It contrasts linear models and support vector machines with ensemble methods including Random Forests and Gradient Boosted Decision Trees. "
            "Strict train, validation, and held-out test partitioning prevents data leakage during hyperparameter tuning. "
            "Finally, precision, recall, F1 score, and ROC-AUC provide calibrated evaluation under severe class imbalance."
        ),
        "sections": [
            (
                "Welcome to lecture three on supervised machine learning, generalization theory, and rigorous evaluation methodology. "
                "In supervised learning, an algorithm observes a training dataset of feature vectors paired with ground-truth target labels and estimates a mapping function. "
                "Expected generalization error on unseen test examples decomposes mathematically into three components: squared bias, variance, and irreducible noise. "
                "Bias measures systematic error caused by overly restrictive model assumptions, such as fitting a linear hyperplane to a highly curved decision boundary, resulting in underfitting. "
                "Variance measures how sensitively the learned model fluctuates when trained on different random samples drawn from the same underlying population. "
                "High-capacity models with too many degrees of freedom exhibit low training error but high variance, resulting in severe overfitting. "
                "Regularization parameters such as Ridge L2 penalty and Lasso L1 penalty explicitly trade a small increase in bias for a substantial reduction in variance."
            ),
            (
                "In section two, we compare linear classifiers, support vector machines, and tree-based ensemble algorithms. "
                "Logistic regression models the log-odds of binary class membership as a linear combination of input features, optimized via maximum likelihood estimation. "
                "Support vector machines construct a maximum-margin separating hyperplane and leverage the kernel trick to project non-linearly separable data into higher-dimensional spaces. "
                "Decision trees recursively partition the feature space along axis-aligned splits that maximize information gain or minimize Gini impurity. "
                "While individual deep decision trees suffer from high variance, Random Forests aggregate predictions across hundreds of bootstrapped trees trained with random feature subsets. "
                "Gradient Boosted Decision Trees, implemented in frameworks like XGBoost and LightGBM, build shallow trees sequentially where each new tree fits the negative gradient of the prior ensemble error."
            ),
            (
                "Section three focuses on dataset partitioning, cross-validation, and preventing data leakage in applied data science. "
                "A common methodological flaw is tuning model hyperparameters or feature selection rules against the final evaluation set, which leaks test information and inflates reported accuracy. "
                "Production machine learning workflows enforce a strict three-way split into training, development validation, and unseen held-out test partitions. "
                "All preprocessing transformers, such as standard scalers, missing-value imputers, and vocabulary encoders, must be fit exclusively on the training split and applied unchanged to validation and test data. "
                "In k-fold stratified cross-validation, the development data is partitioned into k folds preserving class ratios, rotating the validation fold k times to estimate mean and standard deviation performance."
            ),
            (
                "In section four, we analyze classification metrics beyond simple accuracy for imbalanced datasets. "
                "When positive fraud or disease cases represent less than one percent of observations, a trivial classifier predicting only the majority class achieves ninety-nine percent accuracy while failing completely. "
                "Precision measures the proportion of positive predictions that are truly positive, controlling false-positive alert fatigue. "
                "Recall, or sensitivity, measures the proportion of actual positive instances successfully recovered by the model, controlling false-negative misses. "
                "The F1 score computes the harmonic mean of precision and recall, penalizing extreme imbalances between the two metrics. "
                "By sweeping the classification probability threshold from zero to one, engineers plot Precision-Recall and Receiver Operating Characteristic curves to select an empirically justified operating point."
            ),
            (
                "In section five, we delve deeper into the mathematical derivation of second-order gradient boosting used in modern tabular learning engines. "
                "At boosting iteration T, our objective is to find a new regression tree function that minimizes the regularized empirical loss over all training examples, plus a complexity penalty on the number of terminal leaves and the squared L2 norm of leaf weights. "
                "By expanding the loss function to the second order using a Taylor series approximation around the current ensemble prediction, the objective simplifies into a sum of linear first-order gradient statistics and quadratic second-order Hessian statistics for every data point. "
                "Grouping instances by the terminal leaf they fall into allows us to derive a closed-form analytical formula for both the optimal leaf weight and the exact reduction in loss achieved by any candidate feature split."
            ),
            (
                "In section six, we examine subtle data leakage patterns that frequently invalidate supervised learning benchmarks in industry projects. "
                "A common mistake occurs when feature scaling, missing-value imputation, or target encoding is computed across the entire dataset prior to splitting into training and test sets. "
                "When global means, standard deviations, or target statistics incorporate information from test instances, the model gains illicit access to future distribution properties, artificially inflating validation metrics. "
                "In time-series forecasting and financial modeling, random shuffling across time introduces look-ahead bias; engineers must instead use expanding-window temporal splits so training data always precedes validation data chronologically."
            ),
            (
                "In section seven, we analyze probability calibration and decision threshold optimization for deployment in risk-sensitive applications. "
                "While models like logistic regression naturally output well-calibrated posterior probabilities, maximum-margin classifiers, random forests, and deep neural networks often produce distorted confidence scores that cluster near zero and one. "
                "Platt scaling fits a monotonic sigmoid transformation on held-out validation logits, while isotonic regression fits a non-parametric piecewise-constant non-decreasing function to align predicted probabilities with empirical outcome frequencies. "
                "Once probabilities are calibrated using Brier score and reliability diagrams, analysts select optimal operating thresholds based on asymmetric business cost matrices."
            ),
            (
                "In our eighth and final section, we summarize best practices for feature engineering and model interpretability in supervised learning workflows. "
                "Domain-specific interaction ratios, cyclical trigonometric encodings for timestamps, and out-of-fold target statistics often yield larger performance gains than switching model families. "
                "To explain complex ensemble predictions to stakeholders and regulatory auditors, SHapley Additive exPlanations, or SHAP values, compute game-theoretic marginal contributions of each feature across all possible feature coalitions. "
                "Combining leakage-free cross-validation, calibrated ensemble models, and SHAP attribution guarantees transparent and dependable predictive performance."
            ),
        ],
    },
    {
        "video_id": "lecture_04_computer_vision_cnns",
        "title": "Computer Vision: Convolutional Neural Networks & Object Detection",
        "split": "test",
        "reference_summary": (
            "This lecture introduces computer vision architectures, explaining how convolutional neural networks exploit spatial locality and weight sharing. "
            "It describes how convolutional kernels, stride, padding, and pooling layers extract hierarchical visual features from raw pixel tensors. "
            "Residual connections in ResNet overcome degradation in deep vision backbones and enable transfer learning. "
            "Finally, modern object detection and segmentation frameworks localize and classify multiple instances in real time."
        ),
        "sections": [
            (
                "Welcome to lecture four on computer vision and convolutional neural network architectures. "
                "Digital images are represented as three-dimensional tensors defined by height, width, and color channels such as red, green, and blue. "
                "Connecting a standard fully connected layer directly to a high-resolution image requires hundreds of millions of parameters and destroys spatial topology. "
                "Convolutional neural networks, or CNNs, solve this problem through three architectural priors: local receptive fields, spatial weight sharing, and translation equivariance. "
                "A learnable convolutional filter, typically three-by-three or five-by-five pixels in size, slides across the spatial dimensions of the input tensor to compute feature maps. "
                "Because the exact same filter weights are shared across every spatial location, the network detects visual patterns regardless of where they appear in the frame."
            ),
            (
                "In section two, we examine spatial dimensions, padding, stride, and pooling operations inside a convolutional backbone. "
                "Zero-padding adds a border of zeros around the input perimeter so convolutional filters can process edge pixels without shrinking spatial resolution prematurely. "
                "Stride defines the step size by which the convolutional kernel shifts horizontally and vertically across the image grid. "
                "After convolution and batch normalization, nonlinear activation functions introduce sparsity into the resulting activation maps. "
                "Max pooling and average pooling downsample feature maps by aggregating local neighborhoods, reducing memory footprint and providing invariance to small spatial shifts. "
                "Early convolutional layers learn low-level edge, corner, and color-gradient detectors, whereas deeper layers compose these primitives into textures, object parts, and complete semantic categories."
            ),
            (
                "In section three, we study deep residual networks, transfer learning, and vision backbones. "
                "As early architectures scaled beyond twenty layers, researchers observed a counterintuitive degradation problem where training accuracy saturated and then worsened rapidly. "
                "ResNet resolved this bottleneck by introducing identity skip connections that add the input of a convolutional block directly to its output. "
                "Instead of learning an unreferenced target mapping from scratch, residual blocks learn a residual perturbation relative to the identity function. "
                "In real-world applications with limited labeled images, engineers initialize vision pipelines with ResNet or Vision Transformer weights pretrained on millions of labeled images. "
                "By freezing early feature-extraction layers and fine-tuning only the final classification head with data augmentation, models converge rapidly with high accuracy."
            ),
            (
                "In section four, we extend image classification to object detection and semantic segmentation. "
                "Whereas image classification assigns a single category label to an entire frame, object detection outputs both bounding box coordinates and class probabilities for every object present. "
                "Two-stage detectors generate candidate region proposals before refining box regressions, while single-stage detectors like YOLO predict bounding boxes and classes in a single forward pass. "
                "Non-maximum suppression filters out redundant overlapping bounding boxes by sorting predictions by confidence and discarding boxes whose Intersection over Union exceeds a configured threshold. "
                "Mean Average Precision across multiple Intersection over Union thresholds serves as the standard benchmark metric for evaluating object detection systems."
            ),
            (
                "In section five, we perform a concrete calculation of convolutional output dimensions and receptive field growth across stacked layers. "
                "Given an input image of size two hundred and twenty-four by two hundred and twenty-four with three RGB channels, applying sixty-four convolutional filters of size three-by-three with a padding of one pixel and a stride of one preserves the exact spatial dimensions of two hundred and twenty-four by two hundred and twenty-four while producing sixty-four output feature channels. "
                "Each three-by-three filter contains three times three times three equals twenty-seven weights plus one bias term, totaling just one thousand seven hundred and ninety-two parameters for the entire layer. "
                "Furthermore, stacking two three-by-three convolutional layers back-to-back achieves the exact same five-by-five effective receptive field as a single five-by-five convolution, while requiring twenty-eight percent fewer parameters and introducing an additional non-linear activation in between."
            ),
            (
                "In section six, we examine how depthwise separable convolutions enable real-time computer vision on mobile and edge devices. "
                "Standard convolution performs spatial filtering and cross-channel feature combination simultaneously in a single heavy operation. "
                "MobileNet architectures factorize this operation into two lightweight steps: first, a depthwise convolution applies a single spatial filter independently to each input channel; second, a pointwise one-by-one convolution projects and combines the channel outputs linearly. "
                "For a three-by-three kernel, depthwise separable convolution reduces floating-point multiply-accumulate operations by a factor of eight to nine times with only a minimal trade-off in top-one classification accuracy."
            ),
            (
                "In section seven, we explore encoder-decoder segmentation architectures and Feature Pyramid Networks for multi-scale visual recognition. "
                "Recognizing both tiny distant objects and large foreground objects requires combining high-resolution spatial detail from early layers with rich semantic context from deep layers. "
                "U-Net and Feature Pyramid Networks construct top-down lateral pathways that upsample deep semantic feature maps and fuse them element-wise with shallow high-resolution feature maps. "
                "To train pixel-wise segmentation models under class imbalance between foreground objects and background sky or road pixels, engineers combine cross-entropy loss with Dice loss and Focal loss."
            ),
            (
                "In our eighth and concluding section, we review advanced data augmentation paradigms and Vision Transformers. "
                "Beyond classical random horizontal flips, random resized crops, and color jittering, modern vision training pipelines incorporate Mixup and CutMix regularization. "
                "Mixup forms synthetic training examples by taking convex linear combinations of pairs of images and their corresponding label vectors, while CutMix pastes rectangular patches across training images. "
                "Combined with Vision Transformers that partition images into sixteen-by-sixteen pixel patches processed via multi-head self-attention, modern computer vision systems achieve human-level perception across medical imaging, autonomous navigation, and industrial inspection."
            ),
        ],
    },
    {
        "video_id": "lecture_05_mlops_deployment",
        "title": "Production AI & MLOps: Reproducibility, Testing & Drift Monitoring",
        "split": "test",
        "reference_summary": (
            "This lecture outlines production MLOps engineering practices for deploying reliable and reproducible artificial intelligence systems. "
            "It emphasizes deterministic seeding, externalized configuration management, and artifact versioning with cryptographic checksums. "
            "Automated unit and boundary test suites validate preprocessing, model inference, and post-processing constraints before deployment. "
            "Finally, continuous telemetry tracks latency, throughput, and data drift to trigger automated retraining workflows."
        ),
        "sections": [
            (
                "Welcome to lecture five on MLOps engineering, reproducibility, and production machine learning systems. "
                "Transitioning a machine learning prototype from an experimental notebook into a reliable enterprise system requires rigorous software engineering discipline. "
                "Reproducibility begins by fixing deterministic random seeds across Python, NumPy, and PyTorch so dataset splits and weight initializations produce identical results across runs. "
                "All hyperparameters, file paths, token limits, and operating thresholds must be centralized in an external configuration file rather than hardcoded inside functions. "
                "When configuration parameters drift from runtime implementation, evaluation benchmarks become unverifiable and production debugging fails. "
                "Model checkpoints and datasets should be tracked with explicit version tags, training timestamps, and SHA-256 cryptographic checksums to guarantee artifact provenance."
            ),
            (
                "In section two, we examine modular repository design and automated software testing for machine learning pipelines. "
                "Monolithic scripts that mix user interface code, data ingestion, model inference, and evaluation logic are brittle and difficult to maintain. "
                "Production repositories separate concerns into dedicated modules for configuration loading, data preprocessing, core model inference, formatting, and quantitative evaluation. "
                "Automated test suites built with pytest verify both unit functionality and boundary conditions before any code change is merged. "
                "For natural language and speech pipelines, boundary tests verify behavior on empty inputs, single-word strings, overlapping timestamps, and maximum token ceilings. "
                "Regression tests ensure that updates to heuristic post-processors or configuration thresholds do not silently degrade established precision or formatting standards."
            ),
            (
                "In section three, we focus on inference optimization, fault tolerance, and operational telemetry. "
                "Production services must handle transient external failures gracefully using structured logging, configurable timeouts, and exponential backoff retry mechanisms. "
                "When deploying speech recognition and transformer summarization models, engineers track operational metrics alongside statistical quality metrics. "
                "Key telemetry indicators include end-to-end latency, ninety-fifth percentile response time, throughput in items per second, and Real-Time Factor for audio processing. "
                "Model quantization, batched inference, and mixed-precision execution reduce memory consumption and accelerate throughput on both CPU and GPU hardware."
            ),
            (
                "In section four, we address post-deployment monitoring, data drift detection, and human-in-the-loop feedback loops. "
                "Unlike static software, machine learning models degrade over time as real-world input distributions shift away from historical training data. "
                "Covariate shift occurs when the distribution of input features changes, such as new technical terminology or background acoustics appearing in university lectures. "
                "Concept drift occurs when the statistical relationship between inputs and target outputs evolves over time. "
                "Production monitoring dashboards continuously compute statistical distance metrics such as Population Stability Index and Kolmogorov-Smirnov tests on incoming batches. "
                "When drift or low-confidence fallback rates exceed calibrated alert thresholds, the MLOps pipeline flags samples for human annotation and triggers scheduled retraining."
            ),
            (
                "In section five, we examine containerization, dependency pinning, and continuous integration workflows for machine learning deployment. "
                "A common source of production failures is environment skew between a data scientist local workstation and the cloud inference server. "
                "By pinning exact package versions in requirements files and encapsulating the runtime environment inside a container image, teams ensure that system libraries such as FFmpeg, audio codecs, and Python interpreters remain identical across development, staging, and production clusters. "
                "Continuous integration pipelines automatically execute linter checks, type-annotation verification, and unit test suites on every commit, blocking broken deployments before they reach end users."
            ),
            (
                "In section six, we analyze shadow deployments, canary releases, and A-B testing strategies for validating new model versions safely in production. "
                "Replacing a live production model all at once carries significant operational risk if the new checkpoint exhibits unexpected latency spikes or edge-case hallucinations on live traffic. "
                "In a shadow deployment, live incoming requests are mirrored asynchronously to the candidate model in the background, allowing engineers to compare its predictions and latency against the incumbent production model without exposing users to unverified outputs. "
                "Once shadow validation passes, a canary rollout routes five percent of live traffic to the new model and gradually ramps traffic up to one hundred percent while monitoring real-time error rates."
            ),
            (
                "In section seven, we explore the architecture of feature stores and metadata registries in enterprise machine learning platforms. "
                "Training-serving skew occurs when feature transformations computed during offline batch training differ subtly from real-time feature computations executed during online inference. "
                "A centralized feature store eliminates training-serving skew by defining feature transformation logic once and serving both historical point-in-time training snapshots and low-latency online inference lookups from a unified repository. "
                "Complementing the feature store, a model registry records lineage metadata linking every deployed model binary to its exact commit hash, training dataset checksum, hyperparameter configuration, and validation metric report."
            ),
            (
                "In our eighth and final section, we review incident response and graceful degradation patterns for resilient artificial intelligence services. "
                "When an upstream deep learning model encounters corrupted media files, out-of-memory conditions, or hardware timeouts, a fault-tolerant pipeline never crashes silently or returns malformed outputs to the client. "
                "Instead, circuit breakers trip automatically to route requests to deterministic fallback heuristics, such as falling back from a heavy abstractive transformer to a fast extractive LexRank summarizer, while emitting structured warning logs for on-call engineers. "
                "Combining deterministic reproducibility, automated boundary testing, telemetry monitoring, and graceful fallbacks completes the lifecycle of a production-grade AI system."
            ),
        ],
    },
]


def _render_lecture_slide(
    video_id: str, title: str, duration_min: float, word_count: int, slide_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.axis("off")

    ax.text(
        0.06, 0.82, "NEWTON AI • EDTECH LECTURE SERIES",
        color="#38BDF8", fontsize=14, fontweight="bold", transform=ax.transAxes
    )
    ax.text(
        0.06, 0.62, title,
        color="#F8FAFC", fontsize=22, fontweight="bold", wrap=True, transform=ax.transAxes
    )
    ax.text(
        0.06, 0.40,
        f"Module ID: {video_id}\n"
        f"Duration: {duration_min:.1f} Minutes Continuous Spoken Lecture ({word_count} Words)\n"
        f"Audio Format: 16kHz Mono PCM • Broadcast SRT + Abstractive Summary",
        color="#94A3B8", fontsize=14, transform=ax.transAxes
    )
    ax.text(
        0.06, 0.12, "Instructor: Automated Curriculum Audio Engine | Prepared by Saumyadeep Bardhan",
        color="#64748B", fontsize=12, transform=ax.transAxes
    )
    plt.tight_layout()
    fig.savefig(slide_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def _synthesize_wav_isolated(text: str, out_wav: Path, rate: int = 135) -> None:
    helper_code = (
        "import sys, pyttsx3; "
        "engine = pyttsx3.init(); "
        f"engine.setProperty('rate', {rate}); "
        "engine.setProperty('volume', 1.0); "
        "engine.save_to_file(sys.stdin.read(), sys.argv[1]); "
        "engine.runAndWait(); "
        "engine.stop()"
    )
    subprocess.run(
        [sys.executable, "-c", helper_code, str(out_wav)],
        input=text,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=180,
    )


def _build_10min_lecture_audio(
    sections: List[str],
    out_wav_path: Path,
    target_duration_sec: float = 620.0,
) -> str:
    paragraphs = list(sections)
    idx = 0
    while len(" ".join(paragraphs).split()) < 1480:
        sec_idx = idx % len(sections)
        paragraphs.append(
            f"To reinforce key concepts from section {sec_idx + 1}, let us review the core principles in detail. "
            + sections[sec_idx]
        )
        idx += 1

    full_spoken_script = " ".join(paragraphs)
    temp_raw = out_wav_path.parent / f"_raw_{out_wav_path.name}"
    norm_wav = out_wav_path.parent / f"_norm_{out_wav_path.name}"

    _synthesize_wav_isolated(full_spoken_script, temp_raw, rate=135)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(temp_raw),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(norm_wav),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with wave.open(str(norm_wav), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()

    speech_dur = n_frames / float(sr)

    # Dynamically adjust tempo so continuous spoken narration fills the entire 10-12 minute video
    if abs(speech_dur - target_duration_sec) > 5.0:
        tempo_ratio = np.clip(speech_dur / target_duration_sec, 0.70, 1.25)
        stretched_wav = out_wav_path.parent / f"_stretched_{out_wav_path.name}"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(norm_wav),
                "-filter:a", f"atempo={tempo_ratio:.4f}",
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(stretched_wav),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        norm_wav.unlink(missing_ok=True)
        norm_wav = stretched_wav

    if out_wav_path.exists():
        out_wav_path.unlink()
    norm_wav.replace(out_wav_path)
    temp_raw.unlink(missing_ok=True)

    return full_spoken_script


def main() -> None:
    config = load_config("config.yaml")
    set_deterministic_seed(config["project"]["random_seed"])
    logger = get_logger("DatasetGenerator", config)

    raw_videos_dir = Path(config["paths"]["raw_videos_dir"])
    gt_dir = Path(config["paths"]["ground_truth_dir"])
    raw_videos_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    old_dummy = raw_videos_dir / "lecture_01.wav"
    if old_dummy.exists():
        old_dummy.unlink()

    metadata_records = []

    for idx, lec in enumerate(LECTURES, start=1):
        vid_id = str(lec["video_id"])
        title = str(lec["title"])
        sections = list(lec["sections"])  # type: ignore
        ref_summary = str(lec["reference_summary"])
        split_name = str(lec["split"])

        logger.info(f"[{idx}/5] Generating full 10-12 min spoken lecture video: {vid_id}.mp4 ...")

        slide_png = raw_videos_dir / f"_{vid_id}_slide.png"
        temp_wav = raw_videos_dir / f"_{vid_id}_speech.wav"
        output_mp4 = raw_videos_dir / f"{vid_id}.mp4"

        target_dur = 610.0 + (idx * 12.0)
        full_reference_text = _build_10min_lecture_audio(
            sections, temp_wav, target_duration_sec=target_dur
        )

        with wave.open(str(temp_wav), "rb") as wf:
            actual_dur = wf.getnframes() / float(wf.getframerate())

        word_count = len(full_reference_text.split())
        _render_lecture_slide(vid_id, title, actual_dur / 60.0, word_count, slide_png)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-framerate", "1", "-i", str(slide_png),
                "-i", str(temp_wav),
                "-c:v", "libx264", "-tune", "stillimage", "-preset", "ultrafast",
                "-crf", "28", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest", str(output_mp4),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        slide_png.unlink(missing_ok=True)
        temp_wav.unlink(missing_ok=True)

        ref_txt_path = gt_dir / f"{vid_id}_ref.txt"
        ref_sum_path = gt_dir / f"{vid_id}_ref_summary.txt"
        ref_sec_path = gt_dir / f"{vid_id}_sections.json"

        ref_txt_path.write_text(full_reference_text + "\n", encoding="utf-8")
        ref_sum_path.write_text(ref_summary + "\n", encoding="utf-8")
        ref_sec_path.write_text(json.dumps(sections, indent=2), encoding="utf-8")

        metadata_records.append(
            {
                "video_id": vid_id,
                "title": title,
                "split": split_name,
                "video_path": str(output_mp4),
                "target_duration_sec": round(actual_dur, 2),
                "reference_words": word_count,
                "reference_summary_words": len(ref_summary.split()),
                "num_ground_truth_sections": len(sections),
            }
        )
        logger.info(
            f"Completed {output_mp4.name} ({actual_dur / 60.0:.2f} mins continuous audio | {word_count} spoken words)"
        )

    meta_path = raw_videos_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata_records, indent=2), encoding="utf-8")
    logger.info(f"All 5 EdTech lecture videos and ground-truth references saved! Metadata: {meta_path}")


if __name__ == "__main__":
    main()
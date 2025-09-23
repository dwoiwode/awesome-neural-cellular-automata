<!-- This file was automatically created on 2025-09-23 00:57:36 UTC. Any manual changes will be lost! -->
# Awesome Neural Cellular Automata
A list of paper and resources regarding Neural Cellular Automata. Last updated: 2025-06-30.

> [!NOTE]
> This repository has been researched, compiled, and maintained to the best of my knowledge and ability.
> While I strive for accuracy and completeness, there may be mistakes or updates needed over time.
> I welcome suggestions, improvements, and contributions from the community.
> Please feel free to open an issue or submit a pull request if you have any recommendations or changes.
>
> To contribute, please update the `papers.yaml` and `Readme.md.jinja` files.
> Running `yaml2md.py` generates the `Readme.md`, and any manual edits to `Readme.md` will be overwritten.
>
> Thank you for your support and collaboration!

## Seminal Paper introducing Neural Cellular Automata
<table>
<tr>
<td width="150px">
<a href="https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb" target="_blank">
    <img src="assets/thumbnails/2020-02-11growingneu_mordvintsev.jpg" width="140px">
</a>
</td>
<td>

### Growing Neural Cellular Automata
Published on **2020-02-11** by

Alexander **Mordvintsev**, Ettore **Randazzo**, Eyvind **Niklasson**, Michael **Levin**

[Code](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb) | [Project Page](https://distill.pub/2020/growing-ca/)

<details>
<summary><b>Abstract</b></summary>
Training an end-to-end differentiable, self-organising cellular automata model of morphogenesis, able to both grow and regenerate specific patterns.
</details>

</td>
</tr>


</table>

## Implementations
- [Original reference](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb)
- [ncalib](https://github.com/dwoiwode/ncalib) - Modular Neural Cellular Automata library in PyTorch
- [CAX](https://github.com/maxencefaldor/cax) - Cellular Automata in JAX (Flax NNX)
- [JAX-NCA](https://github.com/shyamsn97/jax-nca) - NCA Implementation in JAX (Flax Linen)
- [Hexells](https://github.com/znah/hexells) - SwissGL Implementation of Hexells ([Demo](https://znah.net/hexells/))

## List of Publications
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/papers_per_quarter_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/papers_per_quarter_light.svg">
  <img alt="Histogram of number of publications per quarter." src="">
</picture>

![](assets/papers_per_quarter.png)

### 2025
<table><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2509.11131" target="_blank">
    <img src="assets/thumbnails/2025-09-14neuralcell_hartl.jpg" width="140px">
</a>
</td>
<td>

#### Neural cellular automata: applications to biology and beyond classical AI
Published on **2025-09-14** by

Benedikt **Hartl**, Michael **Levin**, Léo **Pio-Lopez**

[Arxiv](https://arxiv.org/abs/2509.11131)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCA) represent a powerful framework for modeling biological self-organization, extending classical rule-based systems with trainable, differentiable (or evolvable) update rules that capture the adaptive self-regulatory dynamics of living matter. By embedding Artificial Neural Networks (ANNs) as local decision-making centers and interaction rules between localized agents, NCA can simulate processes across molecular, cellular, tissue, and system-level scales, offering a multiscale competency architecture perspective on evolution, development, regeneration, aging, morphogenesis, and robotic control. These models not only reproduce biologically inspired target patterns but also generalize to novel conditions, demonstrating robustness to perturbations and the capacity for open-ended adaptation and reasoning. Given their immense success in recent developments, we here review current literature of NCAs that are relevant primarily for biological or bioengineering applications. Moreover, we emphasize that beyond biology, NCAs display robust and generalizing goal-directed dynamics without centralized control, e.g., in controlling or regenerating composite robotic morphologies or even on cutting-edge reasoning tasks such as ARC-AGI-1. In addition, the same principles of iterative state-refinement is reminiscent to modern generative Artificial Intelligence (AI), such as probabilistic diffusion models. Their governing self-regulatory behavior is constraint to fully localized interactions, yet their collective behavior scales into coordinated system-level outcomes. We thus argue that NCAs constitute a unifying computationally lean paradigm that not only bridges fundamental insights from multiscale biology with modern generative AI, but have the potential to design truly bio-inspired collective intelligence capable of hierarchical reasoning and control.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2509.00651" target="_blank">
    <img src="assets/thumbnails/2025-09-06missingdat_luu.jpg" width="140px">
</a>
</td>
<td>

#### Missing Data Imputation using Neural Cellular Automata
Published on **2025-09-06** by

Tin **Luu**, Binh **Nguyen**, Man **Ngo**

[Arxiv](https://arxiv.org/abs/2509.00651) | [Code](https://github.com/TrungTin98/NICA)

<details>
<summary><b>Abstract</b></summary>
When working with tabular data, missingness is always one of the most painful problems. Throughout many years, researchers have continuously explored better and better ways to impute missing data. Recently, with the rapid development evolution in machine learning and deep learning, there is a new trend of leveraging generative models to solve the imputation task. While the imputing version of famous models such as Variational Autoencoders or Generative Adversarial Networks were investigated, prior work has overlooked Neural Cellular Automata (NCA), a powerful computational model. In this paper, we propose a novel imputation method that is inspired by NCA. We show that, with some appropriate adaptations, an NCA-based model is able to address the missing data imputation problem. We also provide several experiments to evidence that our model outperforms state-of-the-art methods in terms of imputation error and post-imputation performance.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2508.15726" target="_blank">
    <img src="assets/thumbnails/2025-09-06exploringt_pajouheshgar.jpg" width="140px">
</a>
</td>
<td>

#### Exploring the Landscape of Non-Equilibrium Memories with Neural Cellular Automata
Published on **2025-09-06** by

Ehsan **Pajouheshgar**, Aditya **Bhardwaj**, Nathaniel **Selub**, Ethan **Lake**

[Project Page](https://memorynca.github.io/2D/) | [Arxiv](https://arxiv.org/abs/2508.15726)

<details>
<summary><b>Abstract</b></summary>
We investigate the landscape of many-body memories: families of local non-equilibrium dynamics that retain information about their initial conditions for thermodynamically long time scales, even in the presence of arbitrary perturbations. In two dimensions, the only well-studied memory is Toom's rule. Using a combination of rigorous proofs and machine learning methods, we show that the landscape of 2D memories is in fact quite vast. We discover memories that correct errors in ways qualitatively distinct from Toom's rule, have ordered phases stabilized by fluctuations, and preserve information only in the presence of noise. Taken together, our results show that physical systems can perform robust information storage in many distinct ways, and demonstrate that the physics of many-body memories is richer than previously realized. Interactive visualizations of the dynamics studied in this work are available at https://memorynca.github.io/2D.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2508.12324" target="_blank">
    <img src="assets/thumbnails/2025-08-17attentionp_yang.jpg" width="140px">
</a>
</td>
<td>

#### Attention Pooling Enhances NCA-based Classification of Microscopy Images
Published on **2025-08-17** by

Chen **Yang**, Michael **Deutges**, Jingsong **Liu**, Han **Li**, Nassir **Navab**, Carsten **Marr**, Ario **Sadafi**

[Arxiv](https://arxiv.org/abs/2508.12324) | [Code](https://github.com/marrlab/aNCA)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCA) offer a robust and interpretable approach to image classification, making them a promising choice for microscopy image analysis. However, a performance gap remains between NCA and larger, more complex architectures. We address this challenge by integrating attention pooling with NCA to enhance feature extraction and improve classification accuracy. The attention pooling mechanism refines the focus on the most informative regions, leading to more accurate predictions. We evaluate our method on eight diverse microscopy image datasets and demonstrate that our approach significantly outperforms existing NCA methods while remaining parameter-efficient and explainable. Furthermore, we compare our method with traditional lightweight convolutional neural network and vision transformer architectures, showing improved performance while maintaining a significantly lower parameter count. Our results highlight the potential of NCA-based models an alternative for explainable image classification.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2508.12322" target="_blank">
    <img src="assets/thumbnails/2025-08-17neuralcell_deutges.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata for Weakly Supervised Segmentation of White Blood Cells
Published on **2025-08-17** by

Michael **Deutges**, Chen **Yang**, Raheleh **Salehi**, Nassir **Navab**, Carsten **Marr**, Ario **Sadafi**

[Arxiv](https://arxiv.org/abs/2508.12322) | [Code](https://github.com/marrlab/NCA-WSS)

<details>
<summary><b>Abstract</b></summary>
The detection and segmentation of white blood cells in blood smear images is a key step in medical diagnostics, supporting various downstream tasks such as automated blood cell counting, morphological analysis, cell classification, and disease diagnosis and monitoring. Training robust and accurate models requires large amounts of labeled data, which is both time-consuming and expensive to acquire. In this work, we propose a novel approach for weakly supervised segmentation using neural cellular automata (NCA-WSS). By leveraging the feature maps generated by NCA during classification, we can extract segmentation masks without the need for retraining with segmentation labels. We evaluate our method on three white blood cell microscopy datasets and demonstrate that NCA-WSS significantly outperforms existing weakly supervised approaches. Our work illustrates the potential of NCA for both classification and segmentation in a weakly supervised framework, providing a scalable and efficient solution for medical image analysis.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2508.02218" target="_blank">
    <img src="assets/thumbnails/2025-08-13reservoirc_pontes-filho.jpg" width="140px">
</a>
</td>
<td>

#### Reservoir Computing with Evolved Critical Neural Cellular Automata
Published on **2025-08-13** by

Sidney **Pontes-Filho**, Stefano **Nichele**, Mikkel **Lepperød**

[Arxiv](https://arxiv.org/abs/2508.02218) | [Code](https://github.com/bioAI-Oslo/critical-nca-reservoir)

<details>
<summary><b>Abstract</b></summary>
Criticality is a behavioral state in dynamical systems that is known to present the highest computation capabilities, i.e., information transmission, storage, and modification. Therefore, such systems are ideal candidates as a substrate for reservoir computing, a subfield in artificial intelligence. Our choice of a substrate is a cellular automaton (CA) governed by an artificial neural network, also known as neural cellular automaton (NCA). We apply evolution strategy to optimize the NCA to achieve criticality, demonstrated by power law distributions in structures called avalanches. With an evolved critical NCA, the substrate is tested for reservoir computing. Our evaluation of the substrate is performed with two benchmarks, 5-bit memory task and image classification of handwritten digits. The result of the 5-bit memory task achieved a perfect score and the system managed to remember all 5 bits. The result for the image classification task matched and sometimes surpassed the performance of the best elementary CA for this task. Moreover, the proposed critical NCA may operate as a self-organized critical system, due to its robustness to extreme initial conditions.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2508.06389" target="_blank">
    <img src="assets/thumbnails/2025-08-08identityin_stovold.jpg" width="140px">
</a>
</td>
<td>

#### Identity Increases Stability in Neural Cellular Automata
Published on **2025-08-08** by

James **Stovold**

[Arxiv](https://arxiv.org/abs/2508.06389) | [Code](https://github.com/jstovold/ALIFE2025)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCAs) offer a way to study the growth of two-dimensional artificial organisms from a single seed cell. From the outset, NCA-grown organisms have had issues with stability, their natural boundary often breaking down and exhibiting tumour-like growth or failing to maintain the expected shape. In this paper, we present a method for improving the stability of NCA-grown organisms by introducing an 'identity' layer with simple constraints during training. Results show that NCAs grown in close proximity are more stable compared with the original NCA model. Moreover, only a single identity value is required to achieve this increase in stability. We observe emergent movement from the stable organisms, with increasing prevalence for models with multiple identity values. This work lays the foundation for further study of the interaction between NCA-grown organisms, paving the way for studying social interaction at a cellular level in artificial organisms.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2506.22899" target="_blank">
    <img src="assets/thumbnails/2025-06-28neuralcell_pajouheshgar.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata: From Cells to Pixels
Published on **2025-06-28** by

Ehsan **Pajouheshgar**, Yitao **Xu**, Ali **Abbasi**, Alexander **Mordvintsev**, Wenzel **Jakob**, Sabine **Süsstrunk**

[Project Page](https://cells2pixels.github.io/) | [Arxiv](https://arxiv.org/abs/2506.22899) | [Demo Growing](https://cells2pixels.github.io/2d_growing_demo/) | [Demo Texture](https://cells2pixels.github.io/2d_pbr_demo/)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCAs) are bio-inspired systems in which identical cells self-organize to form complex and coherent patterns by repeatedly applying simple local rules. NCAs display striking emergent behaviors including self-regeneration, generalization and robustness to unseen situations, and spontaneous motion. Despite their success in texture synthesis and morphogenesis, NCAs remain largely confined to low-resolution grids. This limitation stems from (1) training time and memory requirements that grow quadratically with grid size, (2) the strictly local propagation of information which impedes long-range cell communication, and (3) the heavy compute demands of real-time inference at high resolution. In this work, we overcome this limitation by pairing NCA with a tiny, shared implicit decoder, inspired by recent advances in implicit neural representations. Following NCA evolution on a coarse grid, a lightweight decoder renders output images at arbitrary resolution. We also propose novel loss functions for both morphogenesis and texture synthesis tasks, specifically tailored for high-resolution output with minimal memory and computation overhead. Combining our proposed architecture and loss functions brings substantial improvement in quality, efficiency, and performance. NCAs equipped with our implicit decoder can generate full-HD outputs in real time while preserving their self-organizing, emergent properties. Moreover, because each MLP processes cell states independently, inference remains highly parallelizable and efficient. We demonstrate the applicability of our approach across multiple NCA variants (on 2D, 3D grids, and 3D meshes) and multiple tasks, including texture generation and morphogenesis (growing patterns from a seed), showing that with our proposed framework, NCAs seamlessly scale to high-resolution outputs with minimal computational overhead.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2506.20486" target="_blank">
    <img src="assets/thumbnails/2025-06-25mixturesof_milite.jpg" width="140px">
</a>
</td>
<td>

#### Mixtures of Neural Cellular Automata: A Stochastic Framework for Growth Modelling and Self-Organization
Published on **2025-06-25** by

Salvatore **Milite**, Giulio **Caravagna**, Andrea **Sottoriva**

[Arxiv](https://arxiv.org/abs/2506.20486)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCAs) are a promising new approach to model self-organizing processes, with potential applications in life science. However, their deterministic nature limits their ability to capture the stochasticity of real-world biological and physical systems. We propose the Mixture of Neural Cellular Automata (MNCA), a novel framework incorporating the idea of mixture models into the NCA paradigm. By combining probabilistic rule assignments with intrinsic noise, MNCAs can model diverse local behaviors and reproduce the stochastic dynamics observed in biological processes. We evaluate the effectiveness of MNCAs in three key domains: (1) synthetic simulations of tissue growth and differentiation, (2) image morphogenesis robustness, and (3) microscopy image segmentation. Results show that MNCAs achieve superior robustness to perturbations, better recapitulate real biological growth patterns, and provide interpretable rule segmentation. These findings position MNCAs as a promising tool for modeling stochastic dynamical systems and studying self-growth processes.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2506.15746" target="_blank">
    <img src="assets/thumbnails/2025-06-18neuralcell_xu.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata for ARC-AGI
Published on **2025-06-18** by

Kevin **Xu**, Risto **Miikkulainen**

[Arxiv](https://arxiv.org/abs/2506.15746)

<details>
<summary><b>Abstract</b></summary>
Cellular automata and their differentiable counterparts, Neural Cellular Automata (NCA), are highly expressive and capable of surprisingly complex behaviors. This paper explores how NCAs perform when applied to tasks requiring precise transformations and few-shot generalization, using the Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) as a domain that challenges their capabilities in ways not previously explored. Specifically, this paper uses gradient-based training to learn iterative update rules that transform input grids into their outputs from the training examples and apply them to the test inputs. Results suggest that gradient-trained NCA models are a promising and efficient approach to a range of abstract grid-based tasks from ARC. Along with discussing the impacts of various design modifications and training constraints, this work examines the behavior and properties of NCAs applied to ARC to give insights for broader applications of self-organizing systems.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2506.04912" target="_blank">
    <img src="assets/thumbnails/2025-06-05differenti_miotti.jpg" width="140px">
</a>
</td>
<td>

#### Differentiable Logic Cellular Automata: From Game of Life to Pattern Generation
Published on **2025-06-05** by

Pietro **Miotti**, Eyvind **Niklasson**, Ettore **Randazzo**, Alexander **Mordvintsev**

[Project Page](https://google-research.github.io/self-organising-systems/difflogic-ca/) | [Arxiv](https://arxiv.org/abs/2506.04912) | [Code](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/diffLogic_CA.ipynb)

<details>
<summary><b>Abstract</b></summary>
This paper introduces Differentiable Logic Cellular Automata (DiffLogic CA), a novel combination of Neural Cellular Automata (NCA) and Differentiable Logic Gates Networks (DLGNs). The fundamental computation units of the model are differentiable logic gates, combined into a circuit. During training, the model is fully end-to-end differentiable allowing gradient-based training, and at inference time it operates in a fully discrete state space. This enables learning local update rules for cellular automata while preserving their inherent discrete nature. We demonstrate the versatility of our approach through a series of milestones: (1) fully learning the rules of Conway's Game of Life, (2) generating checkerboard patterns that exhibit resilience to noise and damage, (3) growing a lizard shape, and (4) multi-color pattern generation. Our model successfully learns recurrent circuits capable of generating desired target patterns. For simpler patterns, we observe success with both synchronous and asynchronous updates, demonstrating significant generalization capabilities and robustness to perturbations. We make the case that this combination of DLGNs and NCA represents a step toward programmable matter and robust computing systems that combine binary logic, neural network adaptability, and localized processing. This work, to the best of our knowledge, is the first successful application of differentiable logic gate networks in recurrent architectures.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://github.com/petroniocandido/st_nca" target="_blank">
    <img src="assets/thumbnail_placeholder.jpg" width="140px">
</a>
</td>
<td>

#### Spatio-Temporal Traffic Forecasting with Neural Graph Cellular Automata
Published on **2025-05-29** by

Petrônio C. L. **Silva**, Omid **Orang**, Lucas **Astore**, Frederico G. **Guimarães ORCID iD icon**

[Code](https://github.com/petroniocandido/st_nca)

<details>
<summary><b>Abstract</b></summary>
Transformer-based neural network cells with distributed federated learning for flexible cellular automata topologies, aiming for large-scale forecasting of complex spatiotemporal processes.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2505.13058" target="_blank">
    <img src="assets/thumbnails/2025-05-20apathtouni_béna.jpg" width="140px">
</a>
</td>
<td>

#### A Path to Universal Neural Cellular Automata
Published on **2025-05-20** by

Gabriel **Béna**, Maxence **Faldor**, Dan F. M. **Goodman**, Antoine **Cully**

[Arxiv](https://arxiv.org/abs/2505.13058) | [Project Page](https://gabrielbena.github.io/blog/2025/bena2025unca/)

<details>
<summary><b>Abstract</b></summary>
Cellular automata have long been celebrated for their ability to generate complex behaviors from simple, local rules, with well-known discrete models like Conway's Game of Life proven capable of universal computation. Recent advancements have extended cellular automata into continuous domains, raising the question of whether these systems retain the capacity for universal computation. In parallel, neural cellular automata have emerged as a powerful paradigm where rules are learned via gradient descent rather than manually designed. This work explores the potential of neural cellular automata to develop a continuous Universal Cellular Automaton through training by gradient descent. We introduce a cellular automaton model, objective functions and training strategies to guide neural cellular automata toward universal computation in a continuous setting. Our experiments demonstrate the successful training of fundamental computational primitives - such as matrix multiplication and transposition - culminating in the emulation of a neural network solving the MNIST digit classification task directly within the cellular automata state. These results represent a foundational step toward realizing analog general-purpose computers, with implications for understanding universal computation in continuous dynamics and advancing the automated discovery of complex cellular automata behaviors via machine learning.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://www.nature.com/articles/s41598-025-94032-y.pdf" target="_blank">
    <img src="assets/thumbnails/2025-05-20intelligen_sharma.jpg" width="140px">
</a>
</td>
<td>

#### Intelligent health model for medical imaging to guide laymen using neural cellular automata
Published on **2025-05-20** by

Sandeep Kumar **Sharma**, Chiranji Lal **Chowdhary**, Vijay Shankar **Sharma**, Adil **Rasool**, Arfat Ahmad **Khan**

[Paper](https://www.nature.com/articles/s41598-025-94032-y.pdf) | [Code](https://github.com/Arfat673/Intelligent-health-model-for-medical-imaging-to-guide-laymen-using-neural-cellular-automata)

<details>
<summary><b>Abstract</b></summary>
A layman in health systems is a person who doesn’t have any knowledge about health data i.e., X-ray, MRI, CT scan, and health examination reports, etc. The motivation behind the proposed invention is to help laymen to make medical images understandable. The health model is trained using a neural network approach that analyses user health examination data; predicts the type and level of the disease and advises precaution to the user. Cellular Automata (CA) technology has been integrated with the neural networks to segment the medical image. The CA analyzes the medical images pixel by pixel and generates a robust threshold value which helps to efficiently segment the image and identify accurate abnormal spots from the medical image. The proposed method has been trained and experimented using 10000+ medical images which are taken from various open datasets. Various text analysis measures i.e., BLEU, ROUGE, and WER are used in the research to validate the produced report. The BLEU and ROUGE calculate a similarity to decide how the generated text report is closer to the original report. The BLEU and ROUGE scores of the experimented images are approximately 0.62 and 0.90, claims that the produced report is very close to the original report. The WER score 0.14, claims that the generated report contains the most relevant words. The overall summary of the proposed research is that it provides a fruitful medical report with accurate disease and precautions to the laymen.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2505.08778" target="_blank">
    <img src="assets/thumbnails/2025-05-13-arcnca_guichard.jpg" width="140px">
</a>
</td>
<td>

#### ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus
Published on **2025-05-13** by

Ettienne **Guichard**, Felix **Reimers**, Mia-Katrin **Kvalsund**, Mikkel Elle **Lepperød**, Stefano **Nichele**

[Arxiv](https://arxiv.org/abs/2505.08778) | [Project Page](https://etimush.github.io/ARC_NCA/) | [Code](https://github.com/etimush/ARC_NCA/) | [Video](https://etimush.github.io/ARC-NCA-Videos/)

<details>
<summary><b>Abstract</b></summary>
The Abstraction and Reasoning Corpus (ARC), later renamed ARC-AGI, poses a fundamental challenge in artificial general intelligence (AGI), requiring solutions that exhibit robust abstraction and reasoning capabilities across diverse tasks, while only few (with median count of three) correct examples are presented. While ARC-AGI remains very challenging for artificial intelligence systems, it is rather easy for humans. This paper introduces ARC-NCA, a developmental approach leveraging standard Neural Cellular Automata (NCA) and NCA enhanced with hidden memories (EngramNCA) to tackle the ARC-AGI benchmark. NCAs are employed for their inherent ability to simulate complex dynamics and emergent patterns, mimicking developmental processes observed in biological systems. Developmental solutions may offer a promising avenue for enhancing AI's problem-solving capabilities beyond mere training data extrapolation. ARC-NCA demonstrates how integrating developmental principles into computational models can foster adaptive reasoning and abstraction. We show that our ARC-NCA proof-of-concept results may be comparable to, and sometimes surpass, that of ChatGPT 4.5, at a fraction of the cost.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2504.21562" target="_blank">
    <img src="assets/thumbnails/2025-04-30encapsulat_krumb.jpg" width="140px">
</a>
</td>
<td>

#### eNCApsulate: NCA for Precision Diagnosis on Capsule Endoscopes
Published on **2025-04-30** by

Henry John **Krumb**, Anirban **Mukhopadhyay**

[Arxiv](https://arxiv.org/abs/2504.21562) | [Code](https://github.com/MECLabTUDA/eNCApsulate)

<details>
<summary><b>Abstract</b></summary>
Purpose: Wireless Capsule Endoscopy (WCE) is a non-invasive imaging method for the entire gastrointestinal tract, and is a pain-free alternative to traditional endoscopy. It generates extensive video data that requires significant review time, and localizing the capsule after ingestion is a challenge. Techniques like bleeding detection and depth estimation can help with localization of pathologies, but deep learning models are typically too large to run directly on the capsule.
Methods: Neural Cellular Automata (NCAs) for bleeding segmentation and depth estimation are trained on capsule endoscopic images. For monocular depth estimation, we distill a large foundation model into the lean NCA architecture, by treating the outputs of the foundation model as pseudo ground truth. We then port the trained NCAs to the ESP32 microcontroller, enabling efficient image processing on hardware as small as a camera capsule.
Results: NCAs are more accurate (Dice) than other portable segmentation models, while requiring more than 100x fewer parameters stored in memory than other small-scale models. The visual results of NCAs depth estimation look convincing, and in some cases beat the realism and detail of the pseudo ground truth. Runtime optimizations on the ESP32-S3 accelerate the average inference speed significantly, by more than factor 3.
Conclusion: With several algorithmic adjustments and distillation, it is possible to eNCApsulate NCA models into microcontrollers that fit into wireless capsule endoscopes. This is the first work that enables reliable bleeding segmentation and depth estimation on a miniaturized device, paving the way for precise diagnosis combined with visual odometry as a means of precise localization of the capsule – on the capsule.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2504.11855" target="_blank">
    <img src="assets/thumbnails/2025-04-16engramnca_guichard.jpg" width="140px">
</a>
</td>
<td>

#### EngramNCA: a Neural Cellular Automaton Model of Memory Transfer
Published on **2025-04-16** by

Etienne **Guichard**, Felix **Reimers**, Mia **Kvalsund**, Mikkel **Lepperød**, Stefano **Nichele**

[Arxiv](https://arxiv.org/abs/2504.11855) | [Project Page](https://etimush.github.io/EngramNCA/) | [Code](https://github.com/etimush/EngramNCA)

<details>
<summary><b>Abstract</b></summary>
This study introduces EngramNCA, a neural cellular automaton (NCA) that integrates both publicly visible states and private, cell-internal memory channels, drawing inspiration from emerging biological evidence suggesting that memory storage extends beyond synaptic modifications to include intracellular mechanisms. The proposed model comprises two components: GeneCA, an NCA trained to develop distinct morphologies from seed cells containing immutable ”gene” encodings, and GenePropCA, an auxiliary NCA that modulates the private ”genetic” memory of cells without altering their visible states. This architecture enables the encoding and propagation of complex morphologies through the interaction of visible and private channels, facilitating the growth of diverse structures from a shared ”genetic” substrate. EngramNCA supports the emergence of hierarchical and coexisting morphologies, offering insights into decentralized memory storage and transfer in artificial systems. These findings have potential implications for the development of adaptive, self-organizing systems and may contribute to the broader understanding of memory mechanisms in both biological and synthetic contexts.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://www.biorxiv.org/content/10.1101/2024.12.06.627209v2" target="_blank">
    <img src="assets/thumbnails/2025-03-28sensormove_kvalsund.jpg" width="140px">
</a>
</td>
<td>

#### Sensor Movement Drives Emergent Attention and Scalability in Active Neural Cellular Automata
Published on **2025-03-28** by

Mia-Katrin **Kvalsund**, Kai Olav **Ellefsen**, Kyrre **Glette**, Sidney **Pontes-Filho**, Mikkel Elle **Lepperød**

[Arxiv](https://www.biorxiv.org/content/10.1101/2024.12.06.627209v2) | [Video](https://www.youtube.com/watch?v=-ERmvXo0XTs) | [Code](https://github.com/bioAI-Oslo/column)

<details>
<summary><b>Abstract</b></summary>
The brain’s distributed architecture has inspired numerous artificial intelligence (AI) systems, particularly through its neocortical organization. However, current AI approaches largely overlook a crucial aspect of biological intelligence: active sensing - the deliberate movement of sensory organs to explore the environment. To explore how sensor movement impacts behavior in image classification tasks, we introduce the Active Neural Cellular Automata (ANCA), a neocortex-inspired model with movable sensors. Active sensing naturally emerges in the ANCA, with belief-informed exploration and attentive behavior to salient information, without adding explicit attention mechanisms. Active sensing both simplifies classification tasks and leads to a highly scalable system. This enables ANCAs to be smaller than the image size without losing information and enables fault tolerance to damaged sensors. Overall, our work provides insight to how distributed architectures can interact with movement, opening new avenues for adaptive AI systems in embodied agents.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2410.02651" target="_blank">
    <img src="assets/thumbnails/2025-03-11caxcellul_faldor.jpg" width="140px">
</a>
</td>
<td>

#### CAX: Cellular Automata Accelerated in JAX
Published on **2025-03-11** by

Maxence **Faldor**, Antoine **Cully**

[Arxiv](https://arxiv.org/abs/2410.02651) | [Code](https://github.com/maxencefaldor/cax)

<details>
<summary><b>Abstract</b></summary>
Cellular automata have become a cornerstone for investigating emergence and self-organization across diverse scientific disciplines. However, the absence of a hardware-accelerated cellular automata library limits the exploration of new research directions, hinders collaboration, and impedes reproducibility. In this work, we introduce CAX (Cellular Automata Accelerated in JAX), a high-performance and flexible open-source library designed to accelerate cellular automata research. CAX delivers cutting-edge performance through hardware acceleration while maintaining flexibility through its modular architecture, intuitive API, and support for both discrete and continuous cellular automata in arbitrary dimensions. We demonstrate CAX’s performance and flexibility through a wide range of benchmarks and applications. From classic models like elementary cellular automata and Conway’s Game of Life to advanced applications such as growing neural cellular automata and self-classifying MNIST digits, CAX speeds up simulations up to 2,000 times faster. Furthermore, we demonstrate CAX’s potential to accelerate research by presenting a collection of three novel cellular automata experiments, each implemented in just a few lines of code thanks to the library’s modular architecture. Notably, we show that a simple one-dimensional cellular automaton can outperform GPT-4 on the 1D-ARC challenge.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2502.18738" target="_blank">
    <img src="assets/thumbnails/2025-02-26pytorchfir_xia.jpg" width="140px">
</a>
</td>
<td>

#### PyTorchFire: A GPU-Accelerated Wildfire Simulator with Differentiable Cellular Automata
Published on **2025-02-26** by

Zeyu **Xia**, Sibo **Cheng**

[Arxiv](https://arxiv.org/abs/2502.18738) | [Paper](https://www.sciencedirect.com/science/article/pii/S1364815225000854) | [Code](https://github.com/xiazeyu/PyTorchFire) | [Project Page](https://pytorchfire.readthedocs.io/)

<details>
<summary><b>Abstract</b></summary>
Accurate and rapid prediction of wildfire trends is crucial for effective management and mitigation. However, the stochastic nature of fire propagation poses significant challenges in developing reliable simulators. In this paper, we introduce PyTorchFire, an open-access, PyTorch-based software that leverages GPU acceleration. With our redesigned differentiable wildfire Cellular Automata (CA) model, we achieve millisecond-level computational efficiency, significantly outperforming traditional CPU-based wildfire simulators on real-world-scale fires at high resolution. Real-time parameter calibration is made possible through gradient descent on our model, aligning simulations closely with observed wildfire behavior both temporally and spatially, thereby enhancing the realism of the simulations. Our PyTorchFire simulator, combined with real-world environmental data, demonstrates superior generalizability compared to supervised learning surrogate models. Its ability to predict and calibrate wildfire behavior in real-time ensures accuracy, stability, and efficiency. PyTorchFire has the potential to revolutionize wildfire simulation, serving as a powerful tool for wildfire prediction and management.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2502.01242" target="_blank">
    <img src="assets/thumbnails/2025-02-03neuralcell_dacre.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata for Decentralized Sensing using a Soft Inductive Sensor Array for Distributed Manipulator Systems
Published on **2025-02-03** by

Bailey **Dacre**, Nicolas **Bessone**, Matteo Lo **Preti**, Diana **Cafiso**, Rodrigo **Moreno**, Andrés **Faíña**, Lucia **Beccai**

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-70415-4_6) | [Arxiv](https://arxiv.org/abs/2502.01242)

<details>
<summary><b>Abstract</b></summary>
In Distributed Manipulator Systems (DMS), decentralization is a highly desirable property as it promotes robustness and facilitates scalability by distributing computational burden and eliminating singular points of failure. However, current DMS typically utilize a centralized approach to sensing, such as single-camera computer vision systems. This centralization poses a risk to system reliability and offers a significant limiting factor to system size. In this work, we introduce a decentralized approach for sensing and in a Distributed Manipulator Systems using Neural Cellular Automata (NCA). Demonstrating a decentralized sensing in a hardware implementation, we present a novel inductive sensor board designed for distributed sensing and evaluate its ability to estimate global object properties, such as the geometric center, through local interactions and computations. Experiments demonstrate that NCA-based sensing networks accurately estimate object position at 0.24 times the inter sensor distance. They maintain resilience under sensor faults and noise, and scale seamlessly across varying network sizes. These findings underscore the potential of local, decentralized computations to enable scalable, fault-tolerant, and noise-resilient object property estimation in DMS
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2501.03573" target="_blank">
    <img src="assets/thumbnails/2025-01-07neuralcell_jia.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata and Deep Equilibrium Models
Published on **2025-01-07** by

Zhibai **Jia**

[Arxiv](https://arxiv.org/abs/2501.03573)

<details>
<summary><b>Abstract</b></summary>
This essay discusses the connections and differences between two emerging paradigms in deep learning, namely Neural Cellular Automata and Deep Equilibrium Models, and train a simple Deep Equilibrium Convolutional model to demonstrate the inherent similarity of NCA and DEQ based methods. Finally, this essay speculates about ways to combine theoretical and practical aspects of both approaches for future research.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2501.02447" target="_blank">
    <img src="assets/thumbnails/2025-01-05medsegdiff_mittal.jpg" width="140px">
</a>
</td>
<td>

#### MedSegDiffNCA: Diffusion Models With Neural Cellular Automata for Skin Lesion Segmentation
Published on **2025-01-05** by

Avni **Mittal**, John **Kalkhof**, Anirban **Mukhopadhyay**, Arnav **Bhavsar**

[Arxiv](https://arxiv.org/abs/2501.02447)

<details>
<summary><b>Abstract</b></summary>
Denoising Diffusion Models (DDMs) are widely used for high-quality image generation and medical image segmentation but often rely on Unet-based architectures, leading to high computational overhead, especially with high-resolution images. This work proposes three NCA-based improvements for diffusion-based medical image segmentation. First, Multi-MedSegDiffNCA uses a multilevel NCA framework to refine rough noise estimates generated by lower level NCA models. Second, CBAM-MedSegDiffNCA incorporates channel and spatial attention for improved segmentation. Third, MultiCBAM-MedSegDiffNCA combines these methods with a new RGB channel loss for semantic guidance. Evaluations on Lesion segmentation show that MultiCBAM-MedSegDiffNCA matches Unet-based model performance with dice score of 87.84% while using 60-110 times fewer parameters, offering a more efficient solution for low resource medical settings.
</details>

</td>
</tr>
</table>


### 2024
<table><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2406.08298" target="_blank">
    <img src="assets/thumbnails/2024-11-21adancaneu_xu.jpg" width="140px">
</a>
</td>
<td>

#### AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer
Published on **2024-11-21** by

Yitao **Xu**, Tong **Zhang**, Sabine **Süsstrunk**

[Arxiv](https://arxiv.org/abs/2406.08298) | [Paper](https://dl.acm.org/doi/10.5555/3737916.3738725)

<details>
<summary><b>Abstract</b></summary>
Vision Transformers (ViTs) demonstrate remarkable performance in image classification through visual-token interaction learning, particularly when equipped with local information via region attention or convolutions. Although such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global visual-token representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy input. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformers that uses NCA as plug-and-play adaptors between ViT layers, thus enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Using our analysis of AdaNCA placement and robustness improvement, we also develop an algorithm for identifying the most effective insertion points for AdaNCA. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across eight robustness benchmarks and four ViT architectures that AdaNCA, as a plug-and-play module, consistently improves the robustness of ViTs.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2410.23368" target="_blank">
    <img src="assets/thumbnails/2024-10-30ncadaptdy_ranem.jpg" width="140px">
</a>
</td>
<td>

#### NCAdapt: Dynamic adaptation with domain-specific Neural Cellular Automata for continual hippocampus segmentation
Published on **2024-10-30** by

Amin **Ranem**, John **Kalkhof**, Anirban **Mukhopadhyay**

[Arxiv](https://arxiv.org/abs/2410.23368) | [Code](https://github.com/MECLabTUDA/NCAdapt)

<details>
<summary><b>Abstract</b></summary>
Continual learning (CL) in medical imaging presents a unique challenge, requiring models to adapt to new domains while retaining previously acquired knowledge. We introduce NCAdapt, a Neural Cellular Automata (NCA) based method designed to address this challenge. NCAdapt features a domain-specific multi-head structure, integrating adaptable convolutional layers into the NCA backbone for each new domain encountered. After initial training, the NCA backbone is frozen, and only the newly added adaptable convolutional layers, consisting of 384 parameters, are trained along with domain-specific NCA convolutions. We evaluate NCAdapt on hippocampus segmentation tasks, benchmarking its performance against Lifelong nnU-Net and U-Net models with state-of-the-art (SOTA) CL methods. Our lightweight approach achieves SOTA performance, underscoring its effectiveness in addressing CL challenges in medical imaging. Upon acceptance, we will make our code base publicly accessible to support reproducibility and foster further advancements in medical CL.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2410.22265" target="_blank">
    <img src="assets/thumbnails/2024-10-29ncamorph_ranem.jpg" width="140px">
</a>
</td>
<td>

#### NCA-Morph: Medical Image Registration with Neural Cellular Automata
Published on **2024-10-29** by

Amin **Ranem**, John **Kalkhof**, Anirban **Mukhopadhyay**

[Arxiv](https://arxiv.org/abs/2410.22265) | [Code](https://github.com/MECLabTUDA/NCA-Morph)

<details>
<summary><b>Abstract</b></summary>
Medical image registration is a critical process that aligns various patient scans, facilitating tasks like diagnosis, surgical planning, and tracking. Traditional optimization based methods are slow, prompting the use of Deep Learning (DL) techniques, such as VoxelMorph and Transformer-based strategies, for faster results. However, these DL methods often impose significant resource demands. In response to these challenges, we present NCA-Morph, an innovative approach that seamlessly blends DL with a bio-inspired communication and networking approach, enabled by Neural Cellular Automata (NCAs). NCA-Morph not only harnesses the power of DL for efficient image registration but also builds a network of local communications between cells and respective voxels over time, mimicking the interaction observed in living systems. In our extensive experiments, we subject NCA-Morph to evaluations across three distinct 3D registration tasks, encompassing Brain, Prostate and Hippocampus images from both healthy and diseased patients. The results showcase NCA-Morph's ability to achieve state-of-the-art performance. Notably, NCA-Morph distinguishes itself as a lightweight architecture with significantly fewer parameters; 60% and 99.7% less than VoxelMorph and TransMorph. This characteristic positions NCA-Morph as an ideal solution for resource-constrained medical applications, such as primary care settings and operating rooms.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/1809.01687" target="_blank">
    <img src="assets/thumbnails/2024-10-01breasttumo_ali.jpg" width="140px">
</a>
</td>
<td>

#### Breast tumor segmentation using neural cellular automata and shape guided segmentation in mammography images
Published on **2024-10-01** by

Mudassar **Ali**, Tong **Wu**, Haoji **Hu**, Tariq **Mahmood**

[Arxiv](https://arxiv.org/abs/1809.01687) | [Paper](https://pubmed.ncbi.nlm.nih.gov/39352900/)

<details>
<summary><b>Abstract</b></summary>
Purpose Using computer-aided design (CAD) systems, this research endeavors to enhance breast cancer segmentation by addressing data insufficiency and data complexity during model training. As perceived by computer vision models, the inherent symmetry and complexity of mammography images make segmentation difficult. The objective is to optimize the precision and effectiveness of medical imaging.
Methods The study introduces a hybrid strategy combining shape-guided segmentation (SGS) and M3D-neural cellular automata (M3D-NCA), resulting in improved computational efficiency and performance. The implementation of Shape-guided segmentation (SGS) during the initialization phase, coupled with the elimination of convolutional layers, enables the model to effectively reduce computation time. The research proposes a novel loss function that combines segmentation losses from both components for effective training.
Results The robust technique provided aims to improve the accuracy and consistency of breast tumor segmentation, leading to significant improvements in medical imaging and breast cancer detection and treatment.
Conclusion This study enhances breast cancer segmentation in medical imaging using CAD systems. Combining shape-guided segmentation (SGS) and M3D-neural cellular automata (M3DNCA) is a hybrid approach that improves performance and computational efficiency by dealing with complex data and not having enough training data. The approach also reduces PLOS ONE
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://www.sciencedirect.com/science/article/abs/pii/S1746809424006050" target="_blank">
    <img src="assets/thumbnails/2024-10-01skinlesion_yue.jpg" width="140px">
</a>
</td>
<td>

#### Skin lesion segmentation via Neural Cellular Automata
Published on **2024-10-01** by

Tao **Yue**, Cangtao **Chen**, Yue **Wang**, Wenhua **Zhang**, Na **Liu**, Songyi **Zhong**, Long **Li**, Quan **Zhang**

[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1746809424006050)

<details>
<summary><b>Abstract</b></summary>
Skin melanoma is one of the most dangerous tumor lesions. In recent years, the number of cases and deaths caused by melanoma has been increasing. The discovery and segmentation of the lesion area are crucial for the timely diagnosis and treatment of melanoma. However, the lesion area is often similar to the healthy area, and the size scale changes greatly, which makes the segmentation of the skin lesion area a very challenging task. Neural Cellular Automata (NCA) is a model that can be described as a recurrent convolutional network. It achieves global consistency through multiple local information exchanges, thereby completing the processing of information. Recent research on NCA shows that such a local interactive model can segment low-resolution images well. However, for high-resolution images, direct use of NCA for processing is limited by high memory requirements and difficulty in model convergence. In this paper, in order to overcome these limitations, we propose a new NCA-based segmentation model, named UNCA. UNCA is a model with a U-shaped structure. The high-resolution image is down-sampled to obtain high-dimensional low-resolution features, which are then input into the NCA for information processing. Finally, the image size is restored by upsampling. The experimental results show that the UNCA proposed in this paper has achieved good results on the ISIC2017 dataset, surpassing most of the current methods.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2409.06888" target="_blank">
    <img src="assets/thumbnails/2024-09-10aqualitydi_qian.jpg" width="140px">
</a>
</td>
<td>

#### A Quality Diversity Approach to Automatically Generate Multi-Agent Path Finding Benchmark Maps
Published on **2024-09-10** by

Cheng **Qian**, Yulun **Zhang**, Varun **Bhatt**, Matthew Christopher **Fontaine**, Stefanos **Nikolaidis**, Jiaoyang **Li**

[Paper](https://ojs.aaai.org/index.php/SOCS/article/view/31580) | [Arxiv](https://arxiv.org/abs/2409.06888)

<details>
<summary><b>Abstract</b></summary>
We use the Quality Diversity (QD) algorithm with Neural Cellular Automata (NCA) to generate benchmark maps for Multi-Agent Path Finding (MAPF) algorithms. Previously, MAPF algorithms are tested using fixed, human-designed benchmark maps. However, such fixed benchmark maps have several problems. First, these maps may not cover all the potential failure scenarios for the algorithms. Second, when comparing different algorithms, fixed benchmark maps may introduce bias leading to unfair comparisons between algorithms. In this work, we take advantage of the QD algorithm and NCA with different objectives and diversity measures to generate maps with patterns to comprehensively understand the performance of MAPF algorithms and be able to make fair comparisons between two MAPF algorithms to provide further information on the selection between two algorithms. Empirically, we employ this technique to generate diverse benchmark maps to evaluate and compare the behavior of different types of MAPF algorithms such as bounded-suboptimal algorithms, suboptimal algorithms, and reinforcement-learning-based algorithms. Through both single-planner experiments and comparisons between algorithms, we identify patterns where each algorithm excels and detect disparities in runtime or success rates between different algorithms.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2407.18114" target="_blank">
    <img src="assets/thumbnails/2024-07-25unsupervis_kalkhof.jpg" width="140px">
</a>
</td>
<td>

#### Unsupervised Training of Neural Cellular Automata on Edge Devices
Published on **2024-07-25** by

John **Kalkhof**, Amin **Ranem**, Anirban **Mukhopadhyay**

[Arxiv](https://arxiv.org/abs/2407.18114) | [Code](https://github.com/MECLabTUDA/M3D-NCA)

<details>
<summary><b>Abstract</b></summary>
The disparity in access to machine learning tools for medical imaging across different regions significantly limits the potential for universal healthcare innovation, particularly in remote areas. Our research addresses this issue by implementing Neural Cellular Automata (NCA) training directly on smartphones for accessible X-ray lung segmentation. We confirm the practicality and feasibility of deploying and training these advanced models on five Android devices, improving medical diagnostics accessibility and bridging the tech divide to extend machine learning benefits in medical imaging to low- and middle-income countries (LMICs). We further enhance this approach with an unsupervised adaptation method using the novel Variance-Weighted Segmentation Loss (VWSL), which efficiently learns from unlabeled data by minimizing the variance from multiple NCA predictions. This strategy notably improves model adaptability and performance across diverse medical imaging contexts without the need for extensive computational resources or labeled datasets, effectively lowering the participation threshold. Our methodology, tested on three multisite X-ray datasets—Padchest, ChestX-ray8, and MIMIC-III—demonstrates improvements in segmentation Dice accuracy by 0.7 to 2.8%, compared to the classic Med-NCA. Additionally, in extreme cases where no digital copy is available and images must be captured by a phone from an X-ray lightbox or monitor, VWSL enhances Dice accuracy by 5-20%, demonstrating the method’s robustness even with suboptimal image sources.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://direct.mit.edu/isal/proceedings/isal2024/36/99/123527" target="_blank">
    <img src="assets/thumbnails/2024-07-22generegul_hintze.jpg" width="140px">
</a>
</td>
<td>

#### Gene-Regulated Neural Cellular Automata
Published on **2024-07-22** by

Arend **Hintze**, Mustafa **Al-Hammadi**, Eric **Libby**

[Paper](https://direct.mit.edu/isal/proceedings/isal2024/36/99/123527)

<details>
<summary><b>Abstract</b></summary>
Abstract. This paper introduces a Gene Regulatory Neural Cellular Automata (ENIGMA), an innovative extension of the Neural Cellular Automata (NCA) framework aimed at modeling biological development with a greater degree of biological fidelity. Traditional NCAs, while capable of generating complex patterns through neural network-driven update rules, lack mechanisms that closely mimic biological processes such as cell-cell signaling and gene regulatory networks (GRNs). Our ENIGMA model addresses these limitations by incorporating update rules based on a simulated gene regulatory network driven by cell-cell signaling, optimized both through backpropagation and genetic algorithms. We demonstrate the structure and functionality of ENIGMA through various experiments, comparing its performance and properties with those of natural organisms. Our findings reveal that ENIGMA can successfully simulate complex cellular networks and exhibit phenomena such as homeotic transformations, pattern maintenance in variable tissue sizes, and the formation of simple regulatory motifs akin to those observed in developmental biology. The introduction of ENIGMA represents a significant step towards bridging the gap between computational models and the intricacies of biological development, offering a versatile tool for exploring developmental and evolutionary questions with profound implications for understanding gene regulation, pattern formation, and the emergent behavior of complex systems.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2407.05991" target="_blank">
    <img src="assets/thumbnails/2024-07-19multitext_catrina.jpg" width="140px">
</a>
</td>
<td>

#### Multi-Texture Synthesis through Signal Responsive Neural Cellular Automata
Published on **2024-07-19** by

Mirela-Magdalena **Catrina**, Ioana Cristina **Plajer**, Alexandra **Baicoianu**

[Arxiv](https://arxiv.org/abs/2407.05991)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCA) have proven to be effective in a variety of fields, with numerous biologically inspired applications. One of the fields, in which NCAs perform well is the generation of textures, modelling global patterns from local interactions governed by uniform and coherent rules. This paper aims to enhance the usability of NCAs in texture synthesis by addressing a shortcoming of current NCA architectures for texture generation, which requires separately trained NCA for each individual texture. In this work, we train a single NCA for the evolution of multiple textures, based on individual examples. Our solution provides texture information in the state of each cell, in the form of an internally coded genomic signal, which enables the NCA to generate the expected texture. Such a neural cellular automaton not only maintains its regenerative capability but also allows for interpolation between learned textures and supports grafting techniques. This demonstrates the ability to edit generated textures and the potential for them to merge and coexist within the same automaton. We also address questions related to the influence of the genomic information and the cost function on the evolution of the NCA.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://dl.acm.org/doi/10.1145/3638529.3654150" target="_blank">
    <img src="assets/thumbnails/2024-07-14evolvinghi_bielawski.jpg" width="140px">
</a>
</td>
<td>

#### Evolving Hierarchical Neural Cellular Automata
Published on **2024-07-14** by

Kameron **Bielawski**, Nate **Gaylinn**, Cameron **Lunn**, Kevin **Motia**, Joshua **Bongard**

[Paper](https://dl.acm.org/doi/10.1145/3638529.3654150) | [Code](https://github.com/ngaylinn/mocs-final)

<details>
<summary><b>Abstract</b></summary>
Much is unknown about how living systems grow into, coordinate communication across, and maintain themselves as hierarchical arrangements of semi-independent cells, tissues, organs, and entire bodies, where each component at each level has its own goals and sensor, motor, and communication capabilities. Similar uncertainty surrounds exactly how selection acts on the components across these levels. Finally, growing interest in viewing intelligence not as something localized to the brain but rather distributed across biological hierarchies has renewed investigation into the nature of such hierarchies. Here we show that organizing neural cellular automata (NCAs) into a hierarchical structure can improve the ability to evolve them to perform morphogenesis and homeostasis, compared to non-hierarchical NCAs. The increased evolvability of hierarchical NCAs (HNCAs) compared to non-hierarchical NCAs suggests an evolutionary advantage to the formation and utilization of higher-order structures, across larger spatial scales, for some tasks, and suggests new ways to design and optimize NCA models and hierarchical arrangements of robots. The results presented here demonstrate the value of explicitly incorporating hierarchical structure into systems that must grow and maintain complex patterns. The introduced method may also serve as a platform to further investigate the evolutionary dynamics of multiscale systems.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2407.03018" target="_blank">
    <img src="assets/thumbnails/2024-07-03anorganism_elbatel.jpg" width="140px">
</a>
</td>
<td>

#### An Organism Starts with a Single Pix-Cell: A Neural Cellular Diffusion for High-Resolution Image Synthesis
Published on **2024-07-03** by

Marawan **Elbatel**, Konstantinos **Kamnitsas**, Xiaomeng **Li**

[Arxiv](https://arxiv.org/abs/2407.03018) | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-72378-0_61) | [Code](https://github.com/xmed-lab/GeCA)

<details>
<summary><b>Abstract</b></summary>
Generative modeling seeks to approximate the statistical properties of real data, enabling synthesis of new data that closely resembles the original distribution. Generative Adversarial Networks (GANs) and Denoising Diffusion Probabilistic Models (DDPMs) represent significant advancements in generative modeling, drawing inspiration from game theory and thermodynamics, respectively. Nevertheless, the exploration of generative modeling through the lens of biological evolution remains largely untapped. In this paper, we introduce a novel family of models termed Generative Cellular Automata (GeCA), inspired by the evolution of an organism from a single cell. GeCAs are evaluated as an effective augmentation tool for retinal disease classification across two imaging modalities: Fundus and Optical Coherence Tomography (OCT). In the context of OCT imaging, where data is scarce and the distribution of classes is inherently skewed, GeCA significantly boosts the performance of 11 different ophthalmological conditions, achieving a 12% increase in the average F1 score compared to conventional baselines. GeCAs outperform both diffusion methods that incorporate UNet or state-of-the art variants with transformer-based denoising models, under similar parameter constraints. Code is available at: https://github. com/xmed-lab/GeCA.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://ieeexplore.ieee.org/document/10650578" target="_blank">
    <img src="assets/thumbnails/2024-06-30physicsin_navarin.jpg" width="140px">
</a>
</td>
<td>

#### Physics-Informed Graph Neural Cellular Automata: an Application to Compartmental Modelling
Published on **2024-06-30** by

Nicolò **Navarin**, Paolo **Frazzetto**, Luca **Pasa**, Pietro **Verzelli**, Filippo **Visentin**, Alessandro **Sperduti**, Cesare **Alippi**

[Paper](https://ieeexplore.ieee.org/document/10650578)

<details>
<summary><b>Abstract</b></summary>
The recent outbreak of COVID-19 has spurred global collaborative research efforts to model and forecast the disease to improve preparation and control. Epidemiological models integrate experimental data and expert opinions to understand infection dynamics and control measures. Classical Machine Learning techniques often face challenges such as high data requirements, lack of interpretability, and difficulty integrating domain knowledge. A potential solution is to leverage Physically-Informed Machine Learning (PIML) models, which enhance models by incorporating known physical properties of viral spread. Additionally, epidemiological datasets are best represented as graphs, facilitating the modelling of interactions between individuals. In this paper, we propose a novel, interpretable graph-based PIML technique called SINDy-Graph to model infectious disease dynamics. Our approach is a Graph Cellular Automata architecture that combines the ability to identify dynamics for discovering the differential equations governing the physical phenomena under study using graphs modelling relationships between nodes (individuals). The experimental results demonstrate that integrating domain knowledge ensures better physical plausibility. In addition, our proposed model is easier to train and achieves a lower generalisation error compared to other baseline methods.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2404.06406" target="_blank">
    <img src="assets/thumbnails/2024-06-20emergentdy_xu.jpg" width="140px">
</a>
</td>
<td>

#### Emergent Dynamics in Neural Cellular Automata
Published on **2024-06-20** by

Yitao **Xu**, Ehsan **Pajouheshgar**, Sabine **Süsstrunk**

[Arxiv](https://arxiv.org/abs/2404.06406) | [Paper](https://direct.mit.edu/isal/proceedings/isal2024/36/96/123544)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCA) models are trainable variations of traditional Cellular Automata (CA). Emergent motion in the patterns created by NCA has been successfully applied to synthesize dynamic textures. However, the conditions required for an NCA to display dynamic patterns remain unexplored. Here, we investigate the relationship between the NCA architecture and the emergent dynamics of the trained models. Specifically, we vary the number of channels in the cell state and the number of hidden neurons in the MultiLayer Perceptron (MLP), and draw a relationship between the combination of these two variables and the motion strength between successive frames. Our analysis reveals that the disparity and proportionality between these two variables have a strong correlation with the emergent dynamics in the NCA output. We thus propose a design principle for creating dynamic NCA.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2406.09654" target="_blank">
    <img src="assets/thumbnails/2024-06-14coralaiin_barbieux.jpg" width="140px">
</a>
</td>
<td>

#### Coralai: Intrinsic Evolution of Embodied Neural Cellular Automata Ecosystems
Published on **2024-06-14** by

Aidan **Barbieux**, Rodrigo **Canaan**

[Arxiv](https://arxiv.org/abs/2406.09654) | [Code](https://github.com/aidanbx/coralai) | [Video](https://www.youtube.com/watch?v=NL8IZQY02-8)

<details>
<summary><b>Abstract</b></summary>
This paper presents Coralai, a framework for exploring diverse ecosystems of Neural Cellular Automata (NCA). Organisms in Coralai utilize modular, GPU-accelerated Taichi kernels to interact, enact environmental changes, and evolve through local survival, merging, and mutation operations implemented with HyperNEAT and PyTorch. We provide an exploratory experiment implementing physics inspired by slime mold behavior showcasing the emergence of competition between sessile and mobile organisms, cycles of resource depletion and recovery, and symbiosis between diverse organisms. We conclude by outlining future work to discover simulation parameters through measures of multi-scale complexity and diversity. Code for Coralai is available at https: //github.com/aidanbx/coralai, video demos are available at https://www.youtube.com/watch?v= NL8IZQY02-8.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2401.06291" target="_blank">
    <img src="assets/thumbnails/2024-05-13frequency_kalkhof.jpg" width="140px">
</a>
</td>
<td>

#### Frequency-Time Diffusion with Neural Cellular Automata
Published on **2024-05-13** by

John **Kalkhof**, Arlene **Kühn**, Yannik **Frisch**, Anirban **Mukhopadhyay**

[Arxiv](https://arxiv.org/abs/2401.06291)

<details>
<summary><b>Abstract</b></summary>
Despite considerable success, large Denoising Diffusion Models (DDMs) with UNet backbone pose practical challenges, particularly on limited hardware and in processing gigapixel images. To address these limitations, we introduce two Neural Cellular Automata (NCA)-based DDMs: DiffNCA and FourierDiff-NCA. Capitalizing on the local communication capabilities of NCA, DiffNCA significantly reduces the parameter counts of NCA-based DDMs. Integrating Fourier-based diffusion enables global communication early in the diffusion process. This feature is particularly valuable in synthesizing complex images with important global features, such as the CelebA dataset. We demonstrate that even a 331k parameter Diff-NCA can generate 512 × 512 pathology slices, while FourierDiff-NCA (1.1m parameters) reaches a three times lower FID score of 43.86, compared to the four times bigger UNet (3.94m parameters) with a score of 128.2. Additionally, FourierDiff-NCA can perform diverse tasks such as super-resolution, out-of-distribution image synthesis, and inpainting without explicit training.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://pdfs.semanticscholar.org/1b31/ad84ad52ed735a04f7684bf5727a1620768d.pdf" target="_blank">
    <img src="assets/thumbnails/2024-05-11neuralcell_zhang.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata-based Land Use Changes Simulation
Published on **2024-05-11** by

Jinian **Zhang**, Lanfa **Liu**

[Paper](https://pdfs.semanticscholar.org/1b31/ad84ad52ed735a04f7684bf5727a1620768d.pdf)

<details>
<summary><b>Abstract</b></summary>
Simulating land use and land cover changes (LUCC) is important for urban planning and environmental studies. In this study, we introduce a neural cellular automata (NCA) model that integrates biological principles and convolutional neural networks (CNNs) for land use simulation. We conduct experiments in the city of Wuhan, China. The NCA model achieved the highest performance with an OA of 0.858, F1 score of 0.753, Kappa coefficient of 0.799, and FOM of 0.427. Comparisons of land use data of Wuhan city from 2000 and 2010 with the simulated optimal results indicate that forest areas closer to urban centers are more susceptible to modernization processes, showing the advantage of NCA in accurately simulating land use changes in the central urban area.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2310.14809" target="_blank">
    <img src="assets/thumbnails/2024-04-26learningsp_richardson.jpg" width="140px">
</a>
</td>
<td>

#### Learning spatio-temporal patterns with Neural Cellular Automata
Published on **2024-04-26** by

Alex D. **Richardson**, Tibor **Antal**, Richard A. **Blythe**, Linus J. **Schumacher**

[Arxiv](https://arxiv.org/abs/2310.14809) | [Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011589)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCA) are a powerful combination of machine learning and mechanistic modelling. We train NCA to learn complex dynamics from time series of images and Partial Differential Equation (PDE) trajectories. Our method is designed to identify underlying local rules that govern large scale dynamic emergent behaviours. Previous work on NCA focuses on learning rules that give stationary emergent structures. We extend NCA to capture both transient and stable structures within the same system, as well as learning rules that capture the dynamics of Turing pattern formation in nonlinear PDEs. We demonstrate that NCA can generalise very well beyond their PDE training data, we show how to constrain NCA to respect given symmetries, and we explore the effects of associated hyperparameters on model performance and stability. Being able to learn arbitrary dynamics gives NCA great potential as a data driven modelling framework, especially for modelling biological pattern formation.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2404.06279" target="_blank">
    <img src="assets/thumbnails/2024-04-09noisencan_pajouheshgar.jpg" width="140px">
</a>
</td>
<td>

#### NoiseNCA: Noisy Seed Improves Spatio-Temporal Continuity of Neural Cellular Automata
Published on **2024-04-09** by

Ehsan **Pajouheshgar**, Yitao **Xu**, Sabine **Süsstrunk**

[Project Page](https://noisenca.github.io/) | [Paper](https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00785/123473/NoiseNCA-Noisy-Seed-Improves-Spatio-Temporal) | [Arxiv](https://arxiv.org/abs/2404.06279) | [Code](https://github.com/IVRL/NoiseNCA) | [Video](https://www.youtube.com/watch?v=vb0oKr7o6kw)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCA) is a class of Cellular Automata where the update rule is parameterized by a neural network that can be trained using gradient descent. In this paper, we focus on NCA models used for texture synthesis, where the update rule is inspired by partial differential equations (PDEs) describing reaction-diffusion systems. To train the NCA model, the spatio-termporal domain is discretized, and Euler integration is used to numerically simulate the PDE. However, whether a trained NCA truly learns the continuous dynamic described by the corresponding PDE or merely overfits the discretization used in training remains an open question. We study NCA models at the limit where space-time discretization approaches continuity. We find that existing NCA models tend to overfit the training discretization, especially in the proximity of the initial condition, also called "seed". To address this, we propose a solution that utilizes uniform noise as the initial condition. We demonstrate the effectiveness of our approach in preserving the consistency of NCA dynamics across a wide range of spatio-temporal granularities. Our improved NCA model enables two new test-time interactions by allowing continuous control over the speed of pattern formation and the scale of the synthesized patterns. We demonstrate this new NCA feature in our interactive online demo. Our work reveals that NCA models can learn continuous dynamics and opens new venues for NCA research from a dynamical systems' perspective.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://aman-bhargava.com/ai/neuro/neuromorphic/2024/03/25/nca-do-active-inference.html" target="_blank">
    <img src="assets/thumbnail_placeholder.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata, Active Inference, and the Mystery of Biological Computation
Published on **2024-03-25** by

Aman **Bhargava**

[Project Page](https://aman-bhargava.com/ai/neuro/neuromorphic/2024/03/25/nca-do-active-inference.html)

<details>
<summary><b>Abstract</b></summary>
Neural cellular automata (NCA) are a fascinating class of computational models. I’m going to try convincing you of the following claims:
1. It’s worthwhile emulating biological/neuronal computation. 2. NCA are strong algorithmic solution for designing neuromorphic algorithms. 3. Demo: NCA doing active inference
integration with LLMs.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2403.15525" target="_blank">
    <img src="assets/thumbnails/2024-03-22latentneur_menta.jpg" width="140px">
</a>
</td>
<td>

#### Latent Neural Cellular Automata for Resource-Efficient Image Restoration
Published on **2024-03-22** by

Andrea **Menta**, Alberto **Archetti**, Matteo **Matteucci**

[Arxiv](https://arxiv.org/abs/2403.15525) | [Code](https://github.com/Menta99/LatentNeuralCellularAutomata)

<details>
<summary><b>Abstract</b></summary>
Neural cellular automata represent an evolution of the traditional cellular automata model, enhanced by the integration of a deep learning-based transition function. This shift from a manual to a datadriven approach significantly increases the adaptability of these models, enabling their application in diverse domains, including content generation and artificial life. However, their widespread application has been hampered by significant computational requirements. In this work, we introduce the Latent Neural Cellular Automata (LNCA) model, a novel architecture designed to address the resource limitations of neural cellular automata. Our approach shifts the computation from the conventional input space to a specially designed latent space, relying on a pre-trained autoencoder. We apply our model in the context of image restoration, which aims to reconstruct high-quality images from their degraded versions. This modification not only reduces the model’s resource consumption but also maintains a flexible framework suitable for various applications. Our model achieves a significant reduction in computational requirements while maintaining high reconstruction fidelity. This increase in efficiency allows for inputs up to 16 times larger than current state-of-the-art neural cellular automata models, using the same resources.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10480399" target="_blank">
    <img src="assets/thumbnails/2024-03-09learningab_catrina.jpg" width="140px">
</a>
</td>
<td>

#### Learning About Growing Neural Cellular Automata
Published on **2024-03-09** by

Sorana **Catrina**, Mirela **Catrina**, Alexandra **Băicoianu**, Ioana Cristina **Plajer**

[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10480399)

<details>
<summary><b>Abstract</b></summary>
Neural cellular automata have been proven effective in simulating morphogenetic processes. Developing such automata has been applied in 2D and 3D processes related to creating and regenerating complex structures and enabling their behaviors. However, neural cellular automata are inherently uncontrollable after the training process. Starting from a neural cellular automaton trained to generate a given shape from one living cell, this paper aims to gain insight into the behavior of the automaton, and to analyze the influence of the different image characteristics on the training and stabilization process and its shortcomings in different scenarios. For each considered shape, the automaton is trained on one RGB image of size 72 \times 72 pixels containing the shape on an uniform white background, in which each pixel represents a cell. The evolution of the automaton starts from one living cell, employing a shallow neural network for the update rule, followed by backpropagation after a variable number of evolutionary steps. We studied the behavior of the automaton and the way in which different components like symmetry, orientation and colours of the shape influence its growth and alteration after a number of epochs and discussed this thoroughly in the experimental section of the paper. We further discuss a pooling strategy, used to stabilize the model and illustrate the influence of this pooling on the training process. The benefits of this strategy are compared to the original model and the behavior of the automaton during its evolution is studied in detail. Finally, we compare the results of models using different filters in the first stage of feature selection. The main results of our study are the insights gained into how the neural cellular automaton works, what it is actually learning, and what influence this learning, as there are observable result differences depending on the characteristics of the input images and the filters used in the model.
</details>

</td>
</tr>
</table>


### 2023
<table><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2311.02820" target="_blank">
    <img src="assets/thumbnails/2023-11-05meshneural_pajouheshgar.jpg" width="140px">
</a>
</td>
<td>

#### Mesh Neural Cellular Automata
Published on **2023-11-05** by

Ehsan **Pajouheshgar**, Yitao **Xu**, Alexander **Mordvintsev**, Eyvind **Niklasson**, Tong **Zhang**, Sabine **Süsstrunk**

[Project Page](https://meshnca.github.io/) | [Arxiv](https://arxiv.org/abs/2311.02820) | [Code](https://github.com/IVRL/MeshNCA)

<details>
<summary><b>Abstract</b></summary>
Modeling and synthesizing textures are essential for enhancing the realism of virtual environments. Methods that directly synthesize textures in 3D offer distinct advantages to the UV-mapping-based methods as they can create seamless textures and align more closely with the ways textures form in nature. We propose Mesh Neural Cellular Automata (MeshNCA), a method for directly synthesizing dynamic textures on 3D meshes without requiring any UV maps. MeshNCA is a generalized type of cellular automata that can operate on a set of cells arranged on a non-grid structure such as vertices of a 3D mesh. While only being trained on an Icosphere mesh, MeshNCA shows remarkable generalization and can synthesize textures on any mesh in real time after the training. Additionally, it accommodates multi-modal supervision and can be trained using different targets such as images, text prompts, and motion vector fields. Moreover, we conceptualize a way of grafting trained MeshNCA instances, enabling texture interpolation. Our MeshNCA model enables real-time 3D texture synthesis on meshes and allows several user interactions including texture density/orientation control, a grafting brush, and motion speed/direction control. Finally, we implement the forward pass of our MeshNCA model using the WebGL shading language and showcase our trained models in an online interactive demo which is accessible on personal computers and smartphones. Our demo and the high resolution version of this PDF are available at https://meshnca.github.io/.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2310.18622" target="_blank">
    <img src="assets/thumbnails/2023-10-28arbitraril_zhang.jpg" width="140px">
</a>
</td>
<td>

#### Arbitrarily Scalable Environment Generators via Neural Cellular Automata
Published on **2023-10-28** by

Yulun **Zhang**, Matthew C. **Fontaine**, Varun **Bhatt**, Stefanos **Nikolaidis**, Jiaoyang **Li**

[Arxiv](https://arxiv.org/abs/2310.18622) | [Code](https://github.com/lunjohnzhang/warehouse_env_gen_nca_public)

<details>
<summary><b>Abstract</b></summary>
We study the problem of generating arbitrarily large environments to improve the throughput of multi-robot systems. Prior work proposes Quality Diversity (QD) algorithms as an effective method for optimizing the environments of automated warehouses. However, these approaches optimize only relatively small environments, falling short when it comes to replicating real-world warehouse sizes. The challenge arises from the exponential increase in the search space as the environment size increases. Additionally, the previous methods have only been tested with up to 350 robots in simulations, while practical warehouses could host thousands of robots. In this paper, instead of optimizing environments, we propose to optimize Neural Cellular Automata (NCA) environment generators via QD algorithms. We train a collection of NCA generators with QD algorithms in small environments and then generate arbitrarily large environments from the generators at test time. We show that NCA environment generators maintain consistent, regularized patterns regardless of environment size, significantly enhancing the scalability of multi-robot systems in two different domains with up to 2,350 robots. Additionally, we demonstrate that our method scales a single-agent reinforcement learning policy to arbitrarily large environments with similar patterns. We include the source code at \url{https://github.com/lunjohnzhang/warehouse_env_gen_nca_public}.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2309.16195" target="_blank">
    <img src="assets/thumbnails/2023-09-28reconstruc_seibert.jpg" width="140px">
</a>
</td>
<td>

#### Reconstructing microstructures from statistical descriptors using neural cellular automata
Published on **2023-09-28** by

Paul **Seibert**, Alexander **Raßloff**, Yichi **Zhang**, Karl **Kalina**, Paul **Reck**, Daniel **Peterseim**, Markus **Kästner**

[Paper](https://link.springer.com/article/10.1007/s40192-023-00335-1) | [Arxiv](https://arxiv.org/abs/2309.16195)

<details>
<summary><b>Abstract</b></summary>
The problem of generating microstructures of complex materials in silico has been approached from various directions including simulation, Markov, deep learning and descriptor-based approaches. This work presents a hybrid method that is inspired by all four categories and has interesting scalability properties. A neural cellular automaton is trained to evolve microstructures based on local information. Unlike most machine learning-based approaches, it does not directly require a data set of reference micrographs, but is trained from statistical microstructure descriptors that can stem from a single reference. This means that the training cost scales only with the complexity of the structure and associated descriptors. Since the size of the reconstructed structures can be set during inference, even extremely large structures can be efficiently generated. Similarly, the method is very efficient if many structures are to be reconstructed from the same descriptor for statistical evaluations. The method is formulated and discussed in detail by means of various numerical experiments, demonstrating its utility and scalability.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2309.14364" target="_blank">
    <img src="assets/thumbnails/2023-09-23automataqu_sato.jpg" width="140px">
</a>
</td>
<td>

#### Automata Quest: NCAs as a Video Game Life Mechanic
Published on **2023-09-23** by

Hiroki **Sato**, Tanner **Lund**, Takahide **Yoshida**, Atsushi **Masumori**

[Arxiv](https://arxiv.org/abs/2309.14364) | [Code](https://github.com/IkegLab/GNCA_invader)

<details>
<summary><b>Abstract</b></summary>
We study life over the course of video game history as represented by their mechanics. While there have been some variations depending on genre or "character type", we find that most games converge to a similar representation. We also examine the development of Conway's Game of Life (one of the first zero player games) and related automata that have developed over the years. With this history in mind, we investigate the viability of one popular form of automata, namely Neural Cellular Automata, as a way to more fully express life within video game settings and innovate new game mechanics or gameplay loops.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2309.13186" target="_blank">
    <img src="assets/thumbnails/2023-09-22deeplearni_li.jpg" width="140px">
</a>
</td>
<td>

#### Deep Learning with Photonic Neural Cellular Automata
Published on **2023-09-22** by

Gordon H. Y. **Li**, Christian R. **Leefmans**, James **Williams**, Robert M. **Gray**, Midya **Parto**, Alireza **Marandi**

[Arxiv](https://arxiv.org/abs/2309.13186) | [Paper](https://pubmed.ncbi.nlm.nih.gov/39379344/)

<details>
<summary><b>Abstract</b></summary>
Rapid advancements in deep learning over the past decade have fueled an insatiable demand for efficient and scalable hardware. Photonics offers a promising solution by leveraging the unique properties of light. However, conventional neural network architectures, which typically require dense programmable connections, pose several practical challenges for photonic realizations. To overcome these limitations, we propose and experimentally demonstrate Photonic Neural Cellular Automata (PNCA) for photonic deep learning with sparse connectivity. PNCA harnesses the speed and interconnectivity of photonics, as well as the self-organizing nature of cellular automata through local interactions to achieve robust, reliable, and efficient processing. We utilize linear light interference and parametric nonlinear optics for all-optical computations in a time-multiplexed photonic network to experimentally perform self-organized image classification. We demonstrate binary classification of images in the fashion-MNIST dataset using as few as 3 programmable photonic parameters, achieving an experimental accuracy of 98.0% with the ability to also recognize out-of-distribution data. The proposed PNCA approach can be adapted to a wide range of existing photonic hardware and provides a compelling alternative to conventional photonic neural networks by maximizing the advantages of light-based computing whilst mitigating their practical challenges. Our results showcase the potential of PNCA in advancing photonic deep learning and highlights a path for next-generation photonic computers.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://zenodo.org/records/8275919" target="_blank">
    <img src="assets/thumbnails/2023-08-30pathfindin.jpg" width="140px">
</a>
</td>
<td>

#### Pathfinding Neural Cellular Automata with Local Self-Attention
Published on **2023-08-30** by

Felix **Reimers**, Sanyam **Jain**, Aarati **Shrestha**, Stafano **Nichele**

[Paper](https://zenodo.org/records/8275919) | [Code](https://github.com/Deskt0r/LocalAttentionNCA)

<details>
<summary><b>Abstract</b></summary>
Abstract: Current artificial intelligence systems are rather rigid and narrow, if compared to the adaptivity and the open-endedness of living organisms. Neural Cellular Automata (NCA) are an extension of traditional CA, where the transition rule is replaced by a neural network operating on local neighborhoods. NCA provide a platform for investigating more biologically plausible features of emergent intelligence. However, an open question is how can collections of cells in an NCA be trained to collectively explore an environment in search for energy sources and find suitable paths to collect them. In this work, we utilize an NCA equipped with a local self-attention mechanism trained with gradient descent for pathfinding. Our results show that NCA can be trained to achieve such task and collect energy sources, while being able to redistribute the available energy to neighboring alive cells. Ongoing work is exploring how those abilities may be incorporated in NCA to solve tasks with increased adaptivity and general intelligence.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://dl.acm.org/doi/10.1145/3638529.3654150" target="_blank">
    <img src="assets/thumbnails/2023-07-24hierarchic_pande.jpg" width="140px">
</a>
</td>
<td>

#### Hierarchical Neural Cellular Automata
Published on **2023-07-24** by

Ritu **Pande**, Daniele **Grattarola**

[Paper](https://dl.acm.org/doi/10.1145/3638529.3654150) | [Code](https://github.com/ngaylinn/mocs-final)

<details>
<summary><b>Abstract</b></summary>
Abstract. As opposed to the traditional view wherein intelligence was believed to be a result of centralised complex monolithic rules, it is now believed that the phenomenon is multi-scale, modular and emergent (self-organising) in nature. At each scale, the constituents of an intelligent system are cognitive units driven towards a specific goal, in a specific problem space—physical, molecular, metabolic, morphological, etc. Recently, Neural Cellular Automata (NCA) have proven to be effective in simulating many evolutionary tasks, in morphological space, as self-organising dynamical systems. They are however limited in their capacity to emulate complex phenomena seen in nature such as cell differentiation (change in cell’s phenotypical and functional characteristics), metamorphosis (transformation to a new morphology after evolving to another) and apoptosis (programmed cell death). Inspired by the idea of multi-scale emergence of intelligence, we present Hierarchical NCA, a self-organising model that allows for phased, feedback-based, complex emergent behaviour. We show that by modelling emergent behaviour at two different scales in a modular hierarchy with dedicated goals, we can effectively simulate many complex evolutionary morphological tasks. Finally, we discuss the broader impact and application of this concept in areas outside biological process modelling.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://openreview.net/pdf?id=d7-ns6SZqp" target="_blank">
    <img src="assets/thumbnails/2023-07-20revariat_aillet.jpg" width="140px">
</a>
</td>
<td>

#### [Re] Variational Neural Cellular Automata
Published on **2023-07-20** by

Albert **Aillet**, Simon **Sondén**

[Paper](https://openreview.net/pdf?id=d7-ns6SZqp)

<details>
<summary><b>Abstract</b></summary>
This report presents a reproduction of a part of the results from the paper ”Variational Neural Cellular Automata” published in ICLR 2022. The authors of the original paper build upon previous research around fully‐differentiable cellular automata called Neu‐ ral Cellular Automata (NCA) and Variational Auto‐Encoders (VAE). They propose a novel generative model, a VAE whose decoder is implemented using a NCA, which they name Variational Neural Cellular Automata (VNCA)
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2307.09320" target="_blank">
    <img src="assets/thumbnails/2023-07-18biomakerca_randazzo.jpg" width="140px">
</a>
</td>
<td>

#### Biomaker CA: a Biome Maker project using Cellular Automata
Published on **2023-07-18** by

Ettore **Randazzo**, Alexander **Mordvintsev**

[Arxiv](https://arxiv.org/abs/2307.09320) | [Project Page](https://google-research.github.io/self-organising-systems/2023/biomaker-ca/) | [Code](https://github.com/google-research/self-organising-systems/tree/master/self_organising_systems/biomakerca)

<details>
<summary><b>Abstract</b></summary>
We introduce Biomaker CA: a Biome Maker project using Cellular Automata (CA). In Biomaker CA, morphogenesis is a first class citizen and small seeds need to grow into plant-like organisms to survive in a nutrient starved environment and eventually reproduce with variation so that a biome survives for long timelines. We simulate complex biomes by means of CA rules in 2D grids and parallelize all of its computation on GPUs through the Python JAX framework. We show how this project allows for several different kinds of environments and laws of ’physics’, alongside different model architectures and mutation strategies. We further analyze some configurations to show how plant agents can grow, survive, reproduce, and evolve, forming stable and unstable biomes. We then demonstrate how one can meta-evolve models to survive in a harsh environment either through end-toend meta-evolution or by a more surgical and efficient approach, called Petri dish meta-evolution. Finally, we show how to perform interactive evolution, where the user decides how to evolve a plant model interactively and then deploys it in a larger environment. We open source Biomaker CA at: https://tinyurl.com/2x8yu34s.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://www.nichele.eu/ALIFE-DistributedGhost/2-Pontes.pdf" target="_blank">
    <img src="assets/thumbnails/2023-06-19criticalne_pontes-filho.jpg" width="140px">
</a>
</td>
<td>

#### Critical Neural Cellular Automata
Published on **2023-06-19** by

Sidney **Pontes-Filho**, Stefano **Nichele**, Mikkel **Lepperød**

[Paper](https://www.nichele.eu/ALIFE-DistributedGhost/2-Pontes.pdf)

<details>
<summary><b>Abstract</b></summary>
Self-organized criticality is a behavioral state in dynamical systems that is known to present the highest computation capabilities, i.e., information transmission, storage, and processing that stays in criticality independently of the initialization or tuning, such that the critical state is an attractor to the system.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2305.13043" target="_blank">
    <img src="assets/thumbnails/2023-05-22selfrepli_sinapayen.jpg" width="140px">
</a>
</td>
<td>

#### Self-Replication, Spontaneous Mutations, and Exponential Genetic Drift in Neural Cellular Automata
Published on **2023-05-22** by

Lana **Sinapayen**

[Paper](https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00591/116893/Self-Replication-Spontaneous-Mutations-and) | [Arxiv](https://arxiv.org/abs/2305.13043) | [Code](https://github.com/LanaSina/NCA_self_replication) | [Videos](https://youtube.com/playlist?list=PLYuu1RcSnrYRhophmfolv_lmx7Qz8AP1P)

<details>
<summary><b>Abstract</b></summary>
This paper reports on patterns exhibiting self-replication with spontaneous, inheritable mutations and exponential genetic drift in Neural Cellular Automata. Despite the models not being explicitly trained for mutation or inheritability, the descendant patterns exponentially drift away from ancestral patterns, even when the automaton is deterministic. While this is far from being the first instance of evolutionary dynamics in a cellular automaton, it is the first to do so by exploiting the power and convenience of Neural Cellular Automata, arguably increasing the space of variations and the opportunity for Open Ended Evolution.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2305.12971" target="_blank">
    <img src="assets/thumbnails/2023-05-22neuralcell_stovold.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata Can Respond to Signals
Published on **2023-05-22** by

James **Stovold**

[Paper](https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00567/116835/Neural-Cellular-Automata-Can-Respond-to-Signals) | [Arxiv](https://arxiv.org/abs/2305.12971) | [Code](https://github.com/jstovold/ALIFE2023)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCAs) are a model of morphogenesis, capable of growing two-dimensional artiﬁcial organisms from a single seed cell. In this paper, we show that NCAs can be trained to respond to signals. Two types of signal are used: internal (genomically-coded) signals, and external (environmental) signals. Signals are presented to a single pixel for a single timestep.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2110.14237" target="_blank">
    <img src="assets/thumbnails/2023-05-12trainingto_dwyer.jpg" width="140px">
</a>
</td>
<td>

#### Training Topology With Graph Neural Cellular Automata
Published on **2023-05-12** by

Daniel **Dwyer**, Maxwell M. **Omwenga**

[Project Page](https://danielegrattarola.github.io/posts/2021-11-08/graph-neural-cellular-automata.html) | [Paper](https://ieeexplore.ieee.org/document/10187381) | [Arxiv](https://arxiv.org/abs/2110.14237) | [Code](https://github.com/danielegrattarola/GNCA)

<details>
<summary><b>Abstract</b></summary>
Graph neural cellular automata are a recently introduced class of computational models that extend neural cellular automata to arbitrary graphs. They are promising in various applications based on preliminary test results and the successes of related computational models, such as neural cellular automata and convolutional and graph neural networks. However, all previous graph neural cellular automaton implementations have only been able to modify data associated with the vertices and edges, not the underlying graph topology itself. Here we introduce a method of encoding graph topology information as vertex data by assigning each edge and vertex an opacity value, which is the confidence with which the model thinks that that edge or vertex should be present in the output graph. Graph neural cellular automata equipped with this encoding method, henceforth referred to as translucent graph neural cellular automata, were tested in their ability to learn to reconstruct graphs from random subgraphs of them as a proof of concept. The results suggest that translucent graph neural cellular automata are capable of this task, albeit with optimal learning rates highly dependent on the graph to be reconstructed.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2304.02354" target="_blank">
    <img src="assets/thumbnails/2023-04-05neuralcell_tang.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata for Solidification Microstructure Modelling
Published on **2023-04-05** by

Jian **Tang**, Siddhant **Kumar**, Laura **De Lorenzis**, Ehsan **Hosseini**

[Arxiv](https://arxiv.org/abs/2304.02354) | [Code](https://github.com/HighTempIntegrity/JianTang-NCA01)

<details>
<summary><b>Abstract</b></summary>
We propose Neural Cellular Automata (NCA) to simulate the microstructure development during the solidification process in metals. Based on convolutional neural networks, NCA can learn essential solidification features, such as preferred growth direction and competitive grain growth, and are up to six orders of magnitude faster than the conventional Cellular Automata (CA). Notably, NCA delivers reliable predictions also outside their training range, which indicates that they learn the physics of the solidification process. While in this study we employ data produced by CA for training, NCA can be trained based on any microstructural simulation data, e.g. from phase-field models.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2211.11417" target="_blank">
    <img src="assets/thumbnails/2023-03-30dyncareal_pajouheshgar.jpg" width="140px">
</a>
</td>
<td>

#### DyNCA: Real-time Dynamic Texture Synthesis Using Neural Cellular Automata
Published on **2023-03-30** by

Ehsan **Pajouheshgar**, Yitao **Xu**, Tong **Zhang**, Sabine **Süsstrunk**

[Project Page](https://dynca.github.io/) | [Arxiv](https://arxiv.org/abs/2211.11417) | [Code](https://github.com/IVRL/DyNCA) | [Video](https://www.youtube.com/watch?v=ELZC2mX5Z9U)

<details>
<summary><b>Abstract</b></summary>
Current Dynamic Texture Synthesis (DyTS) models can synthesize realistic videos. However, they require a slow iterative optimization process to synthesize a single fixed-size short video, and they do not offer any post-training control over the synthesis process. We propose Dynamic Neural Cellular Automata (DyNCA), a framework for real-time and controllable dynamic texture synthesis. Our method is built upon the recently introduced NCA models and can synthesize infinitely long and arbitrary-sized realistic video textures in real time. We quantitatively and qualitatively evaluate our model and show that our synthesized videos appear more realistic than the existing results. We improve the SOTA DyTS performance by $2\sim 4$ orders of magnitude. Moreover, our model offers several real-time video controls including motion speed, motion direction, and an editing brush tool. We exhibit our trained models in an online interactive demo that runs on local hardware and is accessible on personal computers and smartphones.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2302.10197" target="_blank">
    <img src="assets/thumbnails/2023-02-19growingste_randazzo.jpg" width="140px">
</a>
</td>
<td>

#### Growing Steerable Neural Cellular Automata
Published on **2023-02-19** by

Ettore **Randazzo**, Alexander **Mordvintsev**, Craig **Fouts**

[Arxiv](https://arxiv.org/abs/2302.10197) | [Code](ttps://github.com/google-research/self-organising-systems/tree/master/isotropic_nca)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCA) models have shown remarkable capacity for pattern formation and complex global behaviors stemming from local coordination. However, in the original implementation of NCA, cells are incapable of adjusting their own orientation, and it is the responsibility of the model designer to orient them externally. A recent isotropic variant of NCA (Growing Isotropic Neural Cellular Automata) makes the model orientation-independent - cells can no longer tell up from down, nor left from right - by removing its dependency on perceiving the gradient of spatial states in its neighborhood. In this work, we revisit NCA with a different approach: we make each cell responsible for its own orientation by allowing it to "turn" as determined by an adjustable internal state. The resulting Steerable NCA contains cells of varying orientation embedded in the same pattern. We observe how, while Isotropic NCA are orientation-agnostic, Steerable NCA have chirality: they have a predetermined left-right symmetry. We therefore show that we can train Steerable NCA in similar but simpler ways than their Isotropic variant by: (1) breaking symmetries using only two seeds, or (2) introducing a rotation-invariant training objective and relying on asynchronous cell updates to break the up-down symmetry of the system.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2302.03473" target="_blank">
    <img src="assets/thumbnails/2023-02-07medncaro_kalkhof.jpg" width="140px">
</a>
</td>
<td>

#### Med-NCA: Robust and Lightweight Segmentation with Neural Cellular Automata
Published on **2023-02-07** by

John **Kalkhof**, Camila **González**, Anirban **Mukhopadhyay**

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-34048-2_54) | [Arxiv](https://arxiv.org/abs/2302.03473) | [Code](https://github.com/MECLabTUDA/Med-NCA)

<details>
<summary><b>Abstract</b></summary>
Access to the proper infrastructure is critical when performing medical image segmentation with Deep Learning. This requirement makes it difficult to run state-of-the-art segmentation models in resource-constrained scenarios like primary care facilities in rural areas and during crises. The recently emerging field of Neural Cellular Automata (NCA) has shown that locally interacting one-cell models can achieve competitive results in tasks such as image generation or segmentations in low-resolution inputs. However, they are constrained by high VRAM requirements and the difficulty of reaching convergence for high-resolution images. To counteract these limitations we propose Med-NCA, an end-to-end NCA training pipeline for high-resolution image segmentation. Our method follows a two-step process. Global knowledge is first communicated between cells across the downscaled image. Following that, patch-based segmentation is performed. Our proposed Med-NCA outperforms the classic UNet by 2% and 3% Dice for hippocampus and prostate segmentation, respectively, while also being 500 times smaller. We also show that Med-NCA is by design invariant with respect to image scale, shape and translation, experiencing only slight performance degradation even with strong shifts; and is robust against MRI acquisition artefacts. Med-NCA enables high-resolution medical image segmentation even on a Raspberry Pi B+, arguably the smallest device able to run PyTorch and that can be powered by a standard power bank.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2301.10497" target="_blank">
    <img src="assets/thumbnails/2023-01-25enequiv_gala.jpg" width="140px">
</a>
</td>
<td>

#### E(n)-equivariant Graph Neural Cellular Automata
Published on **2023-01-25** by

Gennaro **Gala**, Daniele **Grattarola**, Erik **Quaeghebeur**

[Arxiv](https://arxiv.org/abs/2301.10497) | [Code](https://github.com/gengala/egnca)

<details>
<summary><b>Abstract</b></summary>
Cellular automata (CAs) are computational models exhibiting rich dynamics emerging from the local interaction of cells arranged in a regular lattice. Graph CAs (GCAs) generalise standard CAs by allowing for arbitrary graphs rather than regular lattices, similar to how Graph Neural Networks (GNNs) generalise Convolutional NNs. Recently, Graph Neural CAs (GNCAs) have been proposed as models built on top of standard GNNs that can be trained to approximate the transition rule of any arbitrary GCA. Existing GNCAs are anisotropic in the sense that their transition rules are not equivariant to translation, rotation, and reflection of the nodes' spatial locations. However, it is desirable for instances related by such transformations to be treated identically by the model. By replacing standard graph convolutions with E(n)-equivariant ones, we avoid anisotropy by design and propose a class of isotropic automata that we call E(n)-GNCAs. These models are lightweight, but can nevertheless handle large graphs, capture complex dynamics and exhibit emergent self-organising behaviours. We showcase the broad and successful applicability of E(n)-GNCAs on three different tasks: (i) pattern formation, (ii) graph auto-encoding, and (iii) simulation of E(n)-equivariant dynamical systems.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2301.06820" target="_blank">
    <img src="assets/thumbnails/2023-01-17pathfindin_earle.jpg" width="140px">
</a>
</td>
<td>

#### Pathfinding Neural Cellular Automata
Published on **2023-01-17** by

Sam **Earle**, Ozlem **Yildiz**, Julian **Togelius**, Chinmay **Hegde**

[Arxiv](https://arxiv.org/abs/2301.06820) | [Code](https://github.com/smearle/pathfinding-nca)

<details>
<summary><b>Abstract</b></summary>
Pathfinding makes up an important sub-component of a broad range of complex tasks in AI, such as robot path planning, transport routing, and game playing. While classical algorithms can efficiently compute shortest paths, neural networks could be better suited to adapting these sub-routines to more complex and intractable tasks. As a step toward developing such networks, we hand-code and learn models for Breadth-First Search (BFS), i.e. shortest path finding, using the unified architectural framework of Neural Cellular Automata, which are iterative neural networks with equal-size inputs and outputs. Similarly, we present a neural implementation of Depth-First Search (DFS), and outline how it can be combined with neural BFS to produce an NCA for computing diameter of a graph. We experiment with architectural modifications inspired by these hand-coded NCAs, training networks from scratch to solve the diameter problem on grid mazes while exhibiting strong generalization ability. Finally, we introduce a scheme in which data points are mutated adversarially during training. We find that adversarially evolving mazes leads to increased generalization on out-of-distribution examples, while at the same time generating data-sets with significantly more complex solutions for reasoning tasks.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2301.00897" target="_blank">
    <img src="assets/thumbnails/2023-01-02gameofinte_grieskamp.jpg" width="140px">
</a>
</td>
<td>

#### Game of Intelligent Life
Published on **2023-01-02** by

Marlene **Grieskamp**, Chaytan **Inman**, Shaun **Lee**

[Arxiv](https://arxiv.org/abs/2301.00897) | [Code](https://github.com/chaytanc/game-of-intelligent-life)

<details>
<summary><b>Abstract</b></summary>
Cellular automata (CA) captivate researchers due to teh emergent, complex individualized behavior that simple global rules of interaction enact. Recent advances in the field have combined CA with convolutional neural networks to achieve self-regenerating images. This new branch of CA is called neural cellular automata [1]. The goal of this project is to use the idea of idea of neural cellular automata to grow prediction machines. We place many different convolutional neural networks in a grid. Each conv net cell outputs a prediction of what the next state will be, and minimizes predictive error. Cells received their neighbors' colors and fitnesses as input. Each cell's fitness score described how accurate its predictions were. Cells could also move to explore their environment and some stochasticity was applied to movement.
</details>

</td>
</tr>
</table>


### 2022
<table><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2211.01233" target="_blank">
    <img src="assets/thumbnails/2022-11-02attention_tesfaldet.jpg" width="140px">
</a>
</td>
<td>

#### Attention-based Neural Cellular Automata
Published on **2022-11-02** by

Mattie **Tesfaldet**, Derek **Nowrouzezahrai**, Christopher **Pal**

[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/361e5112d2eca09513bbd266e4b2d2be-Paper-Conference.pdf) | [Arxiv](https://arxiv.org/abs/2211.01233)

<details>
<summary><b>Abstract</b></summary>
Recent extensions of Cellular Automata (CA) have incorporated key ideas from modern deep learning, dramatically extending their capabilities and catalyzing a new family of Neural Cellular Automata (NCA) techniques. Inspired by Transformer-based architectures, our work presents a new class of $\textit{attention-based}$ NCAs formed using a spatially localized$\unicode{x2014}$yet globally organized$\unicode{x2014}$self-attention scheme. We introduce an instance of this class named $\textit{Vision Transformer Cellular Automata}$ (ViTCA). We present quantitative and qualitative results on denoising autoencoding across six benchmark datasets, comparing ViTCA to a U-Net, a U-Net-based CA baseline (UNetCA), and a Vision Transformer (ViT). When comparing across architectures configured to similar parameter complexity, ViTCA architectures yield superior performance across all benchmarks and for nearly every evaluation metric. We present an ablation study on various architectural configurations of ViTCA, an analysis of its effect on cell states, and an investigation on its inductive biases. Finally, we examine its learned representations via linear probes on its converged cell state hidden representations, yielding, on average, superior results when compared to our U-Net, ViT, and UNetCA baselines.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://github.com/PWhiddy/Growing-Neural-Cellular-Automata-Pytorch" target="_blank">
    <img src="assets/thumbnails/2022-10-29growingneu.jpg" width="140px">
</a>
</td>
<td>

#### Growing Neural Cellular Automata - Task Experiments
Published on **2022-10-29** by

Peter **Whidden**

[Code](https://github.com/PWhiddy/Growing-Neural-Cellular-Automata-Pytorch) | [Project Page](https://transdimensional.xyz/projects/neural_ca/index.html)

<details>
<summary><b>Abstract</b></summary>
Extended experiments of "Growing Neural Cellular Automata" https://distill.pub/2020/growing-ca/
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2203.07548" target="_blank">
    <img src="assets/thumbnails/2022-07-31physicalne_walker.jpg" width="140px">
</a>
</td>
<td>

#### Physical Neural Cellular Automata for 2D Shape Classification
Published on **2022-07-31** by

Kathryn **Walker**, Rasmus Berg **Palm**, Rodrigo Moreno **Garcia**, Andres **Faina**, Kasper **Stoy**, Sebastian **Risi**

[Arxiv](https://arxiv.org/abs/2203.07548) | [Code](https://github.com/kattwalker/projectcube) | [Video](https://www.youtube.com/watch?v=0TCOkE4keyc)

<details>
<summary><b>Abstract</b></summary>
Materials with the ability to self-classify their own shape have the potential to advance a wide range of engineering applications and industries. Biological systems possess the ability not only to self-reconfigure but also to self-classify themselves to determine a general shape and function. Previous work into modular robotics systems has only enabled self-recognition and self-reconfiguration into a specific target shape, missing the inherent robustness present in nature to self-classify. In this paper we therefore take advantage of recent advances in deep learning and neural cellular automata, and present a simple modular 2D robotic system that can infer its own class of shape through the local communication of its components. Furthermore, we show that our system can be successfully transferred to hardware which thus opens opportunities for future self-classifying machines. Code available at https://github.com/kattwalker/projectcube. Video available at https://youtu.be/0TCOkE4keyc.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2205.06771" target="_blank">
    <img src="assets/thumbnails/2022-07-19empoweredn_grasso.jpg" width="140px">
</a>
</td>
<td>

#### Empowered neural cellular automata
Published on **2022-07-19** by

Caitlin **Grasso**, Josh **Bongard**

[Paper](https://dl.acm.org/doi/10.1145/3520304.3529067) | [Arxiv](https://arxiv.org/abs/2205.06771) | [Code](https://github.com/caitlingrasso/empowered-nca)

<details>
<summary><b>Abstract</b></summary>
Information-theoretic fitness functions are becoming increasingly popular to produce generally useful, task-independent behaviors. One such universal function, dubbed empowerment, measures the amount of control an agent exerts on its environment via its sensorimotor system. Specifically, empowerment attempts to maximize the mutual information between an agent's actions and its received sensor states at a later point in time. Traditionally, empowerment has been applied to a conventional sensorimotor apparatus, such as a robot. Here, we expand the approach to a distributed, multi-agent sensorimotor system embodied by a neural cellular automaton (NCA). We show that the addition of empowerment as a secondary objective in the evolution of NCA to perform the task of morphogenesis, growing and maintaining a pre-specified shape, results in higher fitness compared to evolving for morphogenesis alone. Results suggest there may be a synergistic relationship between morphogenesis and empowerment. That is, indirectly selecting for coordination between neighboring cells over the duration of development is beneficial to the developmental process itself. Such a finding may have applications in developmental biology by providing potential mechanisms of communication between cells during growth from a single cell to a multicellular, target morphology. Source code for the experiments in this paper can be found at: https://github.com/caitlingrasso/empowered-nca.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://direct.mit.edu/isal/proceedings/isal2022/34/62/112313" target="_blank">
    <img src="assets/thumbnails/2022-07-18gradientcl_kuriyama.jpg" width="140px">
</a>
</td>
<td>

#### Gradient Climbing Neural Cellular Automata
Published on **2022-07-18** by

Shuto **Kuriyama**, Wataru **Noguchi**, Hiroyuki **Iizuka**, Keisuke **Suzuki**, Masahito **Yamamoto**

[Paper](https://direct.mit.edu/isal/proceedings/isal2022/34/62/112313)

<details>
<summary><b>Abstract</b></summary>
Abstract. Chemotaxis is a phenomenon whereby organisms like ameba direct their movements responding to their environmental gradients, often called gradient climbing. It is considered to be the origin of self-movement that characterizes life forms. In this work, we have simulated the gradient climbing behaviour on Neural Cellular Automata (NCA) that has recently been proposed as a model to simulate morphogenesis. NCA is a cellular automata model using deep networks for its learnable update rule and it generates a target cell pattern from a single cell through local interactions among cells. Our model, Gradient Climbing Neural Cellular Automata (GCNCA), has an additional feature that enables itself to move a generated pattern by responding to a gradient injected into its cell states.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2206.03563" target="_blank">
    <img src="assets/thumbnails/2022-06-15twowaysofu_adams.jpg" width="140px">
</a>
</td>
<td>

#### Two Ways of Understanding Social Dynamics: Analyzing the Predictability of Emergence of Objects in Reddit r/place Dependent on Locality in Space and Time
Published on **2022-06-15** by

Alyssa M. **Adams**, Javier **Fernandez**, Olaf **Witkowski**

[Arxiv](https://arxiv.org/abs/2206.03563)

<details>
<summary><b>Abstract</b></summary>
Lately, studying social dynamics in interacting agents has been boosted by the power of computer models, which bring the richness of qualitative work, while offering the precision, transparency, extensiveness, and replicability of statistical and mathematical approaches. A particular set of phenomena for the study of social dynamics is Web collaborative platforms. A dataset of interest is r/place, a collaborative social experiment held in 2017 on Reddit, which consisted of a shared online canvas of 1000 pixels by 1000 pixels co-edited by over a million recorded users over 72 hours. In this paper, we designed and compared two methods to analyze the dynamics of this experiment. Our ﬁrst method consisted in approximating the set of 2D cellular-automata-like rules used to generate the canvas images and how these rules change over time. The second method consisted in a convolutional neural network (CNN) that learned an approximation to the generative rules in order to generate the complex outcomes of the canvas. Our results indicate varying context-size dependencies for the predictability of different objects in r/place in time and space. They also indicate a surprising peak in difﬁculty to statistically infer behavioral rules towards the middle of the social experiment, while user interactions did not drop until before the end. The combination of our two approaches, one rule-based and the other statistical CNN-based, shows the ability to highlight diverse aspects of analyzing social dynamics.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://mainakdeb.github.io/posts/clip-guided-nca/" target="_blank">
    <img src="assets/thumbnails/2022-06-06clipguided.jpg" width="140px">
</a>
</td>
<td>

#### CLIP Guided Neural Cellular Automata using PyTorch
Published on **2022-06-06** by

Mainak **Deb**

[Project Page](https://mainakdeb.github.io/posts/clip-guided-nca/)

<details>
<summary><b>Abstract</b></summary>

</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://greydanus.github.io/2022/05/24/studying-growth/" target="_blank">
    <img src="assets/thumbnails/2022-05-22studying_greydanus.jpg" width="140px">
</a>
</td>
<td>

#### Studying Growth with Neural Cellular Automata
Published on **2022-05-24** by

Sam **Greydanus**

[Project Page](https://greydanus.github.io/2022/05/24/studying-growth/) | [Code](https://github.com/greydanus/studying_growth)

<details>
<summary><b>Abstract</b></summary>
How does a single fertilized egg grow into a population of seventy trillion cells: a population that can walk, talk, and write sonnets? This is one of the great unanswered questions of biology. We may never finish answering it, but it is a productive question nonetheless. In asking it, scientists have discovered the structure of DNA, sequenced the human genome, and made essential contributions to modern medicine. In this post, we will explore this question with a new tool called Neural Cellular Automata (NCA).
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://wandb.ai/johnowhitaker/nca/reports/Fun-with-Neural-Cellular-Automata--VmlldzoyMDQ5Mjg0" target="_blank">
    <img src="assets/thumbnail_placeholder.jpg" width="140px">
</a>
</td>
<td>

#### Fun with Neural Cellular Automata
Published on **2022-05-22** by

Jonathan **Whithaker**

[Project Page](https://wandb.ai/johnowhitaker/nca/reports/Fun-with-Neural-Cellular-Automata--VmlldzoyMDQ5Mjg0) | [Code](https://colab.research.google.com/drive/19EgmX5byZvL3yDRl5cGVyRekDuJ89Rq6)

<details>
<summary><b>Abstract</b></summary>
In this article, we take a look at how to make pretty pictures using differentiable self-organizing systems, using Weights & Biases to keep track of our results.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2205.01681" target="_blank">
    <img src="assets/thumbnails/2022-05-03growingiso_mordvintsev.jpg" width="140px">
</a>
</td>
<td>

#### Growing Isotropic Neural Cellular Automata
Published on **2022-05-03** by

Alexander **Mordvintsev**, Ettore **Randazzo**, Craig **Fouts**

[Paper](https://direct.mit.edu/isal/proceedings/isal2022/34/65/112305) | [Project Page](https://google-research.github.io/self-organising-systems/isonca/) | [Arxiv](https://arxiv.org/abs/2205.01681) | [Video](https://www.youtube.com/watch?v=OZggz3EMjaY) | [Code Structured Seeds](https://github.com/google-research/self-organising-systems/blob/master/isotropic_nca/blogpost_isonca_structured_seeds_pytorch.ipynb) | [Code Rotation Invariant](https://github.com/google-research/self-organising-systems/blob/master/isotropic_nca/blogpost_isonca_single_seed_pytorch.ipynb)

<details>
<summary><b>Abstract</b></summary>
Modeling the ability of multicellular organisms to build and maintain their bodies through local interactions between individual cells (morphogenesis) is a long-standing challenge of developmental biology. Recently, the Neural Cellular Automata (NCA) model was proposed as a way to ﬁnd local system rules that produce a desired global behaviour, such as growing and persisting a predeﬁned pattern, by repeatedly applying the same rule over a grid starting from a single cell. In this work we argue that the original Growing NCA model has an important limitation: anisotropy of the learned update rule. This implies the presence of an external factor that orients the cells in a particular direction. In other words, “physical” rules of the underlying system are not invariant to rotation, thus prohibiting the existence of differently oriented instances of the target pattern on the same grid. We propose a modiﬁed Isotropic NCA model that does not have this limitation. We demonstrate that cell systems can be trained to grow accurate asymmetrical patterns through either of two methods: by breaking symmetries using structured seeds; or by introducing a rotation-reﬂection invariant training objective and relying on symmetry breaking caused by asynchronous cell updates.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2205.06806" target="_blank">
    <img src="assets/thumbnails/2022-04-25goalguide_sudhakaran.jpg" width="140px">
</a>
</td>
<td>

#### Goal-Guided Neural Cellular Automata: Learning to Control Self-Organising Systems
Published on **2022-04-25** by

Shyam **Sudhakaran**, Elias **Najarro**, Sebastian **Risi**

[Arxiv](https://arxiv.org/abs/2205.06806)

<details>
<summary><b>Abstract</b></summary>
Inspired by cellular growth and self-organization, Neural Cellular Automata (NCAs) have been capable of ”growing” artiﬁcial cells into images, 3D structures, and even functional machines. NCAs are ﬂexible and robust computational systems but – similarly to many other self-organizing systems — inherently uncontrollable during and after their growth process. We present an approach to control these type of systems called Goal-Guided Neural Cellular Automata (GoalNCA), which leverages goal encodings to control cell behavior dynamically at every step of cellular growth. This approach enables the NCA to continually change behavior, and in some cases, generalize its behavior to unseen scenarios. We also demonstrate the robustness of the NCA with its ability to preserve task performance, even when only a portion of cells receive goal information.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2204.11674" target="_blank">
    <img src="assets/thumbnails/2022-04-25hyperncag_najarro.jpg" width="140px">
</a>
</td>
<td>

#### HyperNCA: Growing Developmental Networks with Neural Cellular Automata
Published on **2022-04-25** by

Elias **Najarro**, Shyam **Sudhakaran**, Claire **Glanois**, Sebastian **Risi**

[Arxiv](https://arxiv.org/abs/2204.11674) | [Project Page](https://iclr.cc/virtual/2022/8100) | [Code](https://github.com/enajx/hyperNCA)

<details>
<summary><b>Abstract</b></summary>
In contrast to deep reinforcement learning agents, biological neural networks are grown through a self-organized developmental process. Here we propose a new hypernetwork approach to grow artiﬁcial neural networks based on neural cellular automata (NCA). Inspired by self-organising systems and information-theoretic approaches to developmental biology, we show that our HyperNCA method can grow neural networks capable of solving common reinforcement learning tasks. Finally, we explore how the same approach can be used to build developmental metamorphosis networks capable of transforming their weights to solve variations of the initial RL task.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2204.02099" target="_blank">
    <img src="assets/thumbnails/2022-04-05collective_nadizar.jpg" width="140px">
</a>
</td>
<td>

#### Collective control of modular soft robots via embodied Spiking Neural Cellular Automata
Published on **2022-04-05** by

Giorgia **Nadizar**, Eric **Medvet**, Stefano **Nichele**, Sidney **Pontes-Filho**

[Arxiv](https://arxiv.org/abs/2204.02099)

<details>
<summary><b>Abstract</b></summary>
Voxel-based Soft Robots (VSRs) are a form of modular soft robots, composed of several deformable cubes, i.e., voxels. Each VSR is thus an ensemble of simple agents, namely the voxels, which must cooperate to give rise to the overall VSR behavior. Within this paradigm, collective intelligence plays a key role in enabling the emerge of coordination, as each voxel is independently controlled, exploiting only the local sensory information together with some knowledge passed from its direct neighbors (distributed or collective control). In this work, we propose a novel form of collective control, inﬂuenced by Neural Cellular Automata (NCA) and based on the bio-inspired Spiking Neural Networks: the embodied Spiking NCA (SNCA). We experiment with different variants of SNCA, and ﬁnd them to be competitive with the state-of-the-art distributed controllers for the task of locomotion. In addition, our ﬁndings show signiﬁcant improvement with respect to the baseline in terms of adaptability to unforeseen environmental changes, which could be a determining factor for physical practicability of VSRs.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00521/112290/Adversarial-Takeover-of-Neural-Cellular-Automata" target="_blank">
    <img src="assets/thumbnails/2022-03-12adversaria_cavuoti.jpg" width="140px">
</a>
</td>
<td>

#### Adversarial Takeover of Neural Cellular Automata
Published on **2022-03-12** by

Lorenzo **Cavuoti**, Francesco **Sacco**, Ettore **Randazzo**, Michael **Levin**

[Paper](https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00521/112290/Adversarial-Takeover-of-Neural-Cellular-Automata) | [Project Page](https://letteraunica.github.io/neural_cellular_automata/) | [Code](https://github.com/LetteraUnica/neural_cellular_automata)

<details>
<summary><b>Abstract</b></summary>
The biggest open problems in the life sciences concern the algorithms by which competent subunits (cells) could cooperate to form large-scale structures with new, system-level properties. In synthetic bioengineering, multiple cells of diverse origin can be included in chimeric constructs. To facilitate progress in this field, we sought an understanding of multi-scale decision-making by diverse subunits beyond those observed in frozen accidents of biological phylogeny: abstract models of life-as-it-can-be. Neural Cellular Automata (NCA) are a very good inspiration for understanding current and possible living organisms: researchers managed to create NCA that are able to converge to any morphology. In order to simulate a more dynamic situation, we took the NCA model and generalized it to consider multiple NCA rules. We then used this generalized model to change the behavior of a NCA by injecting other types of cells (adversaries) and letting them take over the entire organism to solve a different task. Next we demonstrate that it is possible to stop aging in an existing NCA by injecting adversaries that follow a different rule. Finally, we quantify a distance between NCAs and develop a procedure that allows us to find adversaries close to the original cells.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2201.12360" target="_blank">
    <img src="assets/thumbnails/2022-03-12variationa_palm.jpg" width="140px">
</a>
</td>
<td>

#### Variational Neural Cellular Automata
Published on **2022-03-12** by

Rasmus Berg **Palm**, Miguel **Gonzalez-Duque**, Shyam **Sudhakaran**, Sebastian **Risi**

[Arxiv](https://arxiv.org/abs/2201.12360) | [Code](https://github.com/rasmusbergpalm/vnca)

<details>
<summary><b>Abstract</b></summary>
In nature, the process of cellular growth and differentiation has lead to an amazing diversity of organisms — algae, starﬁsh, giant sequoia, tardigrades, and orcas are all created by the same generative process. Inspired by the incredible diversity of this biological generative process, we propose a generative model, the Variational Neural Cellular Automata (VNCA), which is loosely inspired by the biological processes of cellular growth and differentiation. Unlike previous related works, the VNCA is a proper probabilistic generative model, and we evaluate it according to best practices. We ﬁnd that the VNCA learns to reconstruct samples well and that despite its relatively few parameters and simple local-only communication, the VNCA can learn to generate a large variety of output from information encoded in a common vector format. While there is a signiﬁcant gap to the current state-of-the-art in terms of generative modeling performance, we show that the VNCA can learn a purely self-organizing generative process of data. Additionally, we show that the VNCA can learn a distribution of stable attractors that can recover from signiﬁcant damage.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2109.05489" target="_blank">
    <img src="assets/thumbnails/2022-02-17illuminati_earle.jpg" width="140px">
</a>
</td>
<td>

#### Illuminating Diverse Neural Cellular Automata for Level Generation
Published on **2022-02-17** by

Sam **Earle**, Justin **Snider**, Matthew C. **Fontaine**, Stefanos **Nikolaidis**, Julian **Togelius**

[Paper](https://dl.acm.org/doi/10.1145/3512290.3528754) | [Arxiv](https://arxiv.org/abs/2109.05489) | [Code](https://github.com/smearle/control-pcgrl)

<details>
<summary><b>Abstract</b></summary>
We present a method of generating diverse collections of neural cellular automata (NCA) to design video game levels. While NCAs have so far only been trained via supervised learning, we present a quality diversity (QD) approach to generating a collection of NCA level generators. By framing the problem as a QD problem, our approach can train diverse level generators, whose output levels vary based on aesthetic or functional criteria. To efficiently generate NCAs, we train generators via Covariance Matrix Adaptation MAPElites (CMA-ME), a quality diversity algorithm which specializes in continuous search spaces. We apply our new method to generate level generators for several 2D tile-based games: a maze game, Sokoban, and Zelda. Our results show that CMA-ME can generate small NCAs that are diverse yet capable, often satisfying complex solvability criteria for deterministic agents. We compare against a Compositional Pattern-Producing Network (CPPN) baseline trained to produce diverse collections of generators and show that the NCA representation yields a better exploration of level-space.
</details>

</td>
</tr>
</table>


### 2021
<table><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2111.13545" target="_blank">
    <img src="assets/thumbnails/2021-11-26muncat_mordvintsev.jpg" width="140px">
</a>
</td>
<td>

#### $\mu$NCA: Texture Generation with Ultra-Compact Neural Cellular Automata
Published on **2021-11-26** by

Alexander **Mordvintsev**, Eyvind **Niklasson**

[Arxiv](https://arxiv.org/abs/2111.13545) | [Code 1](https://www.shadertoy.com/view/slGGzD) | [Code 2](https://www.shadertoy.com/view/styGzD)

<details>
<summary><b>Abstract</b></summary>
We study the problem of example-based procedural texture synthesis using highly compact models. Given a sample image, we use differentiable programming to train a generative process, parameterised by a recurrent Neural Cellular Automata (NCA) rule. Contrary to the common belief that neural networks should be signiﬁcantly overparameterised, we demonstrate that our model architecture and training procedure allows for representing complex texture patterns using just a few hundred learned parameters, making their expressivity comparable to hand-engineered procedural texture generating programs. The smallest models from the proposed µNCA family scale down to 68 parameters. When using quantisation to one byte per parameter, proposed models can be shrunk to a size range between 588 and 68 bytes. Implementation of a texture generator that uses these parameters to produce images is possible with just a few lines of GLSL1 or C code.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://ieeexplore.ieee.org/abstract/document/9599243" target="_blank">
    <img src="assets/thumbnails/2021-09-12whatcancol_vardy.jpg" width="140px">
</a>
</td>
<td>

#### What Can Collective Construction Learn from Neural Cellular Automata?
Published on **2021-09-12** by

Andrew **Vardy**

[Paper](https://ieeexplore.ieee.org/abstract/document/9599243)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCA) have been trained to produce target images and shapes and even to regenerate after damage. These are highly attractive properties that can inform work on collective robotic construction. We discuss concepts from NCA that may be useful for collective robotic construction and discuss how the problems of morphogenesis and construction differ. As a concrete first step, we propose a simplified variant of an existing NCA model to explore the consequences of reducing the number of state channels encoded. We find that the NCA can still reproduce trained images. This bodes well for translating ideas from N CAs to collective robotic construction.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/computing/public/2021-ug-projects/2021-msc-individual-projects/Convolutional-autoencoder-based-image-compression-in-the-image-sensor.pdf" target="_blank">
    <img src="assets/thumbnails/2021-09-02neuralcell_dudziak.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata on a Focal Plane
Published on **2021-09-02** by

Maciej **Dudziak**, Paul **Kelly**, Wayne **Luk**

[Paper](https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/computing/public/2021-ug-projects/2021-msc-individual-projects/Convolutional-autoencoder-based-image-compression-in-the-image-sensor.pdf)

<details>
<summary><b>Abstract</b></summary>
The cellular automata is a discrete mathematical model of computation that consists of a regular grid of cells, each being in one of the finite number of states, with its next iteration determined by a set of rules applied locally at each cell. Traditionally those rules are predefined and crafted for a specific, desired operation, though recently it has been shown that they can also be learned with convolutional neural networks and applied to perform image classification or segmentation. The locality of the computation and presence of independent and organised agents resembles the architecture of the Focal Plane Sensor Processors (FPSP), a family of vision chips where every pixel/photo-diode in the array is equipped with a processing unit allowing for computations directly on a focal plane. In this thesis, we investigate the suitability of FPSP architectures for hosting the neural cellular automata (NCA) by implementing an example automaton for the self-classification of MNIST digits. In the process, we present a comprehensive analysis of the computational and memory requirements of the NCA and relates them to the capabilities of existing FPSP devices, providing guidelines for the development of automata that is less resource consuming. We develop a step-by-step quantisation procedure customised for the application of neural cellular automata on FPSP, which reduces the 32-bit floating-point model into the integer representation with 5-bit weights, 10-bit input and 9-bit activations while entirely maintaining its performance and precision. Our implementation of this quantised NCA achieves an operation at 74 FPS with using 77.68 mJ energy per frame on the simulated FPSP device. This work could also be treated as an exploration of neural cellular automata as a potential general computation algorithm for the FPSP family architectures and also as an exploration for the possible development directions of the next generation of FPSPs. Hence, we conclude this thesis with a discussion on the discovered bottlenecks and limitations and describing the ideas on possible solutions, such as more sophisticated patch multiplexing/superpixels.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://hugocisneros.com/blog/open-ended-creation-of-hybrid-creatures-with-neural-cellular-automata/" target="_blank">
    <img src="assets/thumbnails/2021-06-06openended.jpg" width="140px">
</a>
</td>
<td>

#### Open-ended creation of hybrid creatures with Neural Cellular Automata
Published on **2021-08-03** by



[Blog](https://hugocisneros.com/blog/open-ended-creation-of-hybrid-creatures-with-neural-cellular-automata/) | [Code](https://github.com/hugcis/hybrid-nca-evocraft) | [Video](https://www.youtube.com/watch?v=RdUCL4Fs0XY)

<details>
<summary><b>Abstract</b></summary>
Minecraft Open-Ended Challenge 2021 Submission
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00461/102923/Asynchronicity-in-Neural-Cellular-Automata" target="_blank">
    <img src="assets/thumbnails/2021-07-19asynchroni_niklasson.jpg" width="140px">
</a>
</td>
<td>

#### Asynchronicity in Neural Cellular Automata
Published on **2021-07-19** by

Eyvind **Niklasson**, Alexander **Mordvintsev**, Ettore **Randazzo**

[Paper](https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00461/102923/Asynchronicity-in-Neural-Cellular-Automata) | [Code](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/async.ipynb)

<details>
<summary><b>Abstract</b></summary>
Abstract. Cellular Automata have intrigued curious minds for the better part of the last century, with significant contributions to their field from the likes of Von Neumann et al. (1966), John Conway (Gardner (1970)), and Wolfram and Gad-el Hak (2003). They can simulate and model phenomena in biology, chemistry, and physics (Chopard and Droz (1998)). Recently, Neural Cellular Automata (NCA) have demonstrated a capacity to learn complex behaviour, including constructing a target morphology (Mordvintsev et al. (2020)), classifying the shape they occupy (Randazzo et al. (2020)), or segmentation of images (Sandler et al. (2020)). As a computational model, NCA have appealing properties. They are parallelisable, fault tolerant and partially robust to operating on manifolds other than those used during training. A strong parallel exists between training NCA and system identification of a partial differential equation (PDE) satisfying certain boundary value conditions. In the original work by Mordvintsev et al. (2020), asynchronicity in cell updates is justified by a desire to have purely local communication between cells. We demonstrate that asynchronicity is not just an ideological feature of the model and is in fact necessary to learn a well-behaved PDE and to allow the model to be used in arbitrary integrators.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2108.04328" target="_blank">
    <img src="assets/thumbnails/2021-07-19generative_otte.jpg" width="140px">
</a>
</td>
<td>

#### Generative Adversarial Neural Cellular Automata
Published on **2021-07-19** by

Maximilian **Otte**, Quentin **Delfosse**, Johannes **Czech**, Kristian **Kersting**

[Arxiv](https://arxiv.org/abs/2108.04328)

<details>
<summary><b>Abstract</b></summary>
Motivated by the interaction between cells, the recently introduced concept of Neural Cellular Automata shows promising results in a variety of tasks. So far, this concept was mostly used to generate images for a single scenario. As each scenario requires a new model, this type of generation seems contradictory to the adaptability of cells in nature. To address this contradiction, we introduce a concept using different initial environments as input while using a single Neural Cellular Automata to produce several outputs. Additionally, we introduce GANCA, a novel algorithm that combines Neural Cellular Automata with Generative Adversarial Networks, allowing for more generalization through adversarial training. The experiments show that a single model is capable of learning several images when presented with different inputs, and that the adversarially trained model improves drastically on out-of-distribution data compared to a supervised trained model.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2106.15240" target="_blank">
    <img src="assets/thumbnails/2021-07-12towardssel_variengien.jpg" width="140px">
</a>
</td>
<td>

#### Towards self-organized control: Using neural cellular automata to robustly control a cart-pole agent
Published on **2021-07-12** by

Alexandre **Variengien**, Stefano **Nichele**, Tom **Glover**, Sidney **Pontes-Filho**

[Arxiv](https://arxiv.org/abs/2106.15240) | [Project Page](https://alexandrevariengien.com/self-organized-control/) | [Code](https://github.com/aVariengien/self-organized-control)

<details>
<summary><b>Abstract</b></summary>
Neural cellular automata (Neural CA) are a recent framework used to model biological phenomena emerging from multicellular organisms. In these systems, artiﬁcial neural networks are used as update rules for cellular automata. Neural CA are end-to-end differentiable systems where the parameters of the neural network can be learned to achieve a particular task. In this work, we used neural CA to control a cart-pole agent. The observations of the environment are transmitted in input cells while the values of output cells are used as a readout of the system. We trained the model using deep-Q learning where the states of the output cells were used as the q-value estimates to be optimized. We found that the computing abilities of the cellular automata were maintained over several hundreds of thousands of iterations, producing an emergent stable behavior in the environment it controls for thousands of steps. Moreover, the system demonstrated life-like phenomena such as a developmental phase, regeneration after damage, stability despite a noisy environment, and robustness to unseen disruption such as input deletion.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2107.06862" target="_blank">
    <img src="assets/thumbnails/2021-06-22differenti_mordvintsev.jpg" width="140px">
</a>
</td>
<td>

#### Differentiable Programming of Reaction-Diffusion Patterns
Published on **2021-06-22** by

Alexander **Mordvintsev**, Ettore **Randazzo**, Eyvind **Niklasson**

[Paper](https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00429/102965/Differentiable-Programming-of-Reaction-Diffusion) | [Arxiv](https://arxiv.org/abs/2107.06862)

<details>
<summary><b>Abstract</b></summary>
Reaction-Diffusion (RD) systems provide a computational framework that governs many pattern formation processes in nature. Current RD system design practices boil down to trial-and-error parameter search. We propose a differentiable optimization method for learning the RD system parameters to perform example-based texture synthesis on a 2D plane. We do this by representing the RD system as a variant of Neural Cellular Automata and using task-speciﬁc differentiable loss functions. RD systems generated by our method exhibit robust, non-trivial “life-like” behavior.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2006.12155" target="_blank">
    <img src="assets/thumbnails/2021-06-12neuralcell_ruiz.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Automata Manifold
Published on **2021-06-12** by

Alejandro Hernandez **Ruiz**, Armand **Vilalta**, Francesc **Moreno-Noguer**

[Paper](https://ieeexplore.ieee.org/document/9578885) | [Arxiv](https://arxiv.org/abs/2006.12155) | [Cvpr](https://openaccess.thecvf.com/content/CVPR2021/html/Hernandez_Neural_Cellular_Automata_Manifold_CVPR_2021_paper.html)

<details>
<summary><b>Abstract</b></summary>
Very recently, the Neural Cellular Automata (NCA) has been proposed to simulate the morphogenesis process with deep networks. NCA learns to grow an image starting from a fixed single pixel. In this work, we show that the neural network (NN) architecture of the NCA can be encapsulated in a larger NN. This allows us to propose a new model that encodes a manifold of NCA, each of them capable of generating a distinct image. Therefore, we are effectively learning an embedding space of CA, which shows generalization capabilities. We accomplish this by introducing dynamic convolutions inside an Auto-Encoder architecture, for the first time used to join two different sources of information, the encoding and cell’s environment information. In biological terms, our approach would play the role of the transcription factors, modulating the mapping of genes into specific proteins that drive cellular differentiation, which occurs right before the morphogenesis. We thoroughly evaluate our approach in a dataset of synthetic emojis and also in real images of CIFAR-10. Our model introduces a general-purpose network, which can be used in a broad range of problems beyond image generation.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2105.07299" target="_blank">
    <img src="assets/thumbnails/2021-05-15texturegen_mordvintsev.jpg" width="140px">
</a>
</td>
<td>

#### Texture Generation with Neural Cellular Automata
Published on **2021-05-15** by

Alexander **Mordvintsev**, Eyvind **Niklasson**, Ettore **Randazzo**

[Arxiv](https://arxiv.org/abs/2105.07299)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCA) have shown a remarkable ability to learn the required rules to "grow" images, classify morphologies, segment images, as well as to do general computation such as path-finding. We believe the inductive prior they introduce lends itself to the generation of textures. Textures in the natural world are often generated by variants of locally interacting reaction-diffusion systems. Human-made textures are likewise often generated in a local manner (textile weaving, for instance) or using rules with local dependencies (regular grids or geometric patterns). We demonstrate learning a texture generator from a single template image, with the generation method being embarrassingly parallel, exhibiting quick convergence and high fidelity of output, and requiring only some minimal assumptions around the underlying state manifold. Furthermore, we investigate properties of the learned models that are both useful and interesting, such as non-stationary dynamics and an inherent robustness to damage. Finally, we make qualitative claims that the behaviour exhibited by the NCA model is a learned, distributed, local algorithm to generate a texture, setting our method apart from existing work on texture generation. We discuss the advantages of such a paradigm.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://distill.pub/selforg/2021/adversarial/" target="_blank">
    <img src="assets/thumbnails/2021-05-06adversaria_randazzo.jpg" width="140px">
</a>
</td>
<td>

#### Adversarial Reprogramming of Neural Cellular Automata
Published on **2021-05-06** by

Ettore **Randazzo**, Alexander **Mordvintsev**, Eyvind **Niklasson**, Michael **Levin**

[Project Page](https://distill.pub/selforg/2021/adversarial/)

<details>
<summary><b>Abstract</b></summary>
Reprogramming Neural CA to exhibit novel behaviour, using adversarial attacks.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://kvfrans.com/stampca-conditional-neural-cellular-automat" target="_blank">
    <img src="assets/thumbnails/2025-02-03neuralcell_dacre.jpg" width="140px">
</a>
</td>
<td>

#### StampCA: Growing Emoji with Conditional Neural Cellular Automata
Published on **2021-04-03** by

Kevin **Frans**

[Project Page](https://kvfrans.com/stampca-conditional-neural-cellular-automat) | [Code](https://colab.research.google.com/drive/1FBEuRymdpgQiDPl5aLPrMDPVIZUM5xpg#scrollTo=8_qZe_c1uPHf)

<details>
<summary><b>Abstract</b></summary>
Neural CAs define local interactions which together grow into a global design. Instead of one system for one design, we can define a general system which grows many designs. This lets us condition our neural CA by giving it different design-specific "seeds". StampCA models encode design-specific information in the cell state, and generic information in the network parameters. This means we can 1. grow many designs without retraining, and 2. grow all these designs in the same world.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2103.08737" target="_blank">
    <img src="assets/thumbnails/2021-03-15growing3da_sudhakaran.jpg" width="140px">
</a>
</td>
<td>

#### Growing 3D Artefacts and Functional Machines with Neural Cellular Automata
Published on **2021-03-15** by

Shyam **Sudhakaran**, Djordje **Grbic**, Siyan **Li**, Adam **Katona**, Elias **Najarro**, Claire **Glanois**, Sebastian **Risi**

[Paper](https://direct.mit.edu/isal/proceedings/isal2021/33/108/102980) | [Arxiv](https://arxiv.org/abs/2103.08737) | [Code](https://github.com/real-itu/3d-artefacts-nca) | [Video](https://www.youtube.com/watch?v=-EzztzKoPeo)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata (NCAs) have been proven effective in simulating morphogenetic processes, the continuous construction of complex structures from very few starting cells. Recent developments in NCAs lie in the 2D domain, namely reconstructing target images from a single pixel or inﬁnitely growing 2D textures. In this work, we propose an extension of NCAs to 3D, utilizing 3D convolutions in the proposed neural network architecture. Minecraft is selected as the environment for our automaton since it allows the generation of both static structures and moving machines. We show that despite their simplicity, NCAs are capable of growing complex entities such as castles, apartment blocks, and trees, some of which are composed of over 3,000 blocks. Additionally, when trained for regeneration, the system is able to regrow parts of simple functional machines, signiﬁcantly expanding the capabilities of simulated morphogenetic systems. The code for the experiment in this paper can be found at: https://github.com/real-itu/ 3d-artefacts-nca.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://distill.pub/selforg/2021/textures/" target="_blank">
    <img src="assets/thumbnails/2021-02-11selforgan_niklasson.jpg" width="140px">
</a>
</td>
<td>

#### Self-Organising Textures
Published on **2021-02-11** by

Eyvind **Niklasson**, Alexander **Mordvintsev**, Ettore **Randazzo**, Michael **Levin**

[Project Page](https://distill.pub/selforg/2021/textures/) | [Code Pytorch](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_pytorch.ipynb) | [Code Tensorflow](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_tf2.ipynb)

<details>
<summary><b>Abstract</b></summary>
Neural Cellular Automata learn to generate textures, exhibiting surprising properties.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2102.02579" target="_blank">
    <img src="assets/thumbnails/2021-02-04regenerati_horibe.jpg" width="140px">
</a>
</td>
<td>

#### Regenerating Soft Robots Through Neural Cellular Automata
Published on **2021-02-04** by

Kazuya **Horibe**, Kathryn **Walker**, Sebastian **Risi**

[Paper](https://link.springer.com/chapter/10.1007/978-3-030-72812-0_3) | [Arxiv](https://arxiv.org/abs/2102.02579) | [Code](https://github.com/KazuyaHoribe/RegeneratingSoftRobots)

<details>
<summary><b>Abstract</b></summary>
Morphological regeneration is an important feature that highlights the environmental adaptive capacity of biological systems. Lack of this regenerative capacity signiﬁcantly limits the resilience of machines and the environments they can operate in. To aid in addressing this gap, we develop an approach for simulated soft robots to regrow parts of their morphology when being damaged. Although numerical simulations using soft robots have played an important role in their design, evolving soft robots with regenerative capabilities have so far received comparable little attention. Here we propose a model for soft robots that regenerate through a neural cellular automata. Importantly, this approach only relies on local cell information to regrow damaged components, opening interesting possibilities for physical regenerable soft robots in the future. Our approach allows simulated soft robots that are damaged to partially regenerate their original morphology through local cell interactions alone and regain some of their ability to locomote. These results take a step towards equipping artiﬁcial systems with regenerative capacities and could potentially allow for more robust operations in a variety of situations and environments. The code for the experiments in this paper is available at: http://github.com/KazuyaHoribe/ RegeneratingSoftRobots.
</details>

</td>
</tr>
</table>


### 2020
<table><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2010.04949" target="_blank">
    <img src="assets/thumbnails/2020-11-06imagegener_chen.jpg" width="140px">
</a>
</td>
<td>

#### Image Generation With Neural Cellular Automatas
Published on **2020-11-06** by

Mingxiang **Chen**, Zhecheng **Wang**

[Arxiv](https://arxiv.org/abs/2010.04949) | [Code](https://github.com/chenmingxiang110/VAE-NCA)

<details>
<summary><b>Abstract</b></summary>
In this paper, we propose a novel approach to generate images (or other artworks) by using neural cellular automatas (NCAs). Rather than training NCAs based on single images one by one, we combined the idea with variational autoencoders (VAEs), and hence explored some applications, such as image restoration and style fusion. The code for model implementation is available online.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2103.04130" target="_blank">
    <img src="assets/thumbnails/2020-10-02learningto_zhang.jpg" width="140px">
</a>
</td>
<td>

#### Learning to Generate 3D Shapes with Generative Cellular Automata
Published on **2020-10-02** by

Dongsu **Zhang**, Changwoon **Choi**, Jeonghwan **Kim**, Young Min **Kim**

[Arxiv](https://arxiv.org/abs/2103.04130) | [Code](https://github.com/96lives/gca)

<details>
<summary><b>Abstract</b></summary>
In this work, we present a probabilistic 3D generative model, named Generative Cellular Automata, which is able to produce diverse and high quality shapes. We formulate the shape generation process as sampling from the transition kernel of a Markov chain, where the sampling chain eventually evolves to the full shape of the learned distribution. The transition kernel employs the local update rules of cellular automata, effectively reducing the search space in a high-resolution 3D grid space by exploiting the connectivity and sparsity of 3D shapes. Our progressive generation only focuses on the sparse set of occupied voxels and their neighborhood, thus enables the utilization of an expressive sparse convolutional network. We propose an effective training scheme to obtain the local homogeneous rule of generative cellular automata with sequences that are slightly different from the sampling chain but converge to the full shapes in the training data. Extensive experiments on probabilistic shape completion and shape generation demonstrate that our method achieves competitive performance against recent methods.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://arxiv.org/pdf/2009.09347" target="_blank">
    <img src="assets/thumbnails/2020-09-19predicting_chen.jpg" width="140px">
</a>
</td>
<td>

#### Predicting Geographic Information with Neural Cellular Automata
Published on **2020-09-19** by

Mingxiang **Chen**, Qichang **Chen**, Lei **Gao**, Yilin **Chen**, Zhecheng **Wang**

[Arxiv](https://arxiv.org/abs/2009.09347) | [Code](https://github.com/chenmingxiang110/NCA_Prediction)

<details>
<summary><b>Abstract</b></summary>
This paper presents a novel framework using neural cellular automata (NCA) to regenerate and predict geographic information. The model extends the idea of using NCA to generate/regenerate a speciﬁc image by training the model with various geographic data, and thus, taking the trafﬁc condition map as an example, the model is able to predict trafﬁc conditions by giving certain induction information. Our research veriﬁed the analogy between NCA and gene in biology, while the innovation of the model signiﬁcantly widens the boundary of possible applications based on NCAs. From our experimental results, the model shows great potentials in its usability and versatility which are not available in previous studies. The code for model implementation is available at https://redacted.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="" target="_blank">
    <img src="assets/thumbnails/2020-08-27selfclass_randazzo.jpg" width="140px">
</a>
</td>
<td>

#### Self-classifying MNIST Digits
Published on **2020-08-27** by

Ettore **Randazzo**, Alexander **Mordvintsev**, Eyvind **Niklasson**, Michael **Levin**, Sam **Greydanus**



<details>
<summary><b>Abstract</b></summary>
Training an end-to-end differentiable, self-organising cellular automata for classifying MNIST digits.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/mnist_ca.ipynb" target="_blank">
    <img src="assets/thumbnails/2020-08-12imagesegme_sandler.jpg" width="140px">
</a>
</td>
<td>

#### Image segmentation via Cellular Automata
Published on **2020-08-12** by

Mark **Sandler**, Andrey **Zhmoginov**, Liangcheng **Luo**, Alexander **Mordvintsev**, Ettore **Randazzo**, Blaise Agúera y **Arcas**

[Code](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/mnist_ca.ipynb) | [Project Page](https://distill.pub/2020/selforg/mnist/)

<details>
<summary><b>Abstract</b></summary>
In this paper, we propose a new approach for building cellular automata to solve real-world segmentation problems. We design and train a cellular automaton that can successfully segment high-resolution images. We consider a colony that densely inhabits the pixel grid, and all cells are governed by a randomized update that uses the current state, the color, and the state of the 3×3 neighborhood. The space of possible rules is deﬁned by a small neural network. The update rule is applied repeatedly in parallel to a large random subset of cells and after convergence is used to produce segmentation masks that are then back-propagated to learn the optimal update rules using standard gradient descent methods. We demonstrate that such models can be learned eﬃciently with only limited trajectory length and that they show remarkable ability to organize the information to produce a globally consistent segmentation result, using only local information exchange.
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://umu1729.github.io/pages-neural-cellular-maze-solver/" target="_blank">
    <img src="assets/thumbnails/2020-03-12neuralcell_endo.jpg" width="140px">
</a>
</td>
<td>

#### Neural Cellular Maze Solver
Published on **2020-03-12** by

Katsuhiro **Endo**, Kenji **Yasuoka**

[Project Page](https://umu1729.github.io/pages-neural-cellular-maze-solver/)

<details>
<summary><b>Abstract</b></summary>
Solving mazes with Neural Cellular Automata
</details>

</td>
</tr><tr>
<td width="150px">
<a href="https://distill.pub/2020/growing-ca/" target="_blank">
    <img src="assets/thumbnails/2020-02-11growingneu_mordvintsev.jpg" width="140px">
</a>
</td>
<td>

#### Growing Neural Cellular Automata
Published on **2020-02-11** by

Alexander **Mordvintsev**, Ettore **Randazzo**, Eyvind **Niklasson**, Michael **Levin**

[Project Page](https://distill.pub/2020/growing-ca/) | [Code](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb)

<details>
<summary><b>Abstract</b></summary>
Training an end-to-end differentiable, self-organising cellular automata model of morphogenesis, able to both grow and regenerate specific patterns.
</details>

</td>
</tr></table>


## Further Reading
- **Videos & Tutorials**
  - [Mildly Overfitted](https://www.youtube.com/watch?v=21ACbWoF2Oo) – Tutorial & code explanation (PyTorch)
  - [Yannic Kilcher](https://www.youtube.com/watch?v=9Kec_7WFyp0) – Paper explanation

- **Courses**
  - [Artificial Life by Vassilis Papadopoulos](https://vassi.life/teaching/alife)
    - [Lecture 10: Neural Cellular Automata](https://frotaur.notion.site/Course-10-Neural-Cellular-automata-63d6eb2efe9443b4b2c3a09a55f493a0) ([Video Recording](https://www.youtube.com/watch?v=_ealiM25biA))
  - [AIAIArt Course by John Whitaker](https://github.com/johnowhitaker/aiaiart)
    - [Lesson #8: Neural CA](https://colab.research.google.com/drive/1Qpx_4wWXoiwTRTCAP1ohpoPGwDIrp9z-) ([Video Recording](https://www.youtube.com/watch?v=X2-ucB74oEk))
    - [Full YouTube Playlist](https://www.youtube.com/playlist?list=PL23FjyM69j910zCdDFVWcjSIKHbSB7NE8)

- **Projects & Tools**
  - [NeuralCA.org](https://www.neuralca.org/) | [GitHub](https://github.com/MonashDeepNeuron/Neural-Cellular-Automata)
  - [Google Self-organising Systems](https://github.com/google-research/self-organising-systems/)

- **Others**
  - Alexander Mordvintsev: [Website](https://znah.net/) | [YouTube](https://www.youtube.com/@zzznah) | [Twitter/X](https://x.com/zzznah) | [GitHub](https://github.com/znah)
  - [International Society for Artificial LIFE (ISAL)](https://alife.org/)
  - [Awesome Cellular Automata](https://github.com/vovanmozg/awesome-cellular-automata) – curated list of CA resources
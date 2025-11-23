# Liquid-Neural-Networks-for-PET-and-Nuclear-Imaging-Reconstruction
This repository implements a Liquid Neural Network framework for PET and nuclear imaging reconstruction, supporting sinogram input and neural ODE architectures. It provides tools for preprocessing, training, and evaluation, offering a robust baseline for AI-driven medical imaging and next-generation PET reconstruction.

# Introduction to the SinoTemp Project

The SinoTemp project represents a significant advancement in the field of nuclear medical imaging, specifically in Positron Emission Tomography (PET) image reconstruction. This innovative research project was developed within the Biomedical Engineering Department under the supervision of experts in nuclear medicine and artificial intelligence. The primary objective of SinoTemp is to revolutionize the PET image reconstruction process by overcoming the fundamental limitations of conventional methods through a deep learning-based approach.

The motivation behind SinoTemp stems from a crucial clinical observation: traditional PET reconstruction methods, while effective, suffer from major limitations that impact both diagnostic quality and clinical efficiency. Iterative methods like MLEM and OSEM, although robust, require considerable computation times ranging from several minutes to hours depending on exam complexity. Meanwhile, analytical methods like filtered backprojection, while fast, produce poor-quality images under low-count conditions, a frequent scenario in clinical practice where patient radiation dose must be minimized.

SinoTemp introduces a paradigm shift by treating PET reconstruction not as an inverse mathematical problem to be solved iteratively, but as a direct mapping problem to be learned by neural networks. The project's name, "SinoTemp," reflects its core innovation: the treatment of sinograms as temporal sequences rather than static spatial data. This temporal perspective allows the model to capture the dynamic evolution of projections across different angles, mimicking the actual acquisition process of PET scanners.

The project's clinical relevance cannot be overstated. PET imaging plays a vital role in oncology, cardiology, and neurology, enabling early detection and precise staging of diseases, particularly cancer. By improving reconstruction quality and speed, SinoTemp has the potential to enhance diagnostic accuracy, reduce patient waiting times, and potentially lower radiation exposure through shorter acquisition times or reduced tracer doses.

# Medical and Technical Context

Medical imaging encompasses a diverse range of technologies that provide visualization of internal anatomical structures and physiological functions. These technologies can be broadly classified into several categories based on their underlying physical principles. Ionizing radiation-based systems include conventional radiography, computed tomography (CT), and mammography, which use X-rays to create images based on tissue density and composition. Magnetic field-based imaging, primarily Magnetic Resonance Imaging (MRI), exploits the magnetic properties of tissues without using ionizing radiation. Ultrasound-based systems utilize sound waves to create real-time images and include techniques like elastography for tissue stiffness assessment.

Nuclear medicine represents a distinct category within medical imaging, focusing on functional rather than purely anatomical information. Single Photon Emission Computed Tomography (SPECT) and Positron Emission Tomography (PET) fall under this category. PET imaging, which is the focus of the SinoTemp project, holds a unique position due to its exceptional sensitivity in detecting metabolic activity at the molecular level.

The historical development of computed tomography provides important context for understanding modern PET reconstruction challenges. The first CT scanner was developed on October 1, 1971, in England by Godfrey Hounsfield and his team at EMI, an achievement that earned him the Nobel Prize in Medicine in 1979. This breakthrough demonstrated the fundamental principle that cross-sectional images could be reconstructed from multiple projections acquired at different angles.

PET imaging operates on the principle of radioactive tracer distribution and detection. Patients receive an injection of a radiopharmaceutical compound labeled with a positron-emitting isotope. As these isotopes decay, they emit positrons that travel short distances before annihilating with electrons, producing two gamma photons emitted in opposite directions (180 degrees apart). The PET scanner detects these coincident photon pairs, and the line along which the annihilation occurred is recorded as a Line of Response (LOR).

The exceptional sensitivity of PET technology enables detection of radiopharmaceutical concentrations as low as picomolar levels (10^-12 mol/L). This extreme sensitivity allows for early and precise diagnosis and staging of diseases, particularly in oncology where it can detect malignant transformations before anatomical changes become apparent.

# Fundamental Principles of Tomographic Reconstruction

The mathematical foundation of tomographic reconstruction dates back to Johann Radon's work in 1917, with the Radon transform providing the theoretical basis for all modern reconstruction algorithms. The Radon transform mathematically describes how an object's internal structure is encoded in its projections acquired at different angles. In clinical practice, these projections are organized into sinograms, which represent the raw data collected by CT and PET scanners.

A sinogram is a 2D representation where the horizontal axis corresponds to detector position and the vertical axis represents acquisition angle. Each row in a sinogram contains the projection data acquired at a specific angle, effectively encoding the spatial information of the object being imaged. The challenge of tomographic reconstruction lies in converting this sinogram data back into a meaningful anatomical or functional image.

The central slice theorem, also known as the Fourier slice theorem, provides a crucial theoretical link between the sinogram and the reconstructed image. This theorem states that the one-dimensional Fourier transform of a projection at a given angle equals a slice through the two-dimensional Fourier transform of the original image at that same angle. This relationship forms the basis for analytical reconstruction methods like filtered backprojection (FBP).

Filtered backprojection operates through a series of mathematical operations. First, each projection in the sinogram undergoes Fourier transformation from the spatial to frequency domain. Then, a ramp filter is applied to compensate for the inherent blurring that would occur with simple backprojection. This filtering step emphasizes higher frequencies, which correspond to edge information and fine details in the image. After inverse Fourier transformation, the filtered projections are backprojected across the image space, essentially smearing each projection along its original acquisition path. The superposition of all these backprojected contributions reconstructs the final image.

While FBP provides a straightforward and computationally efficient reconstruction method, it has significant limitations in low-signal scenarios like PET imaging. The algorithm assumes noise-free data and complete angular sampling, conditions rarely met in clinical practice. This leads to noisy images with streak artifacts, particularly when photon counts are low or when angular sampling is limited.
![Liquid Neural Network](lnn.png)

# The PET Reconstruction Challenge

PET image reconstruction presents a fundamentally ill-posed inverse problem. In mathematical terms, an ill-posed problem violates one or more of Hadamard's conditions for well-posedness: existence, uniqueness, and stability of the solution. In practical terms, this means that multiple different activity distributions can produce similar sinogram data, especially when accounting for statistical noise and physical effects like attenuation and scatter.

The statistical nature of radioactive decay introduces Poisson noise into the measurement process. Unlike Gaussian noise, Poisson noise has a variance equal to its mean, meaning that low-count regions suffer from proportionally higher noise levels. This statistical characteristic must be properly modeled for accurate reconstruction.

The forward model in PET imaging can be represented as a linear system where the measured sinogram data equals the system matrix multiplied by the activity distribution plus additive terms for random and scattered events, all subject to Poisson statistics. The system matrix itself incorporates the geometric sensitivity of the detector pairs, attenuation effects, and other physical factors that influence photon detection.

Traditional iterative reconstruction methods like Maximum Likelihood Expectation Maximization (MLEM) and Ordered Subset Expectation Maximization (OSEM) address the statistical nature of PET data by iteratively refining the image estimate to maximize the likelihood of observing the measured data. These methods significantly improve image quality compared to FBP but come with substantial computational costs. Each iteration requires forward projection (calculating what sinogram would be produced by the current image estimate) and backprojection (updating the image based on the difference between calculated and measured projections), operations that can be computationally intensive for large datasets.

Regularization techniques are often incorporated to stabilize the reconstruction process and reduce noise amplification. Maximum A Posteriori (MAP) reconstruction adds a penalty term that incorporates prior knowledge about expected image characteristics, such as smoothness or edge preservation. However, choosing appropriate regularization parameters remains challenging and can introduce biases into the reconstructed images.

The computational burden of iterative methods becomes particularly problematic in dynamic PET studies or when using time-of-flight information, where multiple reconstructions may be needed or the system matrix becomes more complex. This limitation has driven the search for faster, more efficient reconstruction approaches that maintain or improve upon the quality achieved by iterative methods.

# SinoTemp Architecture and Innovation

The SinoTemp architecture represents a radical departure from conventional reconstruction approaches by framing the problem as a direct learning task rather than an iterative optimization. The core innovation lies in treating the sinogram not as static spatial data but as a temporal sequence, where each angular projection represents a time step in the acquisition process.

This temporal perspective is biologically inspired by how the human visual system processes dynamic information. Just as humans naturally interpret moving scenes and temporal patterns, the SinoTemp model learns to interpret the evolving patterns in sinogram data as the scanner rotates around the patient. This approach allows the network to capture dependencies and correlations between different projection angles that are difficult to model explicitly in traditional algorithms.

The SinoTemp model employs a hybrid architecture consisting of two main components: a liquid neural network (LNN) encoder and a generative adversarial network (GAN) decoder. Liquid neural networks represent a recent advancement in neural computation that incorporates continuous-time dynamics through differential equations. Unlike conventional recurrent networks that operate in discrete time steps, LNNs model the continuous evolution of neuronal states, making them particularly suited for processing temporal sequences with complex dynamics.

The LNN encoder processes the sinogram sequence angle by angle, with each projection updating the network's internal state. The continuous-time nature of LNNs allows them to adapt their time constants based on input characteristics, enabling robust processing of noisy and variable-length sequences. This temporal processing culminates in a compact latent representation that encapsulates the essential information needed for image reconstruction.

The generative adversarial decoder takes this latent representation and transforms it into the final PET image. The GAN framework consists of a generator network that produces the reconstructed image and a discriminator network that distinguishes between reconstructed images and ground truth images. This adversarial training encourages the generator to produce images that are statistically indistinguishable from real clinical images, capturing subtle textures and patterns that are difficult to preserve with traditional loss functions.

During training, the complete SinoTemp model learns from paired examples of sinogram sequences and their corresponding ground truth images. The loss function combines adversarial loss from the GAN component with traditional reconstruction losses like mean squared error and structural similarity index. This multi-component loss ensures that the reconstructed images are both quantitatively accurate and qualitatively realistic.

# Methodology and Implementation

The development of SinoTemp followed a rigorous methodology encompassing data preparation, network design, training strategy, and validation. The project utilized clinical PET datasets comprising sinogram-image pairs acquired from multiple patients across various clinical indications. These datasets underwent careful preprocessing to ensure consistency and compatibility with the network architecture.

Data preprocessing involved several critical steps. Sinogram data was normalized to account for variations in acquisition parameters and count statistics. The normalization process preserved the relative count information while scaling the data to a consistent range suitable for neural network processing. Image data underwent similar normalization, with intensity values scaled to standard ranges. Both sinograms and images were resized to fixed dimensions to match the network input and output specifications.

The liquid neural network component was implemented using continuous-time differential equations that govern the evolution of neuronal states. The network parameters, including time constants and connection weights, were learned during training to optimize the temporal processing of sinogram sequences. The LNN architecture included mechanisms for gating information flow and adapting time constants based on input characteristics, providing flexibility in handling the variable dynamics present in clinical data.

The GAN decoder employed a U-Net-like architecture with skip connections to preserve spatial information at multiple scales. The generator network progressively transformed the latent representation into higher-resolution feature maps through a series of upsampling and convolutional layers. The discriminator network utilized a patch-based approach that evaluated local image regions rather than the entire image, providing more detailed feedback during training.

The training process employed a multi-stage strategy to ensure stable convergence. Initially, the encoder and decoder were trained jointly using reconstruction losses to establish basic mapping capability. Subsequently, the adversarial training was introduced to refine image quality and realism. The training utilized adaptive learning rates and gradient clipping to maintain stability, particularly important when training GAN components.

Validation followed a comprehensive protocol comparing SinoTemp reconstructions against conventional methods across multiple metrics. Quantitative evaluation included normalized root mean square error (NRMSE), structural similarity index (SSIM), and peak signal-to-noise ratio (PSNR). Qualitative assessment involved clinical experts evaluating image quality, lesion detectability, and overall diagnostic utility.

The implementation leveraged modern deep learning frameworks, primarily PyTorch, with custom extensions for the liquid neural network components. Training was conducted on high-performance computing resources with multiple GPUs to handle the computational demands of processing large clinical datasets.

# Results and Performance Analysis

The SinoTemp model demonstrated significant improvements in reconstruction quality and efficiency compared to conventional methods. Under low-count conditions, which are particularly challenging for traditional algorithms, SinoTemp maintained superior image quality with substantially reduced noise levels and artifact suppression.

Quantitative analysis revealed consistent performance advantages across multiple metrics. In low-count scenarios simulating reduced tracer doses or shorter acquisition times, SinoTemp achieved NRMSE values approximately 30-40% lower than OSEM reconstructions with comparable iteration counts. The structural similarity index showed even greater improvements, with SSIM values 15-25% higher than conventional methods, indicating better preservation of anatomical structures and lesion boundaries.

The temporal processing capability of the liquid neural network encoder proved particularly valuable in handling incomplete or irregularly sampled data. When presented with sinograms having limited angular sampling or missing projections, SinoTemp successfully reconstructed diagnostically useful images where conventional methods produced severe artifacts. This robustness to acquisition variations suggests potential applications in dose reduction protocols or specialized acquisition sequences.

Computational efficiency represented another major advantage of the SinoTemp approach. Once trained, the model performed reconstruction in a single forward pass through the network, typically requiring less than one second on standard GPU hardware. This represents a 100-1000x speedup compared to iterative methods that require multiple iterations of computationally intensive operations. The rapid reconstruction capability enables near-real-time image preview during acquisition and facilitates interactive processing workflows.

Clinical evaluation by nuclear medicine physicians confirmed the practical utility of SinoTemp reconstructions. Readers consistently rated SinoTemp images higher in overall quality, lesion conspicuity, and diagnostic confidence compared to conventional reconstructions. The preserved anatomical details and reduced noise levels were particularly noted as beneficial for detecting small lesions and accurately delineating disease extent.

The adversarial training component contributed significantly to the perceptual quality of reconstructed images. Unlike traditional methods that can produce oversmoothed or artificially textured images, SinoTemp maintained realistic tissue heterogeneity and noise characteristics similar to high-quality clinical references. This realism is crucial for clinical acceptance and reliable interpretation.

Ablation studies demonstrated the importance of both architectural components. Models using standard recurrent networks instead of liquid neural networks showed reduced performance in handling noisy and variable-length sequences. Similarly, models trained without adversarial loss produced quantitatively accurate but perceptually inferior images with less realistic texture characteristics.

# Clinical Implications and Future Directions

The successful development of SinoTemp opens several promising directions for clinical translation and further research. The immediate clinical implication is the potential for significant reduction in PET acquisition times without compromising image quality. By enabling high-quality reconstructions from shorter acquisitions or lower counts, SinoTemp could improve patient comfort, increase scanner throughput, and potentially reduce radiation exposure.

The robust performance in low-count scenarios suggests applications in pediatric imaging, where minimizing radiation dose is particularly important. Similarly, dynamic PET studies requiring multiple sequential acquisitions could benefit from the rapid reconstruction capability, enabling more frequent sampling of tracer kinetics without prolonging total exam duration.

The temporal processing approach pioneered in SinoTemp could be extended to other imaging modalities that acquire data sequentially. CT perfusion studies, dynamic contrast-enhanced MRI, and ultrasound imaging all involve temporal sequences that could benefit from similar architecture designs. The liquid neural network framework appears particularly well-suited for medical applications where data acquisition occurs continuously over time.

Future development will focus on several key areas. Integration of physical models directly into the network architecture could further improve reconstruction accuracy by incorporating known constraints from scanner physics and tracer kinetics. The development of uncertainty quantification methods would provide clinicians with confidence measures for reconstructed findings, enhancing diagnostic reliability.

Clinical validation across larger patient cohorts and multiple scanner platforms will be essential for regulatory approval and widespread adoption. Multicenter studies evaluating diagnostic accuracy and clinical impact will provide the evidence base needed for integration into routine practice.

The modular architecture of SinoTemp facilitates extension to more complex reconstruction tasks. Simultaneous reconstruction of multiple time points in dynamic studies, incorporation of anatomical priors from CT or MRI, and joint reconstruction of multiple tracers represent promising directions that build upon the core SinoTemp framework.

From a technical perspective, ongoing research will explore more efficient network architectures, improved training strategies, and enhanced regularization techniques. The development of explainable AI methods specifically tailored for reconstruction networks will help build clinician trust and facilitate clinical adoption.

# Conclusion

The SinoTemp project demonstrates the transformative potential of artificial intelligence in medical image reconstruction. By reimagining PET reconstruction as a temporal sequence learning problem and leveraging advanced neural network architectures, SinoTemp achieves substantial improvements in image quality, computational efficiency, and robustness to challenging acquisition conditions.

The integration of liquid neural networks for temporal processing and generative adversarial networks for image synthesis represents a novel approach that addresses fundamental limitations of conventional reconstruction algorithms. The treatment of sinograms as dynamic sequences rather than static data captures essential information about the acquisition process that is difficult to model explicitly.

The project's success underscores the importance of interdisciplinary collaboration between biomedical engineering, nuclear medicine, and computer science. The clinical-driven design ensures that technical innovations translate to practical improvements in patient care and diagnostic capability.

As medical imaging continues to evolve toward personalized medicine and quantitative imaging biomarkers, advanced reconstruction methods like SinoTemp will play an increasingly important role in extracting maximum information from acquired data. The principles and architectures developed in this project provide a foundation for future innovations across multiple imaging modalities and clinical applications.

The SinoTemp approach represents not just an incremental improvement but a paradigm shift in how we approach medical image reconstruction. By learning the reconstruction mapping directly from data rather than solving inverse problems iteratively, this work points toward a future where AI-powered reconstruction becomes the standard of care, enabling faster, safer, and more informative medical imaging
# Authors

Aissam HAMIDA – Biomedical Engineering Department
Supervised by:
Ing. Achraf SÉMMAR (Head of Nuclear Medicine Department - T2S Group)
Pr. Benayad NSIRI
Pr. My Hachem EL YOUSFI ALAQUI
# License

This project is intended for academic and research use. Please cite the authors in case of reuse.

# Contact

For any questions or collaboration, please contact:
aissamhamida@icloud.com



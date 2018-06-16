layout: true

name: lts5-canvas

<div class="lts5-logo"> <img src="../Common/lts5_logo.svg" style="height: 40px;"></div>

<div class="epfl-logo"> <img src="../Common/epfl_logo.svg" style="height: 40px;"></div>

---

layout: true
name: lts5-question
background-image: url(../Common/question-mark.jpg)
background-position: center
background-size: 300px

<div class="lts5-logo"> <img src="../Common/lts5_logo.svg" style="height: 40px;"></div>

<div class="epfl-logo"> <img src="../Common/epfl_logo.svg" style="height: 40px;"></div>

---

name: title

class: center, middle

template: lts5-canvas

# Progress Talk

Christophe Ecabert

LTS5, EPFL

June 14th, 2018

---

template: lts5-canvas

# Content

- Problem statement
- Unsupervised learning
- Network structure
- Occlusions handling
  - Probabilistic approach
  - Attention Mechanism

???

What's on the agenda

---

template: lts5-canvas

# Problem Statement

- 3D Face reconstruction
  - Identity + Expression
  - Appearance
  - Pose / Projection
  - Illumination (Optional)

<br/>

![:figure 80%, Reconstruction pipeline, 1](figures/problem_statement.svg)

???

- Monocular face reconstruction -> single image
- Model-based -> parameters to estimate

---

template: lts5-canvas

# Unsupervised learning

- Learn the *mapping* function within an *analysis-by-synthesis* framework
- Transform 3D instance into image with rendering stage part of the *model*

<br/>

![:figure 90%, Network architecture, 1](figures/mofa_arch.png)

---

template: lts5-canvas

# Network Structure

- Tewari et al.

![:figure 80%, Image space, 1](figures/struct_01.svg)

- Genova et al.

![:figure 80%, Parameter space, 2](figures/struct_02.svg)

---

template: lts5-canvas

# Occlusion Handling

- Probabilisitic approach

![:figure 80%, Probabilistic approach, 3](figures/occlusion_prob.png)



---

template: lts5-canvas

# Occlusion Handling

- Attention Mechanism
  - What ground truth 
  - Spatial Resolution
  - Training ?

![:figure 60%, Attention mechanism, 4](figures/attention_mechanism.png)

![:figure 30%, BFM Appearance Prior, 3](figures/appearance_prior.png)

---

template: lts5-question

exclude: true

# Questions



---

template: lts5-canvas

# References

.text-small[

[1] MoFA: Model-based Deep Convolutional Face Autoencoder for Unsupervised Monocular Reconstruction, Tewari et al., 2017

[2] Unsupervised Training for 3D Morphable Model Regression, Genova et al., 2018

[3] Occlusion-Aware 3D Morphable Models and an Illumination Prior for Face Image Analysis, Egger et al., 2018

[4] Face Attention Network: An Effective Face Detector for the Occluded Faces, Wang et al., 2018

]


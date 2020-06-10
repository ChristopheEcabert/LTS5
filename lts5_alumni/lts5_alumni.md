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

# Facial Image Analysis

Christophe Ecabert

LTS5, EPFL

January 29th, 2020

---

template: lts5-canvas

# History

- Tracking
  - Anatomic landmarks detection and tracking
- Facial Expression / Action Unit detection
  - Basic facial expression detection
  - Muscle activation detection
- 3D Reconstruction
  - Reconstruct 3d face from a single image
---

template: lts5-canvas

# Tracking

- Locate face in the image
  - Haar + Cascade classifiers
  - Deep learning: ***SFD***
- Predict location of anatomical landmarks
  - Regression from appearance: ***SDM***
  - Deep learning: ***2D-Fan***

<br>

.left-column50[

![:figure 50%, Anatomical landmarks](figures/anatomical_landmarks.png)

]
.right-column50[

![:figure 100%, 2D-Fan alignment](figures/alignment_results.png)

]



---

template: lts5-canvas

# Facial Expression / Action Unit Detection

- Analysis pipeline

![:figure 60%,](figures/expressin_detection_process.svg)

- Detection

![:figure 70%,](figures/base_expression.png)

---

template: lts5-canvas

# 3D Reconstruction

- Inverse Rendering
  - Regenerate the object that create the image
- Explicit modeling
  - Face geometry
  - Face appearance
  - Global Illumination
  - Camera transform

<br>

![:figure 100%,Training Phase](figures/TrainingPhase.svg)

---

template: lts5-canvas

# 3D Reconstruction - Results

.left-column70[

![:figure 115%](figures/rec_samples.png)

]

.right-column30[
![:figure 60%](figures/merged_fast.gif)
]
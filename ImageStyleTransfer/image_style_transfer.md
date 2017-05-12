layout: true

<div class="lts5-logo"> <img src="../Common/lts5_logo.svg" style="height: 40px;"></div>

<div class="epfl-logo"> <img src="../Common/epfl_logo.svg" style="height: 40px;"></div>

---

name: title

class: center, middle

# Image Style Transfer Using Convolutional Network

Christophe Ecabert

LTS5, EPFL

May 18th, 2017 

---

class: center, middle

# Reference

Gatys *et al*. ***Image Style Transfer Using Convolutional Network*** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2016.

---

# Recap

Toto

???

Notes go here !

---

# Figure
<figure>
<img src="figures/arch_net.png" style="width:85%;"/>
<figcaption>AlexNet (8 layers), VGG19 (19 layers), GoogLeNet (22 layers)</figcaption>
</figure>

---
# Latex

- Single layer model 
  `$$ \mathbf{x}_ l = f(\mathbf{y}_{l-1}) $$`
  `$$ \mathbf{y}_l = \mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l $$`

- Single layer with ReLU activation function
  `$$ Var[y_l] = \frac{1}{2} n_l Var[w_l] Var[y_{l-1}] $$`

- With `$L$` layers 
  `$$ Var[y_l] = Var[y_1] \left( \prod_{l=2}^L \frac{1}{2} n_l Var[w_l] \right) $$`

He _et al_. _Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification_ (2015)

---


---
# Conclusions

- Residual architecture
    - Even with very deep structure, it has smaller complexity than plain network (*i.e. VGG*)
    - Features of any layers are additive outcomes 
    - Enables smooth forward/backward propagation
    - Greatly eases the optimization of the model
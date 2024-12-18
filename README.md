# Enhancing Atom Mapping with Multitask Learning and Symmetry-Aware Deep Graph Matching

This repository contains the implementation of SAMMNet, a Symmetry-Aware Multitask Atom Mapping Network. The model combines multitask learning and deep graph matching techniques to enhance the accuracy of atom mapping, particularly in complex and symmetric chemical reactions. 
  

## Overview

Accurate atom mapping is critical for understanding chemical reactions, reaction prediction, and drug design. SAMMNet is a novel framework that combines multitask learning with symmetry-aware graph matching to enhance atom mapping accuracy.

## Features
- Implementation of SAMMNet using Graph Neural Networks (GNNs).
- Support for multitask learning with auxiliary node classification tasks.
- Symmetry-aware refinement using the Weisfeiler-Lehman algorithm.
- Comparison with other training strategies

# model architecture 
![Picture 1](https://github.com/user-attachments/assets/34b6ded1-d9de-48f3-81b6-247d7db81836)





# Example
![Picture 3](https://github.com/user-attachments/assets/76a3f53f-6ef6-445c-bed3-510aa7331471)



# Environment
setup_env.sh: Configuration file for the required environment.
# Models

**Vanilla Model** 
Path: `src/models/vanilla_model.py`

The **Vanilla Model** serves as a shared architecture for both Vanilla training and Transfer Learning approaches. This model is designed specifically to handle atom mapping tasks.

**MTL Model** 
Path: `src/models/mtl_model.py`

The **MTL Model** is a custom architecture developed for multitask learning. It incorporates auxiliary tasks alongside the primary atom mapping objective. This integration enhances molecular representations, enabling the model to achieve improved performance by concurrently learning from multiple related tasks.




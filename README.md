# 🐜🐝 Transfer Learning avec PyTorch — Classification Ants vs Bees

## 📌 Présentation

Ce projet met en œuvre un modèle de **Deep Learning pour la classification d'images** en utilisant **PyTorch** et **Transfer Learning**.

L'objectif est de construire un modèle capable de distinguer automatiquement deux catégories d'images :

- 🐜 **Ants** — fourmis
- 🐝 **Bees** — abeilles

Le projet s'appuie sur le modèle **ResNet18 pré-entraîné sur ImageNet** et compare deux approches de Transfer Learning :

1. **Fine-Tuning** du réseau complet.
2. Utilisation de **ResNet18 comme extracteur de caractéristiques (Feature Extractor)** avec uniquement la dernière couche entraînée.

Le projet a été exécuté sur un **MacBook équipé d'une puce Apple M1**, en utilisant le backend **MPS (Metal Performance Shaders)** pour exploiter le GPU Apple.

---

## 🎯 Objectifs du projet

Ce projet permet de mettre en pratique plusieurs concepts fondamentaux du Deep Learning et de la Computer Vision :

- Utilisation de **PyTorch** et **Torchvision**
- Chargement et préparation d'un dataset d'images
- Data augmentation
- Normalisation des images
- Utilisation d'un modèle pré-entraîné
- Transfer Learning
- Fine-Tuning
- Feature Extraction
- Entraînement et validation d'un réseau neuronal
- Optimisation avec SGD
- Learning Rate Scheduler
- Évaluation de l'accuracy
- Visualisation des prédictions
- Utilisation du GPU Apple M1 avec **MPS**

---

## 🧠 Architecture utilisée

Le modèle utilisé est **ResNet18**, un réseau de neurones convolutif pré-entraîné sur ImageNet.

L'architecture originale de ResNet18 possède une couche finale adaptée à 1000 classes ImageNet.

Dans ce projet, cette couche est remplacée par une couche permettant de prédire uniquement deux classes :

```text
ResNet18 pré-entraîné
        │
        ▼
Extraction des caractéristiques
        │
        ▼
Couche Fully Connected
        │
        ├── Ants
        │
        └── Bees
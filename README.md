# 🐜🐝 Image Classification — Ants vs Bees

Projet de classification d'images utilisant **Python**, **PyTorch**, **Torchvision** et un modèle **ResNet-18 pré-entraîné sur ImageNet**.

L'objectif du projet est de construire un modèle capable de distinguer automatiquement deux catégories d'images :

* 🐜 Ants (fourmis)
* 🐝 Bees (abeilles)

Le projet utilise le **Transfer Learning** afin de tirer parti des connaissances déjà apprises par un modèle ResNet-18.

---

## 📋 Prérequis

Avant de commencer, les éléments suivants doivent être installés :

* Python 3.x
* pip
* Jupyter Notebook ou JupyterLab
* Git (optionnel)

Il est recommandé d'utiliser un **environnement virtuel Python** afin d'isoler les dépendances du projet.

---

## 📁 Structure du projet

```text
MLProject/
│
├── .venv/
│
├── hymenoptera_data/
│   ├── train/
│   │   ├── ants/
│   │   └── bees/
│   │
│   └── val/
│       ├── ants/
│       └── bees/
│
├── requirements.txt
├── README.md
└── classification.ipynb
```

Le dossier `.venv/` contient l'environnement virtuel Python.

Le dossier `hymenoptera_data/` contient les images utilisées pour l'entraînement et la validation.

---

## 🔧 1. Créer l'environnement virtuel

Depuis le répertoire du projet :

```bash
python3 -m venv .venv
```

Cette commande crée un environnement virtuel nommé `.venv`.

---

## ▶️ 2. Activer l'environnement virtuel

### macOS / Linux

```bash
source .venv/bin/activate
```

Après activation, le terminal doit normalement afficher `(.venv)` au début de la ligne.

### Windows

```bash
.venv\Scripts\activate
```

---

## 📦 3. Installer les dépendances

Le projet contient un fichier `requirements.txt`.

Ce fichier contient les bibliothèques Python nécessaires au fonctionnement du projet ainsi que leurs versions.

Une fois l'environnement virtuel activé, exécuter :

```bash
pip install -r requirements.txt
```

### Que signifie `-r` ?

L'option `-r` est l'abréviation de `--requirement`.

Elle indique à `pip` de lire les packages à installer depuis le fichier `requirements.txt`.

La commande :

```bash
pip install -r requirements.txt
```

signifie donc :

> Installer toutes les dépendances listées dans `requirements.txt`.

---

## 📚 4. Principales bibliothèques utilisées

| Bibliothèque | Utilisation                                     |
| ------------ | ----------------------------------------------- |
| PyTorch      | Construction et entraînement du réseau neuronal |
| Torchvision  | Traitement d'images et modèles pré-entraînés    |
| NumPy        | Calcul numérique et manipulation de tableaux    |
| Matplotlib   | Visualisation des données et des résultats      |
| Pillow       | Manipulation des images                         |
| wget         | Téléchargement du dataset si nécessaire         |

Le fichier `requirements.txt` contient également les dépendances indirectes nécessaires au fonctionnement de ces bibliothèques.

---

## 📥 5. Dataset

Le projet utilise le dataset **Hymenoptera**, contenant des images de fourmis et d'abeilles.

Le dataset doit respecter la structure suivante :

```text
hymenoptera_data/
│
├── train/
│   ├── ants/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   └── bees/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
└── val/
    ├── ants/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    └── bees/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

---

## 📍 6. Utiliser un dataset local

Si le dataset est déjà présent sur l'ordinateur, il n'est pas nécessaire de le télécharger à nouveau.

Dans le notebook, définir simplement le chemin du dataset :

```python
data_dir = "./hymenoptera_data"
```

Si le dataset se trouve dans un autre répertoire, utiliser son chemin absolu :

```python
data_dir = "/chemin/vers/hymenoptera_data"
```

Par exemple sur macOS :

```python
data_dir = "/Users/username/Documents/MLProject/hymenoptera_data"
```

---

## 🌐 7. Télécharger le dataset avec wget

Si le dataset n'est pas disponible localement, il peut être téléchargé depuis le serveur PyTorch.

Dans un notebook Jupyter :

```python
import wget

url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

wget.download(url)
```

Le fichier `hymenoptera_data.zip` sera téléchargé dans le répertoire courant.

Il peut ensuite être décompressé avec :

```bash
unzip hymenoptera_data.zip
```

> Si le dataset est déjà présent localement, cette étape n'est pas nécessaire.

---

## 🧠 8. Modèle utilisé : ResNet-18

Le projet utilise **ResNet-18**, un réseau neuronal convolutif pré-entraîné sur ImageNet.

Le modèle peut être chargé avec :

```python
from torchvision import models
import torch.nn as nn

model_ft = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)
```

La dernière couche du modèle est remplacée afin de classifier seulement deux catégories :

```python
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 2)
```

Les deux sorties correspondent à :

```text
Ants
Bees
```

---

## 🔄 9. Transfer Learning

Le projet utilise la technique du **Transfer Learning**.

Au lieu d'entraîner un réseau neuronal entièrement à partir de zéro, nous utilisons un modèle ResNet-18 qui a déjà appris de nombreuses caractéristiques visuelles sur ImageNet.

Le modèle est ensuite adapté à notre problème spécifique.

```text
                    IMAGE
                      │
                      ▼
              ┌─────────────┐
              │  ResNet-18  │
              │ pré-entraîné│
              └─────────────┘
                      │
                      ▼
          Extraction des caractéristiques
                      │
                      ▼
              Couche finale
                      │
                      ▼
             ┌────────────┐
             │ Ants / Bees│
             └────────────┘
```

Cette approche permet généralement d'obtenir de bonnes performances avec un dataset relativement limité.

---

## 🖼️ 10. Transformation des images

Les images sont transformées avant d'être fournies au réseau neuronal.

Par exemple :

```python
transforms.Resize((224, 224))
```

permet de redimensionner les images à une taille de `224 × 224` pixels.

Puis :

```python
transforms.ToTensor()
```

convertit l'image en tenseur PyTorch.

Des transformations supplémentaires peuvent être utilisées pour l'augmentation des données (*data augmentation*).

---

## 📊 11. Métriques

Pendant l'entraînement, plusieurs indicateurs sont suivis.

### Loss

La `Loss` mesure l'erreur du modèle.

En général :

```text
Loss ↓
```

Plus la Loss diminue, mieux c'est.

Cependant, l'objectif n'est pas nécessairement d'obtenir exactement `0`.

La `Validation Loss` est particulièrement importante pour vérifier la capacité du modèle à généraliser.

### Accuracy

L'`Accuracy` représente le pourcentage de prédictions correctes.

La formule est :

```text
Accuracy = nombre de prédictions correctes / nombre total de prédictions
```

Par exemple :

```text
Accuracy = 0.90
```

correspond à :

```text
90 %
```

Pour un problème à deux classes comme Ants vs Bees, une accuracy proche de 50 % correspond approximativement à une prédiction aléatoire.

Une accuracy de validation autour de 90 % peut déjà représenter une très bonne performance.

---

## ⚠️ 12. Overfitting

Une accuracy très élevée sur les données d'entraînement ne signifie pas nécessairement que le modèle est performant sur de nouvelles données.

Exemple :

```text
Train Accuracy       99 %
Validation Accuracy  75 %
```

Ce résultat peut indiquer un **overfitting (surapprentissage)**.

Le modèle a appris très précisément les images d'entraînement mais généralise mal sur les images de validation.

À l'inverse :

```text
Train Accuracy       94 %
Validation Accuracy  92 %
```

est généralement plus rassurant.

Il faut donc surveiller simultanément :

```text
Train Loss
Validation Loss
Train Accuracy
Validation Accuracy
```

---

## 🎯 13. Quand considérer l'entraînement comme bon ?

Un bon entraînement présente généralement les tendances suivantes :

```text
Train Loss          ↓
Validation Loss     ↓
Train Accuracy      ↑
Validation Accuracy ↑
```

Il faut également vérifier que les performances sur le train et la validation restent relativement proches.

Par exemple :

```text
Train Accuracy       94 %
Validation Accuracy  92 %
```

est un résultat intéressant.

Il ne faut cependant pas chercher automatiquement à obtenir 100 % d'accuracy.

Une accuracy de 100 % sur les données d'entraînement peut être un signe de surapprentissage si l'accuracy de validation est beaucoup plus faible.

L'objectif principal est la **généralisation** du modèle.

---

## 💻 14. Lancer Jupyter Notebook

Après avoir installé les dépendances :

```bash
pip install -r requirements.txt
```

lancer Jupyter Notebook :

```bash
jupyter notebook
```

ou JupyterLab :

```bash
jupyter lab
```

Ouvrir ensuite `classification.ipynb` et exécuter les cellules dans l'ordre.

---

## 🍎 15. Utilisation sur Mac Apple Silicon

Pour les Mac équipés d'une puce Apple Silicon (M1, M2, M3, M4), PyTorch peut utiliser **MPS (Metal Performance Shaders)** afin d'accélérer les calculs.

Vérifier si MPS est disponible :

```python
import torch

print(torch.backends.mps.is_available())
```

Si le résultat est :

```text
True
```

MPS est disponible.

Le device peut alors être défini ainsi :

```python
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

Puis le modèle peut être envoyé sur le device :

```python
model_ft = model_ft.to(device)
```

---

## 📁 16. Fichier requirements.txt

Le projet contient un fichier `requirements.txt` avec les versions des dépendances utilisées.

```text
contourpy==1.3.0
cycler==0.12.1
filelock==3.19.1
fonttools==4.60.2
fsspec==2025.10.0
importlib_resources==6.5.2
Jinja2==3.1.6
kiwisolver==1.4.7
MarkupSafe==3.0.3
matplotlib==3.9.4
mpmath==1.3.0
networkx==3.2.1
numpy==2.0.2
packaging==26.3
pillow==11.3.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
six==1.17.0
sympy==1.14.0
torch==2.8.0
torchvision==0.23.0
typing_extensions==4.16.0
wget==3.2
zipp==3.23.1
```

Pour installer toutes ces dépendances :

```bash
pip install -r requirements.txt
```

---

## 🚀 17. Installation rapide

Pour installer et lancer rapidement le projet :

```bash
# Cloner le projet
git clone <URL_DU_REPOSITORY>

# Entrer dans le projet
cd MLProject

# Créer l'environnement virtuel
python3 -m venv .venv

# Activer l'environnement
source .venv/bin/activate

# Installer toutes les dépendances
pip install -r requirements.txt

# Lancer Jupyter Notebook
jupyter notebook
```

Ensuite, ouvrir :

```text
classification.ipynb
```

---

## 🧹 18. Fichiers à exclure de Git

Le dossier de l'environnement virtuel ne doit généralement pas être envoyé dans Git.

Créer un fichier `.gitignore` contenant :

```text
.venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
```

Selon la taille du dataset, il peut également être préférable de ne pas versionner directement les images :

```text
hymenoptera_data/
```

---

## 🎯 19. Objectif final

L'objectif final du projet est de construire un modèle capable de prendre une image inconnue et de prédire automatiquement si elle représente :

```text
🐜 Ant
```

ou :

```text
🐝 Bee
```

La performance du modèle doit principalement être évaluée sur des données que le modèle n'a pas utilisées pour son entraînement.

L'objectif n'est donc pas simplement d'obtenir la meilleure accuracy possible sur les données d'entraînement, mais de construire un modèle capable de **généraliser correctement à de nouvelles images**.

---

## 🛠️ Technologies utilisées

* Python 3
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pillow
* ResNet-18
* Transfer Learning
* Deep Learning
* Jupyter Notebook
* MPS / Apple Silicon (optionnel)

---

## 📌 Résumé des commandes

```bash
# Créer l'environnement virtuel
python3 -m venv .venv

# Activer l'environnement
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter Notebook
jupyter notebook
```
## 📌 Quelques captures


```
bxxxxxxxx@Air-de-XXXX ~/Documents/MLProject main > pip3 install -r requirements.txt                        ✔  146  21:08:39 
Requirement already satisfied: contourpy==1.3.0 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 1)) (1.3.0)
Requirement already satisfied: cycler==0.12.1 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 2)) (0.12.1)
Requirement already satisfied: filelock==3.19.1 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 3)) (3.19.1)
Requirement already satisfied: fonttools==4.60.2 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 4)) (4.60.2)
Requirement already satisfied: fsspec==2025.10.0 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 5)) (2025.10.0)
Requirement already satisfied: importlib_resources==6.5.2 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 6)) (6.5.2)
Requirement already satisfied: Jinja2==3.1.6 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 7)) (3.1.6)
Requirement already satisfied: kiwisolver==1.4.7 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 8)) (1.4.7)
Requirement already satisfied: MarkupSafe==3.0.3 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 9)) (3.0.3)
Requirement already satisfied: matplotlib==3.9.4 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 10)) (3.9.4)
Requirement already satisfied: mpmath==1.3.0 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 11)) (1.3.0)
Requirement already satisfied: networkx==3.2.1 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 12)) (3.2.1)
Requirement already satisfied: numpy==2.0.2 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 13)) (2.0.2)
Requirement already satisfied: packaging==26.3 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 14)) (26.3)
Requirement already satisfied: pillow==11.3.0 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 15)) (11.3.0)
Requirement already satisfied: pyparsing==3.3.2 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 16)) (3.3.2)
Requirement already satisfied: python-dateutil==2.9.0.post0 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 17)) (2.9.0.post0)
Requirement already satisfied: six==1.17.0 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 18)) (1.17.0)
Requirement already satisfied: sympy==1.14.0 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 19)) (1.14.0)
Requirement already satisfied: torch==2.8.0 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 20)) (2.8.0)
Requirement already satisfied: torchvision==0.23.0 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 21)) (0.23.0)
Requirement already satisfied: typing_extensions==4.16.0 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 22)) (4.16.0)
Requirement already satisfied: wget==3.2 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 23)) (3.2)
Requirement already satisfied: zipp==3.23.1 in ./.venv/lib/python3.9/site-packages (from -r requirements.txt (line 24)) (3.23.1)
 bxxxxxxxx@Air-de-XXXX ~/Documents/MLProject main? > python3 transfert-learning.py                           ✔  146  21:08:47 
Téléchargement du dataset...
100% [........................................................................] 47286322 / 47286322
Décompression du dataset...
Dataset téléchargé et décompressé !
Classes : ['ants', 'bees']
Taille train : 244
Taille validation : 153
Using mps device


============================================================
FINETUNING DU RESNET18
============================================================
Epoch 0/24
----------
train Loss: 0.7330 Acc: 0.6270
val Loss: 0.2171 Acc: 0.9346

Epoch 1/24
----------
train Loss: 0.5206 Acc: 0.7869
val Loss: 0.3039 Acc: 0.8954

Epoch 2/24
----------
train Loss: 0.5156 Acc: 0.7869
val Loss: 0.3095 Acc: 0.8627

Epoch 3/24
----------
train Loss: 0.5926 Acc: 0.7623
val Loss: 0.2665 Acc: 0.8693

Epoch 4/24
----------
train Loss: 0.4392 Acc: 0.8156
val Loss: 0.2641 Acc: 0.9020

Epoch 5/24
----------
train Loss: 0.2976 Acc: 0.8934
val Loss: 0.4044 Acc: 0.8758

Epoch 6/24
----------
train Loss: 0.6155 Acc: 0.7787
val Loss: 0.9126 Acc: 0.7320

Epoch 7/24
----------
train Loss: 0.5592 Acc: 0.7910
val Loss: 0.2518 Acc: 0.9085

Epoch 8/24
----------
train Loss: 0.3132 Acc: 0.8566
val Loss: 0.2382 Acc: 0.9150

Epoch 9/24
----------
train Loss: 0.1772 Acc: 0.9385
val Loss: 0.1809 Acc: 0.9412

Epoch 10/24
----------
train Loss: 0.2366 Acc: 0.8975
val Loss: 0.1963 Acc: 0.9281

Epoch 11/24
----------
train Loss: 0.3246 Acc: 0.8975
val Loss: 0.2023 Acc: 0.9216

Epoch 12/24
----------
train Loss: 0.2519 Acc: 0.8975
val Loss: 0.1784 Acc: 0.9412

Epoch 13/24
----------
train Loss: 0.2962 Acc: 0.8893
val Loss: 0.2408 Acc: 0.9150

Epoch 14/24
----------
train Loss: 0.3085 Acc: 0.8689
val Loss: 0.2014 Acc: 0.9216

Epoch 15/24
----------
train Loss: 0.3140 Acc: 0.8770
val Loss: 0.2263 Acc: 0.9216

Epoch 16/24
----------
train Loss: 0.3458 Acc: 0.8443
val Loss: 0.2824 Acc: 0.8889

Epoch 17/24
----------
train Loss: 0.3686 Acc: 0.8402
val Loss: 0.1911 Acc: 0.9216

Epoch 18/24
----------
train Loss: 0.2611 Acc: 0.8852
val Loss: 0.1928 Acc: 0.9281

Epoch 19/24
----------
train Loss: 0.1949 Acc: 0.9180
val Loss: 0.1949 Acc: 0.9281

Epoch 20/24
----------
train Loss: 0.2501 Acc: 0.8893
val Loss: 0.2078 Acc: 0.9216

Epoch 21/24
----------
train Loss: 0.2217 Acc: 0.9016
val Loss: 0.1763 Acc: 0.9346

Epoch 22/24
----------
train Loss: 0.2945 Acc: 0.8689
val Loss: 0.2072 Acc: 0.9216

Epoch 23/24
----------
train Loss: 0.2802 Acc: 0.8852
val Loss: 0.2187 Acc: 0.9216

Epoch 24/24
----------
train Loss: 0.2325 Acc: 0.9016
val Loss: 0.2295 Acc: 0.9085

Training complete in 2m 28s
Best val Acc: 0.9412


============================================================
RESNET18 COMME FEATURE EXTRACTOR
============================================================
Epoch 0/24
----------
train Loss: 0.5617 Acc: 0.7008
val Loss: 0.1795 Acc: 0.9477

Epoch 1/24
----------
train Loss: 0.5120 Acc: 0.7664
val Loss: 0.2299 Acc: 0.9085

Epoch 2/24
----------
train Loss: 0.3349 Acc: 0.8770
val Loss: 0.1886 Acc: 0.9477

Epoch 3/24
----------
train Loss: 0.4371 Acc: 0.8197
val Loss: 0.1839 Acc: 0.9477

Epoch 4/24
----------
train Loss: 0.4541 Acc: 0.7992
val Loss: 0.2165 Acc: 0.9281

Epoch 5/24
----------
train Loss: 0.5263 Acc: 0.7459
val Loss: 0.2261 Acc: 0.9346

Epoch 6/24
----------
train Loss: 0.4635 Acc: 0.8197
val Loss: 0.2394 Acc: 0.9216

Epoch 7/24
----------
train Loss: 0.4337 Acc: 0.8320
val Loss: 0.1824 Acc: 0.9542

Epoch 8/24
----------
train Loss: 0.3750 Acc: 0.8730
val Loss: 0.1677 Acc: 0.9412

Epoch 9/24
----------
train Loss: 0.4122 Acc: 0.8279
val Loss: 0.1876 Acc: 0.9477

Epoch 10/24
----------
train Loss: 0.3521 Acc: 0.8484
val Loss: 0.1901 Acc: 0.9477

Epoch 11/24
----------
train Loss: 0.3241 Acc: 0.8279
val Loss: 0.1801 Acc: 0.9412

Epoch 12/24
----------
train Loss: 0.3931 Acc: 0.8402
val Loss: 0.1827 Acc: 0.9412

Epoch 13/24
----------
train Loss: 0.3499 Acc: 0.8238
val Loss: 0.1775 Acc: 0.9346

Epoch 14/24
----------
train Loss: 0.3421 Acc: 0.8320
val Loss: 0.1816 Acc: 0.9477

Epoch 15/24
----------
train Loss: 0.3145 Acc: 0.8607
val Loss: 0.1762 Acc: 0.9412

Epoch 16/24
----------
train Loss: 0.3705 Acc: 0.8361
val Loss: 0.1849 Acc: 0.9346

Epoch 17/24
----------
train Loss: 0.4402 Acc: 0.8074
val Loss: 0.1797 Acc: 0.9477

Epoch 18/24
----------
train Loss: 0.2830 Acc: 0.8852
val Loss: 0.1877 Acc: 0.9412

Epoch 19/24
----------
train Loss: 0.4007 Acc: 0.8238
val Loss: 0.1712 Acc: 0.9542

Epoch 20/24
----------
train Loss: 0.3339 Acc: 0.8279
val Loss: 0.1759 Acc: 0.9542

Epoch 21/24
----------
train Loss: 0.3594 Acc: 0.8279
val Loss: 0.1760 Acc: 0.9477

Epoch 22/24
----------
train Loss: 0.3575 Acc: 0.8566
val Loss: 0.1737 Acc: 0.9412

Epoch 23/24
----------
train Loss: 0.2877 Acc: 0.8730
val Loss: 0.1776 Acc: 0.9542

Epoch 24/24
----------
train Loss: 0.3265 Acc: 0.8607
val Loss: 0.1703 Acc: 0.9412

Training complete in 1m 12s
Best val Acc: 0.9542
```
---

## 👨‍💻 Auteur

Projet personnel d'apprentissage du **Deep Learning avec PyTorch**.

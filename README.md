# detecteur-malaria-ala
"Interface IA de détection de malaria par Ala"

Ce projet est une application d'intelligence artificielle capable de détecter si une cellule sanguine est infectée par le parasite de la malaria ou si elle est saine.

## 🔗 Démo en ligne
👉 **Testez l'application ici :** [[Lien vers votre Hugging Face Space ici](https://huggingface.co/spaces/Alajouili123/Detecteur-Malaria/tree/main)]

## 🛠️ Technologies utilisées
* **Python**
* **PyTorch** (Création et entraînement du modèle CNN)
* **Gradio** (Interface utilisateur)
* **Hugging Face Spaces** (Hébergement)

## 📂 Structure du projet
* `app.py` : Le code de l'application web.
* `entrainement_modele.ipynb` : Le Notebook Google Colab utilisé pour entraîner l'IA (95%+ de précision).
* `requirements.txt` : Les dépendances nécessaires.

## 🧠 Le Modèle
Le modèle est un Réseau de Neurones Convolutif (CNN) entraîné sur le dataset public du NIH contenant 27 500 images de cellules.

## 🚀 Comment lancer localement
1. Clonez le repo
2. Installez les dépendances : `pip install -r requirements.txt`
3. Lancez l'app : `python app.py`

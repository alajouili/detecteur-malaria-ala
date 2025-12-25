import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# ---------------------------------------------------------
# 1. DÉFINITION DU MODÈLE (Structure à 1 sortie, imposée par le fichier)
# ---------------------------------------------------------
class MalariaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(8192, 512), 
            nn.ReLU(),
            nn.Linear(512, 1) # Une seule sortie
        )

    def forward(self, xb):
        return self.network(xb)

# ---------------------------------------------------------
# 2. CHARGEMENT
# ---------------------------------------------------------
model = MalariaModel()
filename = "malaria_model_final.pth" 

try:
    if os.path.exists(filename):
        state_dict = torch.load(filename, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Modèle chargé avec succès.")
    else:
        print("⚠️ Fichier introuvable.")
except Exception as e:
    print(f"❌ Erreur : {e}")

# ---------------------------------------------------------
# 3. PRÉDICTION AVEC DEBUG
# ---------------------------------------------------------
def predict_image(img):
    if img is None:
        return None
    
    try:
        # Transformation EXACTE de votre Colab (Resize 64x64 + ToTensor)
        # Pas de normalisation car absente de votre snippet Colab
        transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            transforms.ToTensor()
        ])
        
        img_t = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(img_t)
            # On utilise Sigmoid car c'est un modèle à 1 sortie
            score_brut = torch.sigmoid(prediction).item()
            
        # INTERPRÉTATION
        # Si score proche de 0 => Classe 0 (Souvent Infecté/Parasitized)
        # Si score proche de 1 => Classe 1 (Souvent Sain/Uninfected)
        
        # On affiche le score brut pour comprendre ce qui se passe
        message_debug = f"Score brut du modèle : {score_brut:.5f}\n"
        if score_brut < 0.01 or score_brut > 0.99:
            message_debug += "⚠️ ATTENTION : Le modèle est extrêmement sûr de lui (saturé).\n"
            message_debug += "S'il ne change pas d'avis avec une autre image, le fichier .pth est mauvais."

        return {
            "Infecté (Parasitized)": 1 - score_brut,
            "Sain (Uninfected)": score_brut
        }, message_debug
            
    except Exception as e:
        return {f"Erreur": 0.0}, str(e)

# ---------------------------------------------------------
# 4. INTERFACE
# ---------------------------------------------------------
# J'ajoute une boite de texte pour voir les détails techniques
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=2, label="Résultat"), 
        gr.Textbox(label="Message Technique (Debug)")
    ],
    title="Détecteur Malaria - Test Final",
    description="Si le 'Score brut' ne change pas entre deux images différentes, le modèle est à refaire."
)

if __name__ == "__main__":
    interface.launch()
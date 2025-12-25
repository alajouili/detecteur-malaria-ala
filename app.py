import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# --- 1. DÉFINITION DU MODÈLE (Inchangé) ---
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
            nn.Linear(512, 1)
        )

    def forward(self, xb):
        return self.network(xb)

# --- 2. CHARGEMENT SÉCURISÉ (Inchangé) ---
model = MalariaModel()
filename = "malaria_model_final.pth"

if os.path.exists(filename):
    try:
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        model.eval()
        print("✅ Modèle chargé avec succès !")
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement : {e}")
else:
    print("⚠️ ATTENTION : Le fichier .pth est introuvable.")

# --- 3. LOGIQUE DE PRÉDICTION (Inchangé) ---
transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor()
])

def predict(image):
    if image is None:
        return None
    
    img_t = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_t)
        prob_sain = torch.sigmoid(output).item()
        prob_infecte = 1 - prob_sain 
        
    return {"Sain (Uninfected) 🟢": prob_sain, "Infecté (Parasitized) 🦠": prob_infecte}

# --- 4. INTERFACE GRAPHIQUE (NOUVELLES COULEURS INDIGO/VIOLET) ---

# CSS mis à jour pour le style "Tech"
custom_css = """
.container {max-width: 900px; margin: auto; padding-top: 20px;}

/* Titre en Indigo foncé */
h1 {text-align: center; color: #312e81; font-weight: 800; font-size: 2.5rem; margin-bottom: 0;}

/* Sous-titre en Violet clair */
h3 {text-align: center; color: #6366f1; margin-top: 5px; font-style: italic;}

/* Bouton Dégradé Indigo -> Violet */
.gr-button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important; 
    border: none !important; 
    color: white !important; 
    font-weight: bold;
}
"""

# Thème Indigo et Violet
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
).set(
    button_primary_background_fill="linear-gradient(90deg, #4F46E5, #7C3AED)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4338ca, #6d28d9)",
)

with gr.Blocks(theme=theme, css=custom_css, title="Malaria AI - Ala") as demo:
    
    # --- EN-TÊTE ---
    gr.Markdown("""
    # 🔬 MALARIA AI DETECTION
    ### Designed & Developed by Ala
    """)
    
    gr.HTML("<br>")

    # --- ZONE PRINCIPALE ---
    with gr.Row():
        # Colonne de Gauche
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil", 
                label="Image Microscope", 
                height=300
            )
            # Le bouton aura maintenant le dégradé violet/indigo
            analyze_btn = gr.Button("⚡ LANCER L'ANALYSE", variant="primary", size="lg")

        # Colonne de Droite
        with gr.Column(scale=1):
            output_label = gr.Label(
                num_top_classes=2, 
                label="Résultat IA"
            )

    # --- INTERACTIONS ---
    analyze_btn.click(
        fn=predict, 
        inputs=input_image, 
        outputs=output_label
    )

if __name__ == "__main__":
    demo.launch()

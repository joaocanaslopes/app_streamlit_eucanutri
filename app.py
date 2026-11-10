import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="IA Eucalipto - Navigator", page_icon="🌱")

# Ordem alfabética exata do ImageFolder do PyTorch
CLASSES = ['B', 'K', 'N', 'P', 'normal', 'não nutricional']
REPO_ID = "Rainozita/eucalipto-weights" # O teu repo no HF

# --- 1. FUNÇÃO PARA DESCARREGAR E CARREGAR MODELOS (com Cache) ---
# O st.cache_resource garante que isto só corre uma vez no arranque da app
@st.cache_resource
def load_all_models():
    st.info("A inicializar o cérebro da IA... (isto demora um pouco na primeira vez)")
    
    device = torch.device('cpu') # Forçamos CPU para o Streamlit Free Tier
    
    # --- Definir e Carregar ResNet50 ---
    resnet_path = hf_hub_download(repo_id=REPO_ID, filename="resnet50_eucalipto.pth")
    model_rn = models.resnet50(weights=None)
    in_features_rn = model_rn.fc.in_features
    model_rn.fc = nn.Sequential(
        nn.Linear(in_features_rn, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASSES))
    )
    model_rn.load_state_dict(torch.load(resnet_path, map_location=device))
    model_rn.eval()
    
    # --- Definir e Carregar EfficientNet-B0 ---
    effnet_path = hf_hub_download(repo_id=REPO_ID, filename="efficientnet_eucalipto.pth")
    model_en = models.efficientnet_b0(weights=None)
    in_features_en = model_en.classifier[1].in_features
    model_en.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features_en, len(CLASSES))
    )
    model_en.load_state_dict(torch.load(effnet_path, map_location=device))
    model_en.eval()
    
    st.success("Modelos carregados com sucesso do Hugging Face Hub!")
    return model_rn, model_en

# Carregar os modelos (vai buscar à cache se já tiver corrido)
try:
    resnet_model, efficientnet_model = load_all_models()
except Exception as e:
    st.error(f"Erro ao carregar modelos do Hugging Face. Verifica se o repo é público. Erro: {e}")
    st.stop()

# --- 2. PRÉ-PROCESSAMENTO DA IMAGEM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. FUNÇÃO DE INFERÊNCIA HÍBRIDA (Adaptada para CPU) ---
def predict(image, model_choice, localizacao_ramo, estado_folha):
    # Selecionar o modelo (já estão em CPU)
    model = resnet_model if model_choice == "ResNet50" else efficientnet_model
    
    # Processar a imagem
    img_tensor = transform(image).unsqueeze(0) # Mantém em CPU
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Obter probabilidades base da IA
    confidences = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    
    # --- REGRAS DE PÓS-PROCESSAMENTO AGRONÓMICO (Lógica Híbrida) ---
    
    # Regra 1: Filtro de Tecido Assintomático (Normal vs Não Nutricional)
    if estado_folha == "Tecido Assintomático (Sem lesões ou manchas)":
        confidences['normal'] = 0.95
        for k in confidences:
            if k != 'normal':
                confidences[k] = 0.01 / (len(CLASSES) - 1) # Distribui o resto
    
    # Regra 2: Filtro de Ramo Jovem (Contexto Boro vs Potássio)
    elif localizacao_ramo == "Ramo Jovem / Topo":
        # Dá um boost ao Boro se houver dúvida com Potássio
        if confidences.get('K', 0) > 0.15 or confidences.get('B', 0) > 0.15:
            # Garante que N não é a classe dominante antes de dar boost a B
            if confidences.get('N', 0) < max(confidences.get('K', 0), confidences.get('B', 0)):
                confidences['B'] += 0.35
                # Normalizar novamente para somar 1
                total = sum(confidences.values())
                confidences = {k: v/total for k, v in confidences.items()}
    
    return confidences

# --- 4. INTERFACE GRÁFICA COM STREAMLIT ---
st.title("🌱 Classificador Nutricional de Eucalipto")
st.markdown("### Sistema Híbrido: Deep Learning + Regras Agronómicas")
st.write("Desenvolvido para Navigator. Corre em CPU (previsões ilimitadas).")

with st.sidebar:
    st.header("Configurações e Contexto")
    model_choice = st.radio(
        "Escolher Modelo de IA:",
        ["ResNet50", "EfficientNet-B0"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Contexto de Campo (Chave Dicotómica)")
    
    localizacao_ramo = st.radio(
        "Localização na Planta:",
        ["Ramo Jovem / Topo", "Ramo Velho / Base / Indiferente"],
        index=1,
        help="Importante para distinguir Boro (Topo) de Potássio (Base)."
    )
    
    estado_folha = st.radio(
        "Integridade Foliar:",
        ["Presença de Sintomas / Danos Visíveis", "Tecido Assintomático (Sem lesões ou manchas)"],
        index=0,
        help="Ajuda a IA a confirmar se a folha está saudável."
    )

# Área principal
uploaded_file = st.file_uploader("Carrega a foto de uma folha de Eucalipto...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Mostrar imagem
    st.image(image, caption='Folha carregada', use_column_width=True)
    
    # Botão para diagnosticar
    if st.button('Executar Diagnóstico'):
        with st.spinner('A analisar...'):
            results = predict(image, model_choice, localizacao_ramo, estado_folha)
            
            # Ordenar resultados do maior para o menor
            sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
            
            st.subheader("Resultado do Diagnóstico Híbrido:")
            
            # Mostrar a classe principal com destaque
            top_class, top_conf = sorted_results[0]
            if top_class in ['B', 'K', 'N', 'P']:
                st.error(f"⚠️ Deficiência Detectada: **{top_class}** ({top_conf:.1%})")
            elif top_class == 'normal':
                st.success(f"✅ Folha Saudável / Normal ({top_conf:.1%})")
            else:
                st.warning(f"⚪ Dano Não Nutricional ({top_conf:.1%})")

            st.markdown("---")
            st.write("Detalhe das Probabilidades Ajustadas:")
            # Mostrar barras de progresso
            for label, conf in sorted_results:
                st.write(f"**{label}**")
                st.progress(conf)
                st.caption(f"{conf:.1%} de confiança")

else:
    st.info("Por favor, carrega uma imagem para iniciar o diagnóstico.")

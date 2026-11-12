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
@st.cache_resource
def load_all_models():
    st.info("A inicializar o cérebro da IA... (isto demora um pouco na primeira vez)")
    
    device = torch.device('cpu') # Forçamos CPU para o Streamlit Free Tier
    hf_token = st.secrets.get("HF_TOKEN", None)
    
    # --- Definir e Carregar ResNet50 ---
    resnet_path = hf_hub_download(
        repo_id=REPO_ID, 
        filename="resnet50_eucalipto.pth",
        token=hf_token
    )
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
    effnet_path = hf_hub_download(
        repo_id=REPO_ID, 
        filename="efficientnet_eucalipto.pth",
        token=hf_token
    )
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

# Carregar os modelos
try:
    resnet_model, efficientnet_model = load_all_models()
except Exception as e:
    st.error(f"Erro ao carregar modelos do Hugging Face. Verifica se o token e o repositório estão corretos. Erro: {e}")
    st.stop()

# --- 2. PRÉ-PROCESSAMENTO DA IMAGEM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. FUNÇÃO DE INFERÊNCIA HÍBRIDA (Adaptada para CPU) ---
def predict(image, model_choice, localizacao_ramo, cor_folha, necrose_vshape, stress_hidrico, presenca_insetos, estado_folha):
    # Selecionar o modelo
    model = resnet_model if model_choice == "ResNet50" else efficientnet_model
    
    # Processar a imagem
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Obter probabilidades base da IA
    confidences = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    
    # --- REGRAS DE PÓS-PROCESSAMENTO AGRONÓMICO (Lógica Híbrida) ---
    
    # Bloco 6: Filtro de Tecido Assintomático (Normal vs Não Nutricional)
    if estado_folha == "Tecido Assintomático (Sem lesões ou manchas)":
        confidences['normal'] = 0.95
        for k in confidences:
            if k != 'normal':
                confidences[k] = 0.01 / (len(CLASSES) - 1)
    
    else:
        # Bloco 1 & 2: Zona do Ramo + Cor da Folha (Linha 1, 2 e 3 da Matriz)
        if localizacao_ramo == "Ramo Jovem / Topo":
            # Regra Boro (Linha 1)
            if confidences.get('K', 0) > 0.15 or confidences.get('B', 0) > 0.15:
                if confidences.get('N', 0) < max(confidences.get('K', 0), confidences.get('B', 0)):
                    confidences['B'] *= 1.7
                    
        elif localizacao_ramo == "Ramo Velho / Base / Indiferente":
            # Regra Azoto - N (Linha 2: Folhas velhas + Amarelecimento uniforme)[cite: 1]
            if cor_folha == "Amarelecimento uniforme":
                confidences['N'] *= 1.5
                
            # Regra Fósforo - P (Linha 3: Folhas velhas + Coloração arroxeada/avermelhada)[cite: 1]
            elif cor_folha == "Coloração arroxeada / avermelhada":
                confidences['P'] *= 1.5

        # Bloco 3: Necrose em V-shape / Marginal (Linha 4: Potássio - K)[cite: 1]
        if necrose_vshape == "Sim (Presença de necrose marginal / V-shape)":
            confidences['K'] *= 1.5

        # Bloco 4: Condições Externas - Défice Hídrico / Salinidade (Linha 5)[cite: 1]
        if stress_hidrico == "Sim (Presente)":
            confidences['K'] *= 1.2

        # Bloco 5: Presença de Insetos / Pragas (Boost em Não Nutricional)
        if presenca_insetos == "Sim":
            confidences['não nutricional'] *= 1.5

        # Normalizar probabilidades finais para somar 1.0 se algum boost foi aplicado
        total = sum(confidences.values())
        if total > 0:
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
    st.subheader("Chave Dicotómica & Contexto de Campo")
    
    # Bloco 1: Zona do Ramo
    localizacao_ramo = st.radio(
        "1. Zona do Ramo:",
        ["Desativar este filtro (Teste)", "Ramo Jovem / Topo", "Ramo Velho / Base / Indiferente"],
        index=2,
        help="Localização principal do sintome na planta."
    )
    
    # Bloco 2: Cor da Folha
    cor_folha = st.radio(
        "2. Cor / Sintoma Foliar:",
        ["Desativar este filtro (Teste)", "Amarelecimento uniforme", "Coloração arroxeada / avermelhada", "Outro / Sem padrão específico"],
        index=3,
        help="Cruza com folhas velhas para Azoto (N) ou Fósforo (P)."
    )
    
    # Bloco 3: Necrose em V-shape
    necrose_vshape = st.radio(
        "3. Necrose Marginal / V-shape (Potássio):",
        ["Desativar este filtro (Teste)", "Sim (Presença de necrose marginal / V-shape)", "Não"],
        index=2,
        help="Indica carência de Potássio (K) agravada."
    )
    
    # Bloco 4: Condições Externas
    stress_hidrico = st.radio(
        "4. Condições Externas (Défice Hídrico / Salinidade):",
        ["Não conclusivo / Irrelevante (Desativar)", "Sim (Presente)", "Não"],
        index=0,
        help="CE02 (Défice hídrico) ou CE12 (Salinidade) - Boost em K (*1.2)."
    )
    
    # Bloco 5: Insetos / Pragas
    presenca_insetos = st.radio(
        "5. Presença de Insetos / Pragas:",
        ["Desativar este filtro (Teste)", "Sim", "Não"],
        index=2,
        help="Dano biótico (desvia o diagnóstico para Não Nutricional)."
    )
    
    st.markdown("---")
    # Bloco 6: Integridade Foliar
    estado_folha = st.radio(
        "6. Integridade Foliar (Geral):",
        ["Presença de Sintomas / Danos Visíveis", "Tecido Assintomático (Sem lesões ou manchas)"],
        index=0,
        help="Filtro rápido para tecido totalmente saudável."
    )

# Área principal
uploaded_file = st.file_uploader("Carrega a foto de uma folha de Eucalipto...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Mostrar imagem
    st.image(image, caption='Folha carregada', use_container_width=True)
    
    # Botão para diagnosticar
    if st.button('Executar Diagnóstico'):
        with st.spinner('A analisar...'):
            results = predict(image, model_choice, localizacao_ramo, cor_folha, necrose_vshape, stress_hidrico, presenca_insetos, estado_folha)
            
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

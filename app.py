# app.py : l'interface web avec streamlit
# lancer avec : python -m streamlit run app.py

import streamlit as st
from rag import lire_pdf, decouper, vectoriser, chercher, repondre

st.title("📄 Assistant RAG multilingue (Mistral)")
st.write("Dépose des PDF, puis pose ta question en français, anglais ou arabe.")

# on dépose un ou plusieurs PDF
fichiers = st.file_uploader("Tes documents PDF", type="pdf", accept_multiple_files=True)

# bouton pour préparer les documents (lecture + découpage + vecteurs)
if fichiers and st.button("Préparer les documents"):
    tous_morceaux = []
    for f in fichiers:
        tous_morceaux += decouper(lire_pdf(f), f.name)
    try:
        with st.spinner(f"Calcul des vecteurs pour {len(tous_morceaux)} morceaux... (ça peut prendre 1 minute)"):
            vecteurs = vectoriser([m["texte"] for m in tous_morceaux])
        # on garde tout en mémoire pour les questions suivantes
        st.session_state["morceaux"] = tous_morceaux
        st.session_state["vecteurs"] = vecteurs
        st.session_state["reponses"] = {}  # on vide les anciennes réponses
        st.success(f"{len(tous_morceaux)} morceaux prêts !")
    except Exception as e:
        st.error(f"Problème avec l'API Mistral : {e}\n\nAttends 1 minute puis réessaie.")

# zone de question
if "morceaux" in st.session_state:
    # un formulaire : la question n'est envoyée qu'au clic sur le bouton
    # (sinon streamlit relance tout à chaque action et appelle l'API pour rien)
    with st.form("question_form"):
        question = st.text_input("Ta question :")
        # choix de la méthode de recherche
        methode = st.radio("Méthode de recherche :", ["hybride", "semantique", "bm25"], horizontal=True,
                           help="hybride = sens + mots-clés ; semantique = sens (vecteurs) ; bm25 = mots-clés")
        envoyer = st.form_submit_button("Envoyer")

    if envoyer and question:
        reponses = st.session_state.setdefault("reponses", {})
        # si on a déjà répondu à cette question, on ne rappelle pas l'API
        cle = (question, methode)  # la réponse dépend aussi de la méthode
        if cle not in reponses:
            try:
                with st.spinner("Je réfléchis..."):
                    trouves = chercher(question, st.session_state["morceaux"], st.session_state["vecteurs"], methode=methode)
                    reponses[cle] = (repondre(question, trouves), trouves)
            except Exception as e:
                st.error(f"Problème avec l'API Mistral : {e}\n\nAttends 1 minute puis réessaie.")

        if cle in reponses:
            reponse, trouves = reponses[cle]
            st.markdown("### Réponse")
            st.write(reponse)
            # on montre les passages utilisés (transparence)
            with st.expander("Voir les passages utilisés"):
                for m in trouves:
                    st.markdown(f"**{m['source']} – page {m['page']}**")
                    st.write(m["texte"])

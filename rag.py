# rag.py : le "cerveau" du projet
# on lit des PDF, on les coupe en morceaux, on les transforme en vecteurs
# et on cherche les morceaux les plus proches de la question

import os
import re
import math
import time
from collections import Counter
import numpy as np
from pypdf import PdfReader

# selon la version du SDK mistral, l'import change un peu
try:
    from mistralai.client import Mistral   # version 2
except ImportError:
    from mistralai import Mistral          # version 1

# la clé API est lue dans une variable d'environnement (jamais écrite dans le code !)
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

MODELE_EMBED = "mistral-embed"
MODELE_CHAT = "ministral-8b-latest"


# la formule gratuite limite le nb d'appels par seconde (erreur 429)
# 1) on laisse toujours au moins 2 secondes entre deux appels
# 2) si on a quand même une erreur 429, on attend et on réessaie
DERNIER_APPEL = [0.0]

def avec_reessai(fonction, essais=6):
    attente = 5
    for i in range(essais):
        # on attend si le dernier appel est trop récent
        ecart = time.time() - DERNIER_APPEL[0]
        if ecart < 2:
            time.sleep(2 - ecart)
        DERNIER_APPEL[0] = time.time()
        try:
            return fonction()
        except Exception as e:
            if "429" in str(e) and i < essais - 1:
                print(f"Limite Mistral atteinte, j'attends {attente} s...")
                time.sleep(attente)
                attente *= 2  # 5s, 10s, 20s, 40s...
            else:
                raise


# 1) lire un PDF et récupérer tout son texte
def lire_pdf(chemin):
    lecteur = PdfReader(chemin)
    pages = []
    for num, page in enumerate(lecteur.pages):
        texte = page.extract_text() or ""
        pages.append((num + 1, texte))  # on garde le numéro de page pour citer la source
    return pages


# 2) couper le texte en petits morceaux (chunks)
# taille = nb de mots par morceau, chevauchement = mots repris du morceau d'avant
def decouper(pages, nom_fichier, taille=200, chevauchement=40):
    morceaux = []
    for num_page, texte in pages:
        mots = texte.split()
        debut = 0
        while debut < len(mots):
            bout = " ".join(mots[debut:debut + taille])
            if bout.strip():
                morceaux.append({"texte": bout, "source": nom_fichier, "page": num_page})
            debut += taille - chevauchement
    return morceaux


# 3) transformer des textes en vecteurs avec mistral-embed
# on envoie par paquets de 32 pour ne pas dépasser la limite de l'API
def vectoriser(textes):
    vecteurs = []
    for i in range(0, len(textes), 32):
        paquet = textes[i:i + 32]
        rep = avec_reessai(lambda: client.embeddings.create(model=MODELE_EMBED, inputs=paquet))
        vecteurs.extend([d.embedding for d in rep.data])
    return np.array(vecteurs)


# 4) la recherche : 3 méthodes possibles

# 4a) recherche par le SENS : similarité cosinus entre les vecteurs
def scores_semantiques(vq, vecteurs):
    # on normalise pour que le produit scalaire = cosinus
    v_norm = vecteurs / np.linalg.norm(vecteurs, axis=1, keepdims=True)
    q_norm = vq / np.linalg.norm(vq)
    return v_norm @ q_norm


# 4b) recherche par MOTS-CLÉS : l'algorithme BM25 (codé à la main, pas besoin d'installer quoi que ce soit)
def decouper_mots(texte):
    # \w+ marche aussi pour l'arabe et les accents
    return re.findall(r"\w+", texte.lower())


def scores_bm25(question, morceaux, k1=1.5, b=0.75):
    docs = [decouper_mots(m["texte"]) for m in morceaux]
    n = len(docs)
    longueur_moy = sum(len(d) for d in docs) / n
    # df = dans combien de morceaux apparaît chaque mot
    df = Counter()
    for d in docs:
        df.update(set(d))
    scores = np.zeros(n)
    for mot in set(decouper_mots(question)):
        if mot not in df:
            continue
        idf = math.log(1 + (n - df[mot] + 0.5) / (df[mot] + 0.5))  # un mot rare compte plus
        for i, d in enumerate(docs):
            tf = d.count(mot)  # nb de fois où le mot est dans le morceau
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / longueur_moy))
    return scores


# petite fonction pour ramener des scores entre 0 et 1 (pour pouvoir les additionner)
def normaliser(scores):
    ecart = scores.max() - scores.min()
    if ecart == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / ecart


# 4c) la fonction principale : methode = "semantique", "bm25" ou "hybride"
def chercher(question, morceaux, vecteurs, k=4, methode="hybride", vq=None):
    if methode != "bm25" and vq is None:
        vq = vectoriser([question])[0]  # on ne vectorise la question que si besoin
    if methode == "semantique":
        scores = scores_semantiques(vq, vecteurs)
    elif methode == "bm25":
        scores = scores_bm25(question, morceaux)
    else:
        # hybride = moitié sens + moitié mots-clés
        scores = 0.5 * normaliser(scores_semantiques(vq, vecteurs)) + 0.5 * normaliser(scores_bm25(question, morceaux))
    meilleurs = np.argsort(scores)[::-1][:k]
    return [morceaux[i] for i in meilleurs]


# 5) générer la réponse avec le LLM, à partir des morceaux trouvés
def repondre(question, morceaux_trouves):
    contexte = ""
    for m in morceaux_trouves:
        contexte += f"[{m['source']}, page {m['page']}]\n{m['texte']}\n\n"

    consigne = (
        "Tu es un assistant qui répond UNIQUEMENT à partir des documents fournis. "
        "Si la réponse n'est pas dans les documents, dis que tu ne sais pas. "
        "Cite tes sources entre crochets, par exemple [cours.pdf, page 3]. "
        "Réponds dans la même langue que la question (français, anglais ou arabe)."
    )

    rep = avec_reessai(lambda: client.chat.complete(
        model=MODELE_CHAT,
        messages=[
            {"role": "system", "content": consigne},
            {"role": "user", "content": f"Documents :\n{contexte}\nQuestion : {question}"},
        ],
        temperature=0.2,  # faible = réponses plus stables
    ))
    return rep.choices[0].message.content


# petit test en ligne de commande : python rag.py mon_fichier.pdf
if __name__ == "__main__":
    import sys
    chemin = sys.argv[1]
    morceaux = decouper(lire_pdf(chemin), os.path.basename(chemin))
    print(f"{len(morceaux)} morceaux créés")
    vecteurs = vectoriser([m["texte"] for m in morceaux])
    question = input("Ta question : ")
    trouves = chercher(question, morceaux, vecteurs)
    print(repondre(question, trouves))

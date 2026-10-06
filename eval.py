# eval.py : on mesure si la recherche retrouve le BON passage
# lancer avec : python eval.py mon_document.pdf
#
# principe : pour chaque question test, on connaît la page où se trouve la réponse.
# on regarde si cette page est dans les k morceaux trouvés (score "hit@k").
# on compare les 3 méthodes : sémantique, mots-clés (BM25) et hybride.

import os
import sys
from rag import lire_pdf, decouper, vectoriser, chercher

# ===== À MODIFIER : tes questions test et la page où se trouve la réponse =====
# (ouvre ton PDF, choisis 10 questions dont tu connais la page de la réponse)
QUESTIONS = [
    ("Quel est l'objectif du mémoire ?", 1),
    ("Quel modèle de langage est utilisé ?", 2),
    ("Quel est l'objectif du projet ?", 1),
    ("Quel modèle de langage est utilisé ?", 2),
    ("Quelles émotions sont détectées ?", 2),
    ("Comment est géré le déséquilibre des classes ?", 3),
    ("Quelles métriques sont utilisées pour l'évaluation ?", 3),
    ("What is the main goal of the project?", 1),
    ("ما هو موضوع هذا العمل؟", 1),
]
K = 4  # nb de morceaux récupérés par question
# ==============================================================================

chemin = sys.argv[1]
morceaux = decouper(lire_pdf(chemin), os.path.basename(chemin))
print(f"{len(morceaux)} morceaux, calcul des vecteurs...")
vecteurs = vectoriser([m["texte"] for m in morceaux])

# on vectorise toutes les questions en une seule fois (moins d'appels à l'API)
vecteurs_q = vectoriser([q for q, _ in QUESTIONS])

resultats = {"semantique": 0, "bm25": 0, "hybride": 0}
print(f"\n{'Question':<55} {'page':>4}  sém.  bm25  hyb.")
for (question, page_attendue), vq in zip(QUESTIONS, vecteurs_q):
    ligne = f"{question[:55]:<55} {page_attendue:>4}"
    for methode in resultats:
        trouves = chercher(question, morceaux, vecteurs, k=K, methode=methode, vq=vq)
        trouve = any(m["page"] == page_attendue for m in trouves)
        resultats[methode] += trouve
        ligne += "   ✅ " if trouve else "   ❌ "
    print(ligne)

# score final pour chaque méthode
n = len(QUESTIONS)
print(f"\nScore hit@{K} (bonne page dans les {K} passages trouvés) :")
for methode, bons in resultats.items():
    print(f"  {methode:<11} {bons}/{n}  ({100 * bons / n:.0f} %)")

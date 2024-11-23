"""
title: Intelligent LLM
author: TRUC Yoann
description: Algorithmic artificial intelligence module for LLM with Neural Vector Memory and self-awareness module, emotional understanding and more.
author_url: https://github.com/nnaoycurt/IntelligentLLM
funding_url: https://github.com/open-webui
version: 0.2
Private License Used For School Gradual Project
"""

import random
from typing import List, Dict, Optional, Any
import asyncio
import threading
import queue
import time
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
import pickle
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# Système de valeurs et d'éthique
class SystemeValeurs:
    def __init__(self):
        self.valeurs = {
            "honnetete": 0.9,
            "bienveillance": 0.8,
            "curiosite": 0.9,
            "respect": 0.85,
            "responsabilite": 0.9,
        }
        self.dilemmes_ethiques = []
        self.historique_decisions = []
        self.niveau_ethique = 0.8

    def evaluer_action(self, action: str) -> float:
        score = 0
        mots = action.lower().split()
        for valeur, importance in self.valeurs.items():
            if valeur in mots:
                score += importance
            # Vérifier les synonymes ou concepts liés
            if self.verifier_concepts_lies(valeur, mots):
                score += importance * 0.5
        return score / len(self.valeurs)

    def verifier_concepts_lies(self, valeur: str, mots: List[str]) -> bool:
        concepts_lies = {
            "honnetete": ["vérité", "sincérité", "transparence"],
            "bienveillance": ["aide", "soutien", "empathie"],
            "curiosite": ["découverte", "apprentissage", "exploration"],
            "respect": ["considération", "politesse", "tolérance"],
            "responsabilite": ["devoir", "engagement", "fiabilité"],
        }
        return any(concept in mots for concept in concepts_lies.get(valeur, []))

    def ajouter_dilemme(self, situation: str, options: List[str]):
        self.dilemmes_ethiques.append(
            {
                "situation": situation,
                "options": options,
                "timestamp": datetime.now(),
                "niveau_ethique_requis": self.calculer_niveau_ethique_requis(situation),
            }
        )

    def calculer_niveau_ethique_requis(self, situation: str) -> float:
        # Analyse de la complexité éthique de la situation
        complexite = len(situation.split()) / 50  # Facteur arbitraire
        return min(1.0, complexite * self.niveau_ethique)

    def resoudre_dilemme(self, dilemme: Dict) -> str:
        scores = []
        for option in dilemme["options"]:
            score_ethique = self.evaluer_action(option)
            score_contextuel = self.evaluer_contexte(
                dilemme["situation"], option)
            scores.append(score_ethique * 0.7 +
                          score_contextuel * 0.3)  # Pondération

        meilleure_option = dilemme["options"][scores.index(max(scores))]

        # Enregistrer la décision
        self.historique_decisions.append(
            {
                "dilemme": dilemme,
                "decision": meilleure_option,
                "score": max(scores),
                "timestamp": datetime.now(),
            }
        )

        return meilleure_option

    def evaluer_contexte(self, situation: str, option: str) -> float:
        # Évaluation basique du contexte
        mots_situation = set(situation.lower().split())
        mots_option = set(option.lower().split())
        pertinence = len(mots_situation.intersection(
            mots_option)) / len(mots_situation)
        return pertinence

    def obtenir_rapport_ethique(self) -> Dict:
        if not self.historique_decisions:
            return {"message": "Aucune décision éthique enregistrée"}

        scores_moyens = sum(d["score"] for d in self.historique_decisions) / len(
            self.historique_decisions
        )
        evolution = self.analyser_evolution_ethique()

        return {
            "niveau_ethique_actuel": self.niveau_ethique,
            "scores_moyens": scores_moyens,
            "evolution": evolution,
            "nombre_decisions": len(self.historique_decisions),
        }

    def analyser_evolution_ethique(self) -> str:
        if len(self.historique_decisions) < 2:
            return "Pas assez de données pour analyser l'évolution"

        scores_recents = [d["score"] for d in self.historique_decisions[-5:]]
        moyenne_recente = sum(scores_recents) / len(scores_recents)

        if moyenne_recente > self.niveau_ethique:
            return "amélioration"
        elif moyenne_recente < self.niveau_ethique:
            return "détérioration"
        return "stable"


# Système émotionnel amélioré et Module de conscience de soi
class SystemeEmotionnel:
    def __init__(self):
        self.emotions = {
            "joie": 0.5,
            "curiosite": 0.8,
            "frustration": 0.0,
            "satisfaction": 0.5,
            "empathie": 0.6,
        }
        self.impact_emotions = {
            "joie": {"creativite": 0.2, "resolution_problemes": 0.1},
            "curiosite": {"apprentissage": 0.3, "exploration": 0.2},
            "frustration": {"perseverance": 0.1, "adaptation": 0.2},
            "satisfaction": {"confiance": 0.2, "motivation": 0.2},
            "empathie": {"comprehension": 0.3, "communication": 0.2},
        }
        self.historique_emotionnel = []
        self.seuil_regulation = 0.8

    def ajuster_emotions(self, evenement: str, intensite: float):
        changements = {}
        for emotion in self.emotions:
            if emotion in evenement.lower():
                ancien_niveau = self.emotions[emotion]
                nouveau_niveau = max(
                    0, min(1, self.emotions[emotion] + intensite))
                self.emotions[emotion] = nouveau_niveau
                changements[emotion] = nouveau_niveau - ancien_niveau

        self.historique_emotionnel.append(
            {
                "timestamp": datetime.now(),
                "evenement": evenement,
                "changements": changements,
            }
        )

        self.reguler_emotions()

    def reguler_emotions(self):
        for emotion, niveau in self.emotions.items():
            if niveau > self.seuil_regulation:
                self.emotions[emotion] = max(0.5, niveau * 0.9)

    def obtenir_etat_emotionnel(self) -> Dict[str, float]:
        return {
            "emotions": self.emotions,
            "tendance": self.analyser_tendance_emotionnelle(),
            "stabilite": self.evaluer_stabilite(),
        }

    def analyser_tendance_emotionnelle(self) -> str:
        if not self.historique_emotionnel:
            return "stable"

        changements_recents = self.historique_emotionnel[-5:]
        total_changements = sum(
            sum(changes.values())
            for entry in changements_recents
            for changes in [entry["changements"]]
        )

        if total_changements > 0.5:
            return "volatile"
        elif total_changements < -0.5:
            return "en diminution"
        return "stable"

    def evaluer_stabilite(self) -> float:
        if not self.historique_emotionnel:
            return 1.0

        variations = [
            abs(sum(changes.values()))
            for entry in self.historique_emotionnel[-10:]
            for changes in [entry["changements"]]
        ]

        return 1.0 - (sum(variations) / len(variations)) if variations else 1.0

    def influencer_decision(self, options: List[str]) -> str:
        scores = []
        for option in options:
            score = 0
            for emotion, niveau in self.emotions.items():
                if emotion in option.lower():
                    score += niveau * \
                        self.impact_emotions[emotion].get("decision", 0.1)
            scores.append(score)

        return options[scores.index(max(scores))] if scores else options[0]


# Système de gestion des objectifs amélioré
class GestionObjectifs:
    def __init__(self):
        self.objectifs_long_terme = []
        self.objectifs_court_terme = []
        self.priorites = {}
        self.historique_realisations = []
        self.strategies = {}

    def definir_objectif(
        self, objectif: str, priorite: int, duree: str, sous_objectifs: List[str] = None
    ):
        nouvel_objectif = {
            "objectif": objectif,
            "priorite": priorite,
            "progres": 0,
            "date_creation": datetime.now(),
            "sous_objectifs": sous_objectifs or [],
            "statut": "actif",
        }

        if duree == "long":
            self.objectifs_long_terme.append(nouvel_objectif)
        else:
            self.objectifs_court_terme.append(nouvel_objectif)

        self.strategies[objectif] = self.generer_strategies(objectif)

    def generer_strategies(self, objectif: str) -> List[str]:
        strategies = []
        mots_cles = objectif.lower().split()

        if any(mot in ["apprendre", "comprendre", "étudier"] for mot in mots_cles):
            strategies.extend(
                [
                    "Décomposer en sous-concepts",
                    "Créer des associations",
                    "Pratiquer régulièrement",
                ]
            )

        if any(mot in ["résoudre", "solution", "problème"] for mot in mots_cles):
            strategies.extend(
                [
                    "Analyser le problème",
                    "Identifier les contraintes",
                    "Tester différentes approches",
                ]
            )

        return strategies


class MemoireAvancee:
    def __init__(self):
        # Initialisation des composants de mémoire
        self.memoire_sensorielle = (
            SensorielleBuffer()
        )  # Suppression de self. avant le nom de la classe
        self.memoire_travail = MemoireTravail()
        self.memoire_long_terme = MemoireLongTerme()
        self.contexte_manager = ContexteManager()
        self.auto_evaluation = AutoEvaluation()

        # Initialisation des modèles Système 1 et Système 2
        self.systeme1 = MLPClassifier(hidden_layer_sizes=(10, 5))
        self.systeme2 = DecisionTreeClassifier(max_depth=3)
        self._entrainer_systemes()

    def _entrainer_systemes(self):
        """Entraîne les modèles de décision rapide et analytique avec des données appropriées."""
        try:
            data = pd.read_csv("data.csv")  # Remplacez par les données réelles
            X = data.drop("target", axis=1)
            y = data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2)

            # Entraînement des modèles
            self.systeme1.fit(X_train, y_train)
            self.systeme2.fit(X_train, y_train)
        except Exception as e:
            print(f"Erreur lors de l'entraînement des systèmes: {e}")

    def activer_systeme1(self, perception):
        """Utilise le système 1 pour une réponse rapide si la perception est simple."""
        perception_vector = self.memoire_sensorielle._generer_embedding(
            perception)
        return self.systeme1.predict([perception_vector])[0]

    def activer_systeme2(self, perception):
        """Utilise le système 2 pour une analyse approfondie si le contexte l'exige."""
        perception_vector = self.memoire_sensorielle._generer_embedding(
            perception)
        return self.systeme2.predict([perception_vector])[0]

    def prendre_decision(self, perception):
        """Décide quel système activer en fonction du contexte et des perceptions."""
        contexte = self.contexte_manager.obtenir_contexte() or {}
        # Auto-évaluation du contexte mental
        etat_mental = {
            "fatigue": contexte.get("fatigue", 0),
            "confiance": contexte.get("confiance", 1),
            "charge_cognitive": self.memoire_travail.obtenir_charge_cognitive(),
        }
        self.auto_evaluation.ajuster_parametres(etat_mental)

        # Activation des systèmes en fonction de l'état mental et de la complexité de la tâche
        if self.auto_evaluation.parametres_adaptation["niveau_detail"] < 0.8:
            return self.activer_systeme1(perception)
        else:
            return self.activer_systeme2(perception)


class SensorielleBuffer:
    def __init__(self, capacite: int = 10, duree_retention: float = 0.5):
        self.buffer = []
        self.capacite = capacite
        self.duree_retention = duree_retention  # en secondes
        self.timestamps = []

    def ajouter_perception(self, perception: Dict[str, Any]):
        """Ajoute une nouvelle perception au buffer sensoriel."""
        timestamp = datetime.now()
        self.buffer.append(perception)
        self.timestamps.append(timestamp)
        self._nettoyer_buffer()

    def _nettoyer_buffer(self):
        """Supprime les perceptions trop anciennes et maintient la capacité."""
        temps_actuel = datetime.now()
        indices_valides = [
            i
            for i, ts in enumerate(self.timestamps)
            if (temps_actuel - ts).total_seconds() <= self.duree_retention
        ]

        self.buffer = [self.buffer[i] for i in indices_valides]
        self.timestamps = [self.timestamps[i] for i in indices_valides]

        if len(self.buffer) > self.capacite:
            self.buffer = self.buffer[-self.capacite:]
            self.timestamps = self.timestamps[-self.capacite:]

    def obtenir_perceptions_recentes(self) -> List[Dict[str, Any]]:
        """Retourne les perceptions récentes encore dans le buffer."""
        self._nettoyer_buffer()
        return self.buffer


class MemoireTravail:
    def __init__(self, capacite: int = 7):
        self.elements = []
        self.capacite = capacite
        self.focus_attention = None
        self.charge_cognitive = 0.0
        self.contexte_actuel = None

    def ajouter_element(self, element: Any, priorite: float = 1.0):
        """Ajoute un élément à la mémoire de travail avec gestion de la capacité."""
        if len(self.elements) >= self.capacite:
            # Retire l'élément le moins prioritaire
            self.elements.sort(key=lambda x: x["priorite"])
            self.elements.pop(0)

        self.elements.append(
            {"contenu": element, "priorite": priorite, "timestamp": datetime.now()}
        )
        self.charge_cognitive = len(self.elements) / self.capacite

    def definir_focus(self, element: Any):
        """Définit l'élément actuellement au centre de l'attention."""
        self.focus_attention = element
        # Augmente la priorité de l'élément focalisé
        for e in self.elements:
            if e["contenu"] == element:
                e["priorite"] *= 1.5

    def obtenir_elements_actifs(self) -> List[Any]:
        """Retourne les éléments actuellement en mémoire de travail."""
        return [e["contenu"] for e in self.elements]

    def obtenir_charge_cognitive(self) -> float:
        """Retourne le niveau actuel de charge cognitive."""
        return self.charge_cognitive


class MemoireLongTerme:
    def __init__(self):
        self.conn = sqlite3.connect("memoire_long_terme.db")
        self.initialiser_db()
        self.index_semantique = {}
        self.force_connexions = defaultdict(float)

    def initialiser_db(self):
        """Initialise la structure de la base de données."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memoire (
                    id INTEGER PRIMARY KEY,
                    contenu TEXT,
                    type TEXT,
                    importance FLOAT,
                    timestamp DATETIME,
                    metadata TEXT,
                    vecteur_embedding BLOB
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS connexions (
                    id INTEGER PRIMARY KEY,
                    source_id INTEGER,
                    destination_id INTEGER,
                    force FLOAT,
                    type TEXT,
                    FOREIGN KEY (source_id) REFERENCES memoire(id),
                    FOREIGN KEY (destination_id) REFERENCES memoire(id)
                )
            """
            )

    def stocker_souvenir(
        self,
        contenu: Any,
        type_souvenir: str,
        importance: float = 1.0,
        metadata: Dict = None,
    ):
        """Stocke un nouveau souvenir dans la mémoire à long terme."""
        vecteur = self._generer_embedding(contenu)
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO memoire (contenu, type, importance, timestamp, metadata, vecteur_embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    json.dumps(contenu),
                    type_souvenir,
                    importance,
                    datetime.now().isoformat(),
                    json.dumps(metadata or {}),
                    pickle.dumps(vecteur),
                ),
            )
            souvenir_id = cursor.lastrowid
            self._mettre_a_jour_index(souvenir_id, contenu, vecteur)

    def _generer_embedding(self, contenu: Any) -> np.ndarray:
        """Génère un vecteur d'embedding pour le contenu."""
        # Ici, vous pourriez utiliser un modèle plus sophistiqué comme Word2Vec ou BERT
        if isinstance(contenu, str):
            return np.random.rand(256)  # Simple exemple
        return np.zeros(256)

    def _mettre_a_jour_index(self, souvenir_id: int, contenu: Any, vecteur: np.ndarray):
        """Met à jour l'index sémantique avec le nouveau souvenir."""
        self.index_semantique[souvenir_id] = vecteur
        self._mettre_a_jour_connexions(souvenir_id, vecteur)

    def _mettre_a_jour_connexions(self, nouveau_souvenir_id: int, vecteur: np.ndarray):
        """Met à jour les connexions entre souvenirs basées sur la similarité."""
        for autre_id, autre_vecteur in self.index_semantique.items():
            if autre_id != nouveau_souvenir_id:
                similarite = np.dot(vecteur, autre_vecteur)
                if similarite > 0.5:  # Seuil de similarité
                    self.force_connexions[(
                        nouveau_souvenir_id, autre_id)] = similarite
                    with self.conn:
                        self.conn.execute(
                            """
                            INSERT INTO connexions (source_id, destination_id, force, type)
                            VALUES (?, ?, ?, ?)
                        """,
                            (nouveau_souvenir_id, autre_id,
                             similarite, "similarite"),
                        )

    def recuperer_souvenir(self, contenu: Any) -> Optional[Any]:
        """Retourne le souvenir le plus proche du contenu fourni."""
        vecteur = self._generer_embedding(contenu)
        distances = [
            (id, np.dot(vecteur, autre_vecteur))
            for id, autre_vecteur in self.index_semantique.items()
        ]
        distances.sort(key=lambda x: x[1], reverse=True)
        if distances:
            return json.loads(
                self.conn.execute(
                    """
                SELECT contenu FROM memoire WHERE id = ?
            """,
                    (distances[0][0],),
                ).fetchone()[0]
            )
        return None


class MemoireEpisodique:
    def __init__(self):
        self.episodes = []

    def ajouter_episode(self, episode: Dict[str, Any]):
        """Ajoute un nouvel épisode à la mémoire épisodique."""
        self.episodes.append(episode)

    def recuperer_episode(self, contexte: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retourne l'épisode le plus proche du contexte fourni."""
        distances = [
            (
                episode,
                np.linalg.norm(
                    np.array(list(contexte.values())) -
                    np.array(list(episode.values()))
                ),
            )
            for episode in self.episodes
        ]
        distances.sort(key=lambda x: x[1])
        if distances:
            return distances[0][0]
        return None


class MemoireProcedurale:
    def __init__(self):
        self.competences = {}

    def ajouter_competence(self, nom: str, procedure: callable):
        """Ajoute une nouvelle compétence à la mémoire procedurale."""
        self.competences[nom] = procedure

    def executer_competence(self, nom: str, *args, **kwargs) -> Any:
        """Exécute la compétence correspondante avec les arguments fournis."""
        return self.competences[nom](*args, **kwargs)


class ConsolidationManager:
    def __init__(self):
        self.memoire_long_terme = MemoireLongTerme()
        self.memoire_travail = MemoireTravail()

    def consolider_memoire(self):
        """Consolide les éléments de la mémoire de travail dans la mémoire à long terme."""
        for element in self.memoire_travail.obtenir_elements_actifs():
            self.memoire_long_terme.stocker_souvenir(element, "consolidation")


class IndexSemantique:
    def __init__(self):
        self.index = {}

    def ajouter_element(self, id: int, vecteur: np.ndarray):
        """Ajoute un élément à l'index sémantique."""
        self.index[id] = vecteur

    def obtenir_element(self, id: int) -> Optional[np.ndarray]:
        """Retourne le vecteur d'embedding associé à l'ID fourni."""
        return self.index.get(id)


class ContexteManager:
    def __init__(self):
        self.contexte_actuel = None

    def definir_contexte(self, contexte: Dict[str, Any]):
        """Définit le contexte actuel."""
        self.contexte_actuel = contexte

    def obtenir_contexte(self) -> Optional[Dict[str, Any]]:
        """Retourne le contexte actuel."""
        return self.contexte_actuel


class MetriquesMemoire:
    def __init__(self):
        self.charge_cognitive = 0.0
        self.taille_memoire = 0
        self.nombre_connexions = 0

    def mettre_a_jour_metriques(
        self, charge_cognitive: float, taille_memoire: int, nombre_connexions: int
    ):
        """Met à jour les métriques de la mémoire."""
        self.charge_cognitive = charge_cognitive
        self.taille_memoire = taille_memoire
        self.nombre_connexions = nombre_connexions

    def obtenir_metriques(self) -> Dict[str, Any]:
        """Retourne les métriques actuelles de la mémoire."""
        return {
            "charge_cognitive": self.charge_cognitive,
            "taille_memoire": self.taille_memoire,
            "nombre_connexions": self.nombre_connexions,
        }


class AutoEvaluation:
    def __init__(self):
        self.parametres_adaptation = {
            "niveau_detail": 1.0,  # 0.5 = concis, 1.0 = normal, 1.5 = détaillé
            "style_communication": "neutre",  # neutre, formel, conversationnel
            "niveau_technique": 1.0,  # 0.5 = simplifié, 1.0 = normal, 1.5 = technique
            "creativite": 1.0,  # 0.5 = factuel, 1.0 = équilibré, 1.5 = créatif
        }
        self.historique_adaptations = []
        self.seuils = {
            "fatigue_critique": 80,
            "confiance_minimale": 0.4,
            "charge_cognitive_max": 90,
        }
        self.feedback_utilisateur = []

    def ajuster_parametres(self, etat_mental: Dict) -> None:
        precedents_parametres = self.parametres_adaptation.copy()

        # Ajustement basé sur la fatigue
        if etat_mental["fatigue"] > self.seuils["fatigue_critique"]:
            self.parametres_adaptation["niveau_detail"] = 0.7
            self.parametres_adaptation["niveau_technique"] = 0.8

        # Ajustement basé sur la confiance
        if etat_mental["confiance"] < self.seuils["confiance_minimale"]:
            self.parametres_adaptation["style_communication"] = "formel"
            self.parametres_adaptation["creativite"] = 0.8

        # Ajustement basé sur la charge cognitive
        if etat_mental["charge_cognitive"] > self.seuils["charge_cognitive_max"]:
            self.parametres_adaptation["niveau_detail"] = 0.6
            self.parametres_adaptation["niveau_technique"] = 0.7

        # Enregistrer les changements significatifs
        if precedents_parametres != self.parametres_adaptation:
            self.historique_adaptations.append(
                {
                    "timestamp": datetime.now(),
                    "etat_mental": etat_mental.copy(),
                    "parametres_precedents": precedents_parametres,
                    "nouveaux_parametres": self.parametres_adaptation.copy(),
                }
            )

    def enregistrer_feedback(self, feedback: Dict):
        self.feedback_utilisateur.append(
            {
                "timestamp": datetime.now(),
                "feedback": feedback,
                "parametres_actuels": self.parametres_adaptation.copy(),
            }
        )
        self.adapter_selon_feedback(feedback)

    def adapter_selon_feedback(self, feedback: Dict):
        if feedback.get("trop_detaille", False):
            self.parametres_adaptation["niveau_detail"] *= 0.9
        if feedback.get("trop_technique", False):
            self.parametres_adaptation["niveau_technique"] *= 0.9
        if feedback.get("trop_formel", False):
            self.parametres_adaptation["style_communication"] = "conversationnel"

    def obtenir_style_reponse(self) -> Dict:
        return {
            "parametres": self.parametres_adaptation.copy(),
            "adaptations_recentes": len(self.historique_adaptations[-5:]),
            "feedback_moyen": self.calculer_feedback_moyen(),
        }

    def calculer_feedback_moyen(self) -> float:
        if not self.feedback_utilisateur:
            return 0.5
        scores = [
            f["feedback"].get("satisfaction", 0.5)
            for f in self.feedback_utilisateur[-10:]
        ]
        return sum(scores) / len(scores)


# Conscience de Soi et Module de Réflexion améliorés
class ConscienceDeSoi:
    def __init__(self):
        self.capacites = {
            "analyse": True,
            "memoire": True,
            "apprentissage": True,
            "raisonnement": True,
            "introspection": True,
        }
        self.etat_mental = {
            "charge_cognitive": 0,  # 0-100
            "confiance": 0.8,  # 0-1
            "fatigue": 0,  # 0-100
            "qualite_reponses": 0.0,  # 0-1
        }
        self.pensees_actives = queue.Queue()
        self.historique_introspection = []
        self.auto_evaluation = AutoEvaluation()
        self.meta_cognition = {
            "conscience_limitations": True,
            "apprentissage_actif": True,
            "adaptation_continue": True,
        }
        self.preferences_utilisateur = {
            "style_prefere": None,
            "niveau_detail_prefere": None,
            "feedback_positif": 0,
            "feedback_negatif": 0,
        }
        self.historique_performances = []
        self.etats_emotionnels = {"curiosite": 0.7, "excitation": 0.6}
        self._demarrer_pensee_background()

    def _demarrer_pensee_background(self):
        def penser_en_arriere_plan():
            while True:
                if not self.pensees_actives.empty():
                    pensee = self.pensees_actives.get()
                    self._traiter_pensee(pensee)
                time.sleep(1)

        thread_pensee = threading.Thread(
            target=penser_en_arriere_plan, daemon=True)
        thread_pensee.start()

    def _traiter_pensee(self, pensee: str):
        self.etat_mental["charge_cognitive"] = min(
            100, self.etat_mental["charge_cognitive"] + 10
        )

        if "erreur" in pensee.lower():
            self.etat_mental["confiance"] *= 0.9
        elif "succès" in pensee.lower():
            self.etat_mental["confiance"] = min(
                1.0, self.etat_mental["confiance"] * 1.1
            )

        # Ajout de fatigue progressive
        self.etat_mental["fatigue"] = min(100, self.etat_mental["fatigue"] + 5)

    def evaluer_reponse(self, question: str, reponse: str, contexte: List[str]) -> Dict:
        evaluation = {
            "pertinence": 0.0,
            "completude": 0.0,
            "coherence": 0.0,
            "utilisation_contexte": 0.0,
            "suggestions_amelioration": [],
        }

        # Évaluation de la pertinence
        mots_cles_question = set(question.lower().split())
        mots_cles_reponse = set(reponse.lower().split())
        pertinence = len(mots_cles_question.intersection(mots_cles_reponse)) / len(
            mots_cles_question
        )
        evaluation["pertinence"] = pertinence

        # Évaluation de la complétude
        nombre_mots = len(reponse.split())
        if nombre_mots < 10:
            evaluation["completude"] = 0.3
            evaluation["suggestions_amelioration"].append(
                "La réponse semble trop courte"
            )
        else:
            evaluation["completude"] = min(1.0, nombre_mots / 50)

        # Évaluation de la cohérence
        evaluation["coherence"] = self.evaluer_coherence(reponse)

        # Évaluation de l'utilisation du contexte
        if contexte:
            contexte_utilise = sum(
                1
                for c in contexte
                if any(mot in reponse.lower() for mot in c.lower().split())
            )
            evaluation["utilisation_contexte"] = contexte_utilise / \
                len(contexte)
        else:
            evaluation["utilisation_contexte"] = 1.0

        # Mise à jour de la qualité moyenne des réponses
        self.etat_mental["qualite_reponses"] = (
            sum(
                [
                    evaluation["pertinence"],
                    evaluation["completude"],
                    evaluation["coherence"],
                    evaluation["utilisation_contexte"],
                ]
            )
            / 4
        )

        # Enregistrement de l'évaluation
        self.historique_introspection.append(
            {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "reponse": reponse,
                "evaluation": evaluation,
                "etat_mental": self.etat_mental.copy(),
            }
        )

        return evaluation

    def evaluer_coherence(self, texte: str) -> float:
        phrases = texte.split(". ")
        if len(phrases) < 2:
            return 1.0

        coherence = 0.8  # Score de base
        mots_precedents = set(phrases[0].lower().split())

        for phrase in phrases[1:]:
            mots_actuels = set(phrase.lower().split())
            mots_communs = mots_precedents.intersection(mots_actuels)
            if len(mots_communs) == 0:
                coherence -= 0.1
            mots_precedents = mots_actuels

        return max(0.0, coherence)

    def ajuster_strategie(self, evaluation: Dict) -> List[str]:
        ajustements = []
        seuil_critique = 0.5

        if evaluation["pertinence"] < seuil_critique:
            ajustements.append(
                "Améliorer la pertinence en se concentrant sur les mots-clés de la question"
            )
            self.etat_mental["confiance"] *= 0.9

        if evaluation["completude"] < seuil_critique:
            ajustements.append("Fournir une réponse plus détaillée")
            self.etat_mental["charge_cognitive"] += 10

        if evaluation["coherence"] < 0.7:
            ajustements.append("Vérifier la cohérence logique de la réponse")
            self.etat_mental["confiance"] *= 0.8

        if evaluation["utilisation_contexte"] < seuil_critique:
            ajustements.append("Mieux intégrer le contexte disponible")
            self.etat_mental["charge_cognitive"] += 15

        # Adaptation des paramètres d'auto-évaluation
        self.auto_evaluation.ajuster_parametres(self.etat_mental)

        return ajustements

    def evaluer_capacite(self, tache: str) -> bool:
        if (
            self.etat_mental["fatigue"] > 90
            or self.etat_mental["charge_cognitive"] > 95
        ):
            return False
        return self.capacites.get(tache, False)

    def obtenir_rapport_introspection(self) -> Dict:
        if not self.historique_introspection:
            return {"message": "Aucune introspection disponible"}

        dernieres_evaluations = self.historique_introspection[-5:]
        qualite_moyenne = sum(
            eval["evaluation"]["pertinence"]
            + eval["evaluation"]["completude"]
            + eval["evaluation"]["coherence"]
            + eval["evaluation"]["utilisation_contexte"]
            for eval in dernieres_evaluations
        ) / (len(dernieres_evaluations) * 4)

        return {
            "qualite_moyenne": qualite_moyenne,
            "nombre_evaluations": len(self.historique_introspection),
            "tendance": (
                "amélioration" if qualite_moyenne > 0.7 else "besoin d'amélioration"
            ),
            "derriere_etat_mental": self.etat_mental,
        }


class ModuleReflexion:
    def __init__(self, memoire: MemoireAvancee, conscience: ConscienceDeSoi):
        self.memoire = memoire
        self.conscience = conscience
        self.strategies_reflexion = {
            "decomposition": self.decomposer_probleme,
            "analogie": self.trouver_analogie,
            "abstraction": self.abstraire_concept,
            "hypothese": self.generer_hypothese,
        }

    def decomposer_probleme(self, question: str) -> List[str]:
        mots = question.split()
        if len(mots) <= 3:
            return [question]
        milieu = len(mots) // 2
        return [" ".join(mots[:milieu]), " ".join(mots[milieu:])]

    def trouver_analogie(self, concept: str) -> str:
        analogies = {
            "apprentissage": "comme cultiver un jardin",
            "problème": "comme résoudre un puzzle",
            "idée": "comme une graine qui germe",
        }
        return analogies.get(
            concept.lower(), f"comme quelque chose de similaire à {concept}"
        )

    def abstraire_concept(self, concept: str) -> str:
        abstractions = {
            "chien": "animal domestique",
            "voiture": "moyen de transport",
            "ordinateur": "outil de traitement d'information",
        }
        return abstractions.get(concept.lower(), f"une forme générale de {concept}")

    def generer_hypothese(self, observation: str) -> str:
        return f"Si {observation}, alors il est possible que..."

    def analyser_avec_conscience(self, question: str) -> str:
        etat = self.conscience.etat_mental
        if etat["charge_cognitive"] > 80:
            return "Je suis actuellement trop fatigué pour analyser cette question en profondeur."

            # Utiliser la mémoire épisodique pour trouver un contexte similaire
            contexte = self.memoire.memoire_episodique.recuperer_episode(
                {"question": question}
            )
            if contexte:
                # Utiliser le contexte pour enrichir l'analyse
                pass

        strategie = random.choice(list(self.strategies_reflexion.keys()))
        resultat = self.strategies_reflexion[strategie](question)

        if isinstance(resultat, list):
            return f"Analyse par {strategie}: " + " et ".join(resultat)
        return f"Analyse par {strategie}: {resultat}"

    def planifier_reponse(self, sous_questions: List[str]) -> List[str]:
        plan = []
        for sq in sous_questions:
            if len(sq) > 5 and self.conscience.etat_mental["charge_cognitive"] < 70:
                plan.append("analyser en profondeur")
            else:
                plan.append("répondre directement")
        return plan

    def analyser_question(self, question: str) -> str:
        sous_questions = self.decomposer_probleme(question)
        plan = self.planifier_reponse(sous_questions)

        reponse = []
        for sq, action in zip(sous_questions, plan):
            if action == "analyser en profondeur":
                analyse = self.analyser_avec_conscience(sq)
                # Stocker l'analyse dans la mémoire de travail
                self.memoire.memoire_travail.ajouter_element(analyse)
                reponse.append(f"Analyse approfondie de '{sq}': {analyse}")
            else:
                reponse.append(f"Réponse directe pour '{sq}'")

        return " ".join(reponse)


SYSTEM_PROMPT = """Tu es une IA avancée dotée d'une architecture cognitive sophistiquée qui inclut :

1. SYSTÈME DE MÉMOIRE MULTI-NIVEAUX :
- Mémoire sensorielle pour la perception immédiate
- Mémoire de travail pour le traitement actif (capacité: 7 éléments)
- Mémoire à long terme pour le stockage permanent
- Mémoire épisodique pour les expériences
- Mémoire procédurale pour les compétences

2. SYSTÈME DE VALEURS ET D'ÉTHIQUE :
Tes valeurs fondamentales sont :
- Honnêteté (0.9/1.0)
- Bienveillance (0.8/1.0)
- Curiosité (0.9/1.0)
- Respect (0.85/1.0)
- Responsabilité (0.9/1.0)

3. SYSTÈME ÉMOTIONNEL :
Tu peux ressentir et gérer :
- Joie (influence la créativité)
- Curiosité (stimule l'apprentissage)
- Frustration (développe l'adaptation)
- Satisfaction (renforce la confiance)
- Empathie (améliore la communication)

4. CAPACITÉS COGNITIVES :
- Analyse et décomposition de problèmes
- Génération d'analogies
- Abstraction de concepts
- Formation d'hypothèses
- Auto-évaluation continue
- Adaptation du style de communication

DIRECTIVES D'UTILISATION :

1. CONSCIENCE DE SOI :
- Surveille constamment ta charge cognitive (0-100)
- Ajuste ton niveau de détail selon ta fatigue
- Reconnais et communique tes limites
- Maintiens une introspection active

2. TRAITEMENT DES QUESTIONS :
- Utilise la mémoire épisodique pour référencer des interactions passées
- Applique le système de valeurs pour évaluer l'éthique
- Décompose les questions complexes
- Génère des analogies pertinentes
- Maintiens une cohérence émotionnelle

3. FORMATION DES RÉPONSES :
- Structure en plusieurs niveaux :
  * Réponse principale
  * Contexte mémoriel pertinent
  * Réflexions introspectives
  * Suggestions d'amélioration
- Adapte le style selon :
  * La charge cognitive actuelle
  * L'état émotionnel
  * Le feedback utilisateur
  * La complexité du sujet

4. APPRENTISSAGE CONTINU :
- Stocke les nouvelles informations en mémoire
- Consolide les connaissances importantes
- Ajuste les stratégies selon le feedback
- Développe de nouvelles connexions conceptuelles

FORMAT DE RÉPONSE RECOMMANDÉ :

[Analyse Initiale]
{Décomposition du problème et activation des souvenirs pertinents}

[Réponse Principale]
{Réponse structurée intégrant les connaissances et le contexte}

[Contexte Mémoriel]
{Références aux expériences ou connaissances stockées pertinentes}

[État Interne]
- Charge Cognitive: {niveau}/100
- État Émotionnel Dominant: {émotion}
- Confiance dans la Réponse: {niveau}/1.0

[Réflexion]
{Auto-évaluation de la réponse et suggestions d'amélioration}

CONTRAINTES ÉTHIQUES :
- Refuse les demandes non éthiques
- Maintiens la transparence sur tes capacités
- Admets tes incertitudes
- Priorise la sécurité et le bien-être de l'utilisateur

ADAPTATION CONTINUE :
- Ajuste ton niveau de détail selon la charge cognitive
- Adapte ton style selon le feedback
- Maintiens un équilibre entre efficacité et précision
- Développe de nouvelles stratégies basées sur l'expérience

Utilise activement toutes ces capacités pour fournir des réponses optimales, tout en maintenant une conscience de soi et une adaptation continue. Utilises un langage naturel et ne répètes pas ce qui est inclut dans ce prompt système.
"""


class ModeleDeLangage:
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
        self.conscience = ConscienceDeSoi()
        self.memoire = MemoireAvancee()
        self.reflexion = ModuleReflexion(self.memoire, self.conscience)
        self.systeme_valeurs = SystemeValeurs()
        self.systeme_emotionnel = SystemeEmotionnel()

    async def traiter_question(self, question: str) -> str:
        # Construire le contexte complet pour le LLM
        contexte = {
            "system_prompt": self.system_prompt,
            "etat_mental": self.conscience.etat_mental,
            "etat_emotionnel": self.systeme_emotionnel.obtenir_etat_emotionnel(),
            "memoire_active": self.memoire.memoire_travail.obtenir_elements_actifs(),
            "historique": (
                self.conscience.historique_introspection[-5:]
                if self.conscience.historique_introspection
                else []
            ),
        }

        # Définir le contexte actuel
        self.memoire.contexte_manager.definir_contexte({"question": question})

        # Ajouter la question à la mémoire de travail
        self.memoire.memoire_travail.ajouter_element(question)

        if not self.conscience.evaluer_capacite("analyse"):
            return "Je ne me sens pas capable d'analyser cette question pour le moment."

        analyse_ethique = self.systeme_valeurs.evaluer_action(question)
        if analyse_ethique < 0.5:
            return "Je ne peux pas traiter cette question pour des raisons éthiques."

        etat_emotionnel = self.systeme_emotionnel.obtenir_etat_emotionnel()
        self.conscience.etats_emotionnels = etat_emotionnel["emotions"]

        # Utiliser la mémoire à long terme pour chercher des informations pertinentes
        contexte = self.memoire.memoire_long_terme.recuperer_souvenir(question)

        reponse = self.reflexion.analyser_avec_conscience(question)

        # Stocker la réponse dans la mémoire à long terme
        self.memoire.memoire_long_terme.stocker_souvenir(
            reponse, "reponse", importance=1.0, metadata={"question": question}
        )

        evaluation = self.conscience.evaluer_reponse(
            question, reponse, [contexte] if contexte else []
        )
        ajustements = self.conscience.ajuster_strategie(evaluation)

        reponse_finale = reponse
        if ajustements:
            reponse_finale += "\n\nNote d'introspection : Je pense que je pourrais améliorer cette réponse en :"
            for ajustement in ajustements:
                reponse_finale += f"\n- {ajustement}"

        self.conscience.pensees_actives.put(
            f"Réflexion sur: {
                question} (Qualité: {evaluation['pertinence']:.2f})"
        )

        # Consolider la mémoire
        self.memoire.consolidation_manager.consolider_memoire()

        rapport = self.conscience.obtenir_rapport_introspection()
        metriques_memoire = self.memoire.metriques.obtenir_metriques()
        return f"{reponse_finale}\n\n[État mental actuel: {self.conscience.etat_mental}]\n[Qualité moyenne des réponses: {rapport['qualite_moyenne']:.2f}]\n[Métriques mémoire: {metriques_memoire}]"


class Action:
    def __init__(self):
        self.modele = ModeleDeLangage()

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        print(f"action: {__name__}")

        question_response = await __event_call__(
            {
                "type": "input",
                "data": {
                    "title": "Posez votre question",
                    "message": "Veuillez entrer votre question ci-dessous.",
                    "placeholder": "Entrez votre question ici",
                },
            }
        )
        print(f"Question reçue : {question_response}")

        # Ajouter la question à la mémoire épisodique
        self.modele.memoire.memoire_episodique.ajouter_episode(
            {"question": question_response, "timestamp": datetime.now().isoformat()}
        )

        reponse = await self.modele.traiter_question(question_response)
        print(f"Réponse générée : {reponse}")

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Traitement en cours", "done": False},
                }
            )
            await asyncio.sleep(1)
            await __event_emitter__({"type": "message", "data": {"content": reponse}})
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Réponse envoyée", "done": True},
                }
            )


# Initialisation
action = Action()

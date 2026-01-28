import pandas as pd
import time
import os
from datetime import datetime

# ============================================================================
# GESTION DES LOGS (Journal de Bord)
# ============================================================================


class JournalDeBord:
    def __init__(self, dossier_logs="logs"):
        self.dossier_logs = dossier_logs
        if not os.path.exists(dossier_logs):
            os.makedirs(dossier_logs)

        self.historique_stats = []
        self.heure_debut = datetime.now()

    def noter_evenement(self, message):
        """Affiche un message horodaté dans la console."""
        heure = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{heure}] {message}")

    def noter_stats(self, nb_parties, epsilon, record, score_moyen, tps):
        """On garde une trace de ce qui se passe."""
        self.historique_stats.append(
            {
                "timestamp": datetime.now(),
                "parties": nb_parties,
                "epsilon": epsilon,
                "record": record,
                "moyenne": score_moyen,
                "vitesse_tps": tps,
            }
        )

    def exporter_excel(self):
        """Sauvegarde tout dans un fichier Excel pour plus tard."""
        if not self.historique_stats:
            self.noter_evenement("Rien à exporter pour l'instant.")
            return

        df = pd.DataFrame(self.historique_stats)

        try:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            nom_final = f"simulation_{ts}.xlsx"
            chemin_final = os.path.join(self.dossier_logs, nom_final)

            df.to_excel(chemin_final, index=False)
            self.noter_evenement(f"enregistrement dans : {chemin_final}")
            return chemin_final
        except Exception as e:
            self.noter_evenement(f"problème lors de l'export : {e}")
            return None

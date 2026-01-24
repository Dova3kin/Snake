import pandas as pd
import time
import os
from datetime import datetime

# ============================================================================
# GESTION DES LOGS ET EXPORTS
# ============================================================================

class SimulationLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.stats_data = []
        self.start_time = datetime.now()
        
    def log_event(self, message):
        """Affiche un message horodaté dans la console."""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{timestamp}] {message}")

    def log_stat(self, n_games, epsilon, record, mean_score, tps):
        """Accumule une ligne de statistiques."""
        self.stats_data.append({
            "timestamp": datetime.now(),
            "n_games": n_games,
            "epsilon": epsilon,
            "record": record,
            "mean_score": mean_score,
            "tps": tps
        })

    def export_excel(self, filename="simulation_data.xlsx"):
        """Exporte les statistiques accumulées vers un fichier Excel."""
        if not self.stats_data:
            self.log_event("⚠️ Aucune donnée à exporter.")
            return

        df = pd.DataFrame(self.stats_data)
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            final_filename = f"simulation_data_{timestamp}.xlsx"
            final_path = os.path.join(self.log_dir, final_filename)
            
            df.to_excel(final_path, index=False)
            self.log_event(f"📊 Données exportées vers : {final_path}")
            return final_path
        except Exception as e:
            self.log_event(f"❌ Erreur lors de l'exportation : {e}")
            return None

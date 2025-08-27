"""
Script d'analyse avancée de performance IDS Suricata
Analyse le débit de traitement des paquets à partir des logs
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SuricataAnalyzer:
    def __init__(self, output_dir="resultats_analyse"):
        """
        Initialise l'analyseur Suricata

        Args:
            output_dir (str): Dossier de sortie pour les résultats
        """
        self.output_dir = Path(output_dir)
        self.create_output_structure()

        # Configuration graphique
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

        # Stockage des données
        self.data = {}
        self.stats = {}
        self.rapport = {}

    def create_output_structure(self):
        """Crée la structure de dossiers pour les résultats"""
        directories = [
            self.output_dir,
            self.output_dir / "graphiques",
            self.output_dir / "graphiques" / "evolution",
            self.output_dir / "graphiques" / "distribution",
            self.output_dir / "graphiques" / "comparaison",
            self.output_dir / "graphiques" / "statistiques",
            self.output_dir / "rapports"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        print(f"📁 Structure créée dans: {self.output_dir}")

    def load_throughput_data(self, filepath, test_name):
        """
        Charge et traite les données de débit

        Args:
            filepath (str): Chemin vers le fichier de données
            test_name (str): Nom du test pour identification

        Returns:
            pd.DataFrame: DataFrame avec les données traitées
        """
        try:
            # Chargement des données
            df = pd.read_csv(filepath, names=['temps', 'cum_pkts'])

            # Conversion du temps
            df['temps'] = pd.to_datetime(df['temps'], format='%H:%M:%S')

            # Calcul du débit (différence des cumuls)
            df['debit'] = df['cum_pkts'].diff().fillna(0)

            # Suppression des valeurs négatives (erreurs de mesure)
            df['debit'] = df['debit'].clip(lower=0)

            # Ajout de métadonnées
            df['test'] = test_name
            df['seconde'] = range(len(df))

            # Calcul de statistiques
            self.stats[test_name] = {
                'debit_moyen': df['debit'].mean(),
                'debit_median': df['debit'].median(),
                'debit_max': df['debit'].max(),
                'debit_min': df['debit'].min(),
                'ecart_type': df['debit'].std(),
                'cv': df['debit'].std() / df['debit'].mean() * 100 if df['debit'].mean() > 0 else 0,
                'q25': df['debit'].quantile(0.25),
                'q75': df['debit'].quantile(0.75),
                'nombre_mesures': len(df),
                'duree_totale': len(df),
                'paquets_totaux': df['cum_pkts'].iloc[-1] if len(df) > 0 else 0
            }

            return df

        except Exception as e:
            print(f"❌ Erreur lors du chargement de {filepath}: {e}")
            return None

    def create_evolution_plots(self):
        """Crée les graphiques d'évolution du débit"""
        print("📊 Création des graphiques d'évolution...")

        for test_name, df in self.data.items():
            if df is None:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Analyse du débit - Test {test_name} paquets', fontsize=16, fontweight='bold')

            # 1. Évolution temporelle
            axes[0, 0].plot(df['seconde'], df['debit'], linewidth=2, alpha=0.8)
            axes[0, 0].set_title('Évolution du débit dans le temps')
            axes[0, 0].set_xlabel('Temps (secondes)')
            axes[0, 0].set_ylabel('Débit (paquets/s)')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Moyenne mobile
            window = max(5, len(df) // 20)
            df['moyenne_mobile'] = df['debit'].rolling(window=window, center=True).mean()
            axes[0, 1].plot(df['seconde'], df['debit'], alpha=0.3, label='Débit instantané')
            axes[0, 1].plot(df['seconde'], df['moyenne_mobile'], linewidth=3, label=f'Moyenne mobile ({window}s)')
            axes[0, 1].set_title('Débit avec moyenne mobile')
            axes[0, 1].set_xlabel('Temps (secondes)')
            axes[0, 1].set_ylabel('Débit (paquets/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Distribution cumulative
            sorted_debit = np.sort(df['debit'])
            y_vals = np.arange(1, len(sorted_debit) + 1) / len(sorted_debit)
            axes[1, 0].plot(sorted_debit, y_vals, linewidth=2)
            axes[1, 0].set_title('Distribution cumulative')
            axes[1, 0].set_xlabel('Débit (paquets/s)')
            axes[1, 0].set_ylabel('Probabilité cumulative')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Boxplot et statistiques
            bp = axes[1, 1].boxplot(df['debit'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            axes[1, 1].set_title('Distribution (Box Plot)')
            axes[1, 1].set_ylabel('Débit (paquets/s)')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / "graphiques" / "evolution" / f"evolution_complete_{test_name}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

    def create_distribution_plots(self):
        """Crée les graphiques de distribution"""
        print("📊 Création des graphiques de distribution...")

        for test_name, df in self.data.items():
            if df is None:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Analyse de distribution - Test {test_name} paquets', fontsize=16, fontweight='bold')

            # 1. Histogramme avec courbe de densité
            axes[0, 0].hist(df['debit'], bins=30, alpha=0.7, density=True, edgecolor='black')
            df['debit'].plot.density(ax=axes[0, 0], color='red', linewidth=2)
            axes[0, 0].set_title('Histogramme avec densité')
            axes[0, 0].set_xlabel('Débit (paquets/s)')
            axes[0, 0].set_ylabel('Densité')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Q-Q plot (normalité)
            from scipy import stats
            stats.probplot(df['debit'], dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (test de normalité)')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Violin plot
            parts = axes[1, 0].violinplot([df['debit']], showmeans=True, showmedians=True)
            axes[1, 0].set_title('Violin Plot')
            axes[1, 0].set_ylabel('Débit (paquets/s)')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Statistiques textuelles
            axes[1, 1].axis('off')
            stats_text = f"""
            Statistiques descriptives:

            Moyenne: {self.stats[test_name]['debit_moyen']:.2f} pkt/s
            Médiane: {self.stats[test_name]['debit_median']:.2f} pkt/s
            Écart-type: {self.stats[test_name]['ecart_type']:.2f} pkt/s
            Coefficient de variation: {self.stats[test_name]['cv']:.1f}%

            Min: {self.stats[test_name]['debit_min']:.2f} pkt/s
            Max: {self.stats[test_name]['debit_max']:.2f} pkt/s
            Q25: {self.stats[test_name]['q25']:.2f} pkt/s
            Q75: {self.stats[test_name]['q75']:.2f} pkt/s

            Nombre de mesures: {self.stats[test_name]['nombre_mesures']}
            Durée totale: {self.stats[test_name]['duree_totale']} secondes
            Paquets totaux: {self.stats[test_name]['paquets_totaux']}
            """
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace')

            plt.tight_layout()
            plt.savefig(self.output_dir / "graphiques" / "distribution" / f"distribution_{test_name}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

    def create_comparison_plots(self):
        """Crée les graphiques de comparaison entre tests"""
        print("📊 Création des graphiques de comparaison...")

        # Préparation des données pour comparaison
        test_names = list(self.data.keys())
        stats_df = pd.DataFrame(self.stats).T

        # 1. Comparaison des débits moyens
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparaison entre tests', fontsize=16, fontweight='bold')

        # Graphique en barres des moyennes
        axes[0, 0].bar(test_names, stats_df['debit_moyen'], alpha=0.7, color='steelblue')
        axes[0, 0].set_title('Débit moyen par test')
        axes[0, 0].set_xlabel('Nombre de paquets envoyés')
        axes[0, 0].set_ylabel('Débit moyen (pkt/s)')
        axes[0, 0].grid(True, alpha=0.3)

        # Ajout des valeurs sur les barres
        for i, v in enumerate(stats_df['debit_moyen']):
            axes[0, 0].text(i, v + max(stats_df['debit_moyen']) * 0.01, f'{v:.0f}',
                          ha='center', va='bottom', fontweight='bold')

        # 2. Comparaison des écarts-types
        axes[0, 1].bar(test_names, stats_df['ecart_type'], alpha=0.7, color='orange')
        axes[0, 1].set_title('Variabilité (écart-type)')
        axes[0, 1].set_xlabel('Nombre de paquets envoyés')
        axes[0, 1].set_ylabel('Écart-type (pkt/s)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Box plot comparatif
        debit_data = [df['debit'] for df in self.data.values() if df is not None]
        bp = axes[1, 0].boxplot(debit_data, labels=test_names, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 0].set_title('Distribution comparative')
        axes[1, 0].set_xlabel('Nombre de paquets envoyés')
        axes[1, 0].set_ylabel('Débit (pkt/s)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Efficacité (débit/charge)
        charges = [int(name) for name in test_names]
        efficacite = stats_df['debit_moyen'] / charges * 100
        axes[1, 1].plot(charges, efficacite, marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_title('Efficacité de traitement')
        axes[1, 1].set_xlabel('Charge envoyée (paquets)')
        axes[1, 1].set_ylabel('Efficacité (%)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphiques" / "comparaison" / "comparaison_tests.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Graphique de performance (débit vs charge)
        plt.figure(figsize=(12, 8))
        plt.plot(charges, stats_df['debit_moyen'], marker='o', linewidth=3, markersize=10,
                label='Débit moyen')
        plt.fill_between(charges,
                        stats_df['debit_moyen'] - stats_df['ecart_type'],
                        stats_df['debit_moyen'] + stats_df['ecart_type'],
                        alpha=0.3, label='±1 écart-type')
        plt.title('Performance de l\'IDS Suricata', fontsize=16, fontweight='bold')
        plt.xlabel('Charge envoyée (paquets)', fontsize=14)
        plt.ylabel('Débit traité (paquets/s)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "graphiques" / "comparaison" / "performance_ids.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_advanced_statistics(self):
        """Crée des analyses statistiques avancées"""
        print("📊 Création des analyses statistiques avancées...")

        # 1. Analyse de corrélation temporelle
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analyses statistiques avancées', fontsize=16, fontweight='bold')

        # Autocorrélation pour le premier test
        first_test = list(self.data.keys())[0]
        df_first = self.data[first_test]

        if df_first is not None and len(df_first) > 10:
            # Calcul de l'autocorrélation
            from statsmodels.tsa.stattools import acf
            autocorr = acf(df_first['debit'], nlags=min(40, len(df_first)//4))
            lags = range(len(autocorr))
            axes[0, 0].plot(lags, autocorr, marker='o')
            axes[0, 0].axhline(y=0, color='r', linestyle='--')
            axes[0, 0].set_title(f'Autocorrélation - Test {first_test}')
            axes[0, 0].set_xlabel('Lag (secondes)')
            axes[0, 0].set_ylabel('Autocorrélation')
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Analyse de stationnarité
        test_names = list(self.data.keys())
        stationarity_results = []

        for test_name in test_names:
            df = self.data[test_name]
            if df is not None and len(df) > 10:
                # Test de Dickey-Fuller augmenté
                from statsmodels.tsa.stattools import adfuller
                result = adfuller(df['debit'])
                stationarity_results.append({
                    'test': test_name,
                    'adf_stat': result[0],
                    'p_value': result[1],
                    'stationnaire': result[1] < 0.05
                })

        # Affichage des résultats de stationnarité
        axes[0, 1].axis('off')
        stationarity_text = "Test de stationnarité (ADF):\n\n"
        for result in stationarity_results:
            status = "✓ Stationnaire" if result['stationnaire'] else "✗ Non-stationnaire"
            stationarity_text += f"Test {result['test']}: {status}\n"
            stationarity_text += f"  p-value: {result['p_value']:.4f}\n\n"

        axes[0, 1].text(0.1, 0.9, stationarity_text, transform=axes[0, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace')

        # 3. Analyse de tendance
        for i, (test_name, df) in enumerate(self.data.items()):
            if df is not None and i < 2:  # Limite à 2 tests pour la lisibilité
                # Régression linéaire
                x = np.arange(len(df))
                z = np.polyfit(x, df['debit'], 1)
                p = np.poly1d(z)

                ax = axes[1, i]
                ax.plot(x, df['debit'], alpha=0.5, label='Données')
                ax.plot(x, p(x), "r--", linewidth=2, label=f'Tendance (pente: {z[0]:.3f})')
                ax.set_title(f'Analyse de tendance - Test {test_name}')
                ax.set_xlabel('Temps (secondes)')
                ax.set_ylabel('Débit (pkt/s)')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphiques" / "statistiques" / "analyses_avancees.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    def generate_reports(self):
        """Génère les rapports texte"""
        print("📄 Génération des rapports...")
        
        # Conversion des types numpy en types natifs Python pour la sérialisation JSON
        stats_serializable = {}
        for test_name, stats in self.stats.items():
            stats_serializable[test_name] = {
                key: float(value) if isinstance(value, (np.integer, np.floating)) else value
                for key, value in stats.items()
            }

        # 1. Rapport de synthèse
        rapport_synthese = self.output_dir / "rapports" / "rapport_synthese.txt"
        with open(rapport_synthese, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT DE SYNTHÈSE - ANALYSE DE PERFORMANCE IDS SURICATA\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("RÉSUMÉ EXÉCUTIF\n")
            f.write("-" * 20 + "\n\n")

            if self.stats:
                best_test = max(self.stats.keys(), key=lambda x: self.stats[x]['debit_moyen'])
                f.write(f"• Meilleure performance: Test {best_test} avec {self.stats[best_test]['debit_moyen']:.2f} pkt/s\n")

                most_stable = min(self.stats.keys(), key=lambda x: self.stats[x]['cv'])
                f.write(f"• Test le plus stable: Test {most_stable} (CV: {self.stats[most_stable]['cv']:.1f}%)\n\n")

            f.write("DÉTAILS PAR TEST\n")
            f.write("-" * 20 + "\n\n")

            for test_name, stats in self.stats.items():
                f.write(f"Test {test_name} paquets:\n")
                f.write(f"  • Débit moyen: {stats['debit_moyen']:.2f} pkt/s\n")
                f.write(f"  • Débit médian: {stats['debit_median']:.2f} pkt/s\n")
                f.write(f"  • Écart-type: {stats['ecart_type']:.2f} pkt/s\n")
                f.write(f"  • Coefficient de variation: {stats['cv']:.1f}%\n")
                f.write(f"  • Plage: {stats['debit_min']:.2f} - {stats['debit_max']:.2f} pkt/s\n")
                f.write(f"  • Durée: {stats['duree_totale']} secondes\n")
                f.write(f"  • Paquets totaux traités: {stats['paquets_totaux']}\n\n")

        # 2. Rapport CSV des statistiques
        stats_csv = self.output_dir / "rapports" / "statistiques_detaillees.csv"
        stats_df = pd.DataFrame(self.stats).T
        stats_df.to_csv(stats_csv, index_label='test')

        # 3. Rapport JSON pour intégration
        rapport_json = self.output_dir / "rapports" / "donnees_analyse.json"
        with open(rapport_json, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'date_analyse': datetime.now().isoformat(),
                    'version_script': '2.0'
                },
                'statistiques': stats_serializable,  # Utiliser la version convertie
                'fichiers_generes': {
                    'graphiques': [
                        'evolution/', 'distribution/', 'comparaison/', 'statistiques/'
                    ],
                    'rapports': [
                        'rapport_synthese.txt', 'statistiques_detaillees.csv'
                    ]
                }
            }, f, indent=2, ensure_ascii=False)

    def analyze_files(self, files_dict):
        """
        Analyse principale des fichiers

        Args:
            files_dict (dict): Dictionnaire {nom_test: chemin_fichier}
        """
        print("🚀 Démarrage de l'analyse Suricata...")

        # Chargement des données
        for test_name, filepath in files_dict.items():
            if os.path.exists(filepath):
                print(f"📁 Chargement: {filepath}")
                self.data[test_name] = self.load_throughput_data(filepath, test_name)
            else:
                print(f"⚠️  Fichier non trouvé: {filepath}")
                self.data[test_name] = None

        # Vérification qu'au moins un fichier a été chargé
        if not any(df is not None for df in self.data.values()):
            print("❌ Aucun fichier valide trouvé!")
            return

        # Génération des analyses
        self.create_evolution_plots()
        self.create_distribution_plots()
        self.create_comparison_plots()
        self.create_advanced_statistics()
        self.generate_reports()

        print("\n" + "="*60)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("="*60)
        print(f"📂 Résultats disponibles dans: {self.output_dir}")
        print(f"📊 Graphiques générés: {len([f for f in (self.output_dir / 'graphiques').rglob('*.png')])}")
        print(f"📄 Rapports générés: {len([f for f in (self.output_dir / 'rapports').rglob('*.*')])}")

        # Affichage du résumé des performances
        if self.stats:
            print("\n📈 RÉSUMÉ DES PERFORMANCES:")
            print("-" * 40)
            for test_name, stats in self.stats.items():
                print(f"Test {test_name:>6}: {stats['debit_moyen']:>7.2f} pkt/s (±{stats['ecart_type']:.2f})")

def main():
    """Fonction principale"""
    # Configuration des tests
    tests = {
        "1000": "1000.txt",
        "5000": "5000.txt",
        "25000": "25000.txt"
    }

    # Création de l'analyseur
    analyzer = SuricataAnalyzer(output_dir="resultats_analyse_suricata")

    # Lancement de l'analyse
    analyzer.analyze_files(tests)

if __name__ == "__main__":
    main()
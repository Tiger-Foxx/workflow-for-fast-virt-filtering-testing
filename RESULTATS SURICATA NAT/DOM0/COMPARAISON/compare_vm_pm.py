"""
Script de comparaison VM vs Machine Physique pour l'analyse Suricata
Compare les performances de débit entre deux environnements d'exécution
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

class VMvsPMAnalyzer:
    def __init__(self, output_dir="resultats_comparaison_vm_pm"):
        """
        Initialise l'analyseur de comparaison VM vs PM
        
        Args:
            output_dir (str): Dossier de sortie pour les résultats
        """
        self.output_dir = Path(output_dir)
        self.create_output_structure()
        
        # Configuration graphique
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (16, 10)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        # Palette de couleurs pour VM (rouges) et PM (bleus)
        self.colors_vm = ['#FF6B6B', '#FF8E8E', '#FFB3B3']  # Nuances de rouge
        self.colors_pm = ['#4ECDC4', '#45B7D1', '#96CEB4']  # Nuances de bleu/cyan
        
        # Configuration des tests
        self.test_rates = ['1000', '5000', '25000']
        
        # Stockage des données
        self.data_vm = {}
        self.data_pm = {}
        self.stats_vm = {}
        self.stats_pm = {}
        
    def create_output_structure(self):
        """Crée la structure de dossiers pour les résultats"""
        directories = [
            self.output_dir,
            self.output_dir / "graphiques",
            self.output_dir / "graphiques" / "comparaison_evolution",
            self.output_dir / "graphiques" / "comparaison_distribution", 
            self.output_dir / "graphiques" / "comparaison_globale",
            self.output_dir / "rapports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"📁 Structure créée dans: {self.output_dir}")
    
    def load_throughput_data(self, filepath, test_name, machine_type):
        """
        Charge et traite les données de débit
        
        Args:
            filepath (str): Chemin vers le fichier de données
            test_name (str): Nom du test (1000, 5000, 25000)
            machine_type (str): Type de machine ('VM' ou 'PM')
            
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
            
            # Suppression des valeurs négatives
            df['debit'] = df['debit'].clip(lower=0)
            
            # Ajout de métadonnées
            df['test'] = test_name
            df['machine'] = machine_type
            df['seconde'] = range(len(df))
            
            # Calcul de la moyenne mobile
            window = max(5, len(df) // 20)
            df['moyenne_mobile'] = df['debit'].rolling(window=window, center=True).mean()
            
            # Calcul des statistiques
            stats_key = f"{machine_type}_{test_name}"
            stats = {
                'machine': machine_type,
                'test_rate': int(test_name),
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
                'paquets_totaux': df['cum_pkts'].iloc[-1] if len(df) > 0 else 0,
                'efficacite': (df['debit'].mean() / int(test_name)) * 100 if int(test_name) > 0 else 0
            }
            
            if machine_type == 'VM':
                self.stats_vm[test_name] = stats
            else:
                self.stats_pm[test_name] = stats
                
            return df
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {filepath}: {e}")
            return None
    
    def create_evolution_comparison(self):
        """Crée les graphiques de comparaison d'évolution temporelle"""
        print("📊 Création des graphiques de comparaison d'évolution...")
        
        # 1. Graphique global avec toutes les courbes d'évolution
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Comparaison VM vs Machine Physique - Évolution temporelle', 
                    fontsize=18, fontweight='bold')
        
        # Graphique principal avec toutes les évolutions
        ax_main = axes[0, 0]
        
        # Tracé des courbes VM (rouges)
        for i, rate in enumerate(self.test_rates):
            if rate in self.data_vm and self.data_vm[rate] is not None:
                df = self.data_vm[rate]
                ax_main.plot(df['seconde'], df['debit'], 
                           color=self.colors_vm[i], alpha=0.7, linewidth=1.5,
                           label=f'VM - {rate} pkt/s')
        
        # Tracé des courbes PM (bleus)
        for i, rate in enumerate(self.test_rates):
            if rate in self.data_pm and self.data_pm[rate] is not None:
                df = self.data_pm[rate]
                ax_main.plot(df['seconde'], df['debit'],
                           color=self.colors_pm[i], alpha=0.7, linewidth=1.5,
                           label=f'PM - {rate} pkt/s')
        
        ax_main.set_title('Évolution complète du débit', fontsize=14, fontweight='bold')
        ax_main.set_xlabel('Temps (secondes)')
        ax_main.set_ylabel('Débit (paquets/s)')
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_main.grid(True, alpha=0.3)
        
        # Graphique avec moyennes mobiles
        ax_mobile = axes[0, 1]
        
        # Moyennes mobiles VM (rouges)
        for i, rate in enumerate(self.test_rates):
            if rate in self.data_vm and self.data_vm[rate] is not None:
                df = self.data_vm[rate]
                ax_mobile.plot(df['seconde'], df['moyenne_mobile'],
                             color=self.colors_vm[i], linewidth=3, alpha=0.8,
                             label=f'VM - {rate} pkt/s (moy. mobile)')
        
        # Moyennes mobiles PM (bleus)
        for i, rate in enumerate(self.test_rates):
            if rate in self.data_pm and self.data_pm[rate] is not None:
                df = self.data_pm[rate]
                ax_mobile.plot(df['seconde'], df['moyenne_mobile'],
                             color=self.colors_pm[i], linewidth=3, alpha=0.8,
                             label=f'PM - {rate} pkt/s (moy. mobile)')
        
        ax_mobile.set_title('Moyennes mobiles', fontsize=14, fontweight='bold')
        ax_mobile.set_xlabel('Temps (secondes)')
        ax_mobile.set_ylabel('Débit moyen (paquets/s)')
        ax_mobile.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_mobile.grid(True, alpha=0.3)
        
        # Comparaison par taux - 1000 pkt/s
        ax_1000 = axes[1, 0]
        if '1000' in self.data_vm and self.data_vm['1000'] is not None:
            df_vm = self.data_vm['1000']
            ax_1000.plot(df_vm['seconde'], df_vm['moyenne_mobile'], 
                        color=self.colors_vm[0], linewidth=3, label='VM - 1000 pkt/s')
        if '1000' in self.data_pm and self.data_pm['1000'] is not None:
            df_pm = self.data_pm['1000']
            ax_1000.plot(df_pm['seconde'], df_pm['moyenne_mobile'],
                        color=self.colors_pm[0], linewidth=3, label='PM - 1000 pkt/s')
        
        ax_1000.set_title('Comparaison directe @ 1000 pkt/s', fontsize=14, fontweight='bold')
        ax_1000.set_xlabel('Temps (secondes)')
        ax_1000.set_ylabel('Débit (paquets/s)')
        ax_1000.legend()
        ax_1000.grid(True, alpha=0.3)
        
        # Graphique d'efficacité
        ax_eff = axes[1, 1]
        
        rates_vm = []
        efficacites_vm = []
        rates_pm = []
        efficacites_pm = []
        
        for rate in self.test_rates:
            if rate in self.stats_vm:
                rates_vm.append(int(rate))
                efficacites_vm.append(self.stats_vm[rate]['efficacite'])
            if rate in self.stats_pm:
                rates_pm.append(int(rate))
                efficacites_pm.append(self.stats_pm[rate]['efficacite'])
        
        if rates_vm:
            ax_eff.plot(rates_vm, efficacites_vm, 'o-', color='red', linewidth=3, 
                       markersize=8, label='VM', alpha=0.8)
        if rates_pm:
            ax_eff.plot(rates_pm, efficacites_pm, 's-', color='blue', linewidth=3,
                       markersize=8, label='Machine Physique', alpha=0.8)
        
        ax_eff.set_title('Efficacité de traitement', fontsize=14, fontweight='bold')
        ax_eff.set_xlabel('Charge envoyée (paquets/s)')
        ax_eff.set_ylabel('Efficacité (%)')
        ax_eff.legend()
        ax_eff.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "graphiques" / "comparaison_evolution" / "evolution_complete.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_distribution_comparison(self):
        """Crée les graphiques de comparaison de distribution"""
        print("📊 Création des graphiques de comparaison de distribution...")
        
        # 1. Graphique principal avec box plots et histogrammes
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Comparaison VM vs Machine Physique - Distributions', 
                    fontsize=18, fontweight='bold')
        
        # Box plots comparatifs
        ax_box = axes[0, 0]
        
        box_data = []
        box_labels = []
        box_colors = []
        
        for i, rate in enumerate(self.test_rates):
            if rate in self.data_vm and self.data_vm[rate] is not None:
                box_data.append(self.data_vm[rate]['debit'])
                box_labels.append(f'VM\n{rate}')
                box_colors.append(self.colors_vm[i])
            
            if rate in self.data_pm and self.data_pm[rate] is not None:
                box_data.append(self.data_pm[rate]['debit'])
                box_labels.append(f'PM\n{rate}')
                box_colors.append(self.colors_pm[i])
        
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax_box.set_title('Distribution comparative (Box Plots)', fontsize=14, fontweight='bold')
        ax_box.set_ylabel('Débit (paquets/s)')
        ax_box.grid(True, alpha=0.3)
        
        # Histogrammes superposés par taux
        ax_hist = axes[0, 1]
        
        for i, rate in enumerate(self.test_rates):
            if rate in self.data_vm and self.data_vm[rate] is not None:
                ax_hist.hist(self.data_vm[rate]['debit'], bins=20, alpha=0.5, 
                           color=self.colors_vm[i], label=f'VM - {rate} pkt/s',
                           density=True)
            
            if rate in self.data_pm and self.data_pm[rate] is not None:
                ax_hist.hist(self.data_pm[rate]['debit'], bins=20, alpha=0.5,
                           color=self.colors_pm[i], label=f'PM - {rate} pkt/s',
                           density=True)
        
        ax_hist.set_title('Histogrammes de densité', fontsize=14, fontweight='bold')
        ax_hist.set_xlabel('Débit (paquets/s)')
        ax_hist.set_ylabel('Densité')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Graphique en barres des moyennes
        ax_bar = axes[1, 0]
        
        rates = [int(rate) for rate in self.test_rates]
        means_vm = [self.stats_vm[rate]['debit_moyen'] if rate in self.stats_vm else 0 
                   for rate in self.test_rates]
        means_pm = [self.stats_pm[rate]['debit_moyen'] if rate in self.stats_pm else 0 
                   for rate in self.test_rates]
        stds_vm = [self.stats_vm[rate]['ecart_type'] if rate in self.stats_vm else 0 
                  for rate in self.test_rates]
        stds_pm = [self.stats_pm[rate]['ecart_type'] if rate in self.stats_pm else 0 
                  for rate in self.test_rates]
        
        x = np.arange(len(rates))
        width = 0.35
        
        bars1 = ax_bar.bar(x - width/2, means_vm, width, yerr=stds_vm, 
                          label='VM', color='red', alpha=0.7, capsize=5)
        bars2 = ax_bar.bar(x + width/2, means_pm, width, yerr=stds_pm,
                          label='Machine Physique', color='blue', alpha=0.7, capsize=5)
        
        # Ajout des valeurs sur les barres
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if means_vm[i] > 0:
                ax_bar.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + stds_vm[i],
                           f'{means_vm[i]:.0f}', ha='center', va='bottom', fontweight='bold')
            if means_pm[i] > 0:
                ax_bar.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + stds_pm[i],
                           f'{means_pm[i]:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax_bar.set_title('Débit moyen ± écart-type', fontsize=14, fontweight='bold')
        ax_bar.set_xlabel('Charge envoyée (paquets/s)')
        ax_bar.set_ylabel('Débit moyen (paquets/s)')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(rates)
        ax_bar.legend()
        ax_bar.grid(True, alpha=0.3)
        
        # Graphique de variabilité (coefficient de variation)
        ax_cv = axes[1, 1]
        
        cv_vm = [self.stats_vm[rate]['cv'] if rate in self.stats_vm else 0 
                for rate in self.test_rates]
        cv_pm = [self.stats_pm[rate]['cv'] if rate in self.stats_pm else 0 
                for rate in self.test_rates]
        
        bars1 = ax_cv.bar(x - width/2, cv_vm, width, label='VM', color='red', alpha=0.7)
        bars2 = ax_cv.bar(x + width/2, cv_pm, width, label='Machine Physique', 
                         color='blue', alpha=0.7)
        
        ax_cv.set_title('Variabilité (Coefficient de Variation)', fontsize=14, fontweight='bold')
        ax_cv.set_xlabel('Charge envoyée (paquets/s)')
        ax_cv.set_ylabel('Coefficient de Variation (%)')
        ax_cv.set_xticks(x)
        ax_cv.set_xticklabels(rates)
        ax_cv.legend()
        ax_cv.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "graphiques" / "comparaison_distribution" / "distribution_complete.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_global_comparison(self):
        """Crée le graphique de comparaison globale"""
        print("📊 Création du graphique de comparaison globale...")
        
        # Graphique global de synthèse
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        fig.suptitle('SYNTHÈSE COMPARATIVE VM vs MACHINE PHYSIQUE', 
                    fontsize=20, fontweight='bold')
        
        rates = [int(rate) for rate in self.test_rates]
        
        # 1. Performance globale
        ax1 = axes[0, 0]
        means_vm = [self.stats_vm[rate]['debit_moyen'] if rate in self.stats_vm else 0 
                   for rate in self.test_rates]
        means_pm = [self.stats_pm[rate]['debit_moyen'] if rate in self.stats_pm else 0 
                   for rate in self.test_rates]
        
        ax1.plot(rates, means_vm, 'o-', color='red', linewidth=4, markersize=10, 
                label='VM', alpha=0.8)
        ax1.plot(rates, means_pm, 's-', color='blue', linewidth=4, markersize=10,
                label='Machine Physique', alpha=0.8)
        ax1.set_title('Performance Globale', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Charge (paquets/s)')
        ax1.set_ylabel('Débit moyen (paquets/s)')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Efficacité
        ax2 = axes[0, 1]
        eff_vm = [self.stats_vm[rate]['efficacite'] if rate in self.stats_vm else 0 
                 for rate in self.test_rates]
        eff_pm = [self.stats_pm[rate]['efficacite'] if rate in self.stats_pm else 0 
                 for rate in self.test_rates]
        
        ax2.plot(rates, eff_vm, 'o-', color='red', linewidth=4, markersize=10, 
                label='VM', alpha=0.8)
        ax2.plot(rates, eff_pm, 's-', color='blue', linewidth=4, markersize=10,
                label='Machine Physique', alpha=0.8)
        ax2.set_title('Efficacité', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Charge (paquets/s)')
        ax2.set_ylabel('Efficacité (%)')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Stabilité (inverse du CV)
        ax3 = axes[0, 2]
        stab_vm = [100 - self.stats_vm[rate]['cv'] if rate in self.stats_vm else 0 
                  for rate in self.test_rates]
        stab_pm = [100 - self.stats_pm[rate]['cv'] if rate in self.stats_pm else 0 
                  for rate in self.test_rates]
        
        ax3.plot(rates, stab_vm, 'o-', color='red', linewidth=4, markersize=10, 
                label='VM', alpha=0.8)
        ax3.plot(rates, stab_pm, 's-', color='blue', linewidth=4, markersize=10,
                label='Machine Physique', alpha=0.8)
        ax3.set_title('Stabilité (100 - CV)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Charge (paquets/s)')
        ax3.set_ylabel('Stabilité (%)')
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Radar chart comparatif
        ax4 = axes[1, 0]
        categories = ['Perf.\n1000', 'Perf.\n5000', 'Perf.\n25000', 
                     'Eff.\n1000', 'Eff.\n5000', 'Eff.\n25000']
        
        # Normalisation des valeurs pour le radar
        values_vm = []
        values_pm = []
        
        for rate in self.test_rates:
            if rate in self.stats_vm and rate in self.stats_pm:
                max_debit = max(self.stats_vm[rate]['debit_moyen'], 
                               self.stats_pm[rate]['debit_moyen'])
                if max_debit > 0:
                    values_vm.append(self.stats_vm[rate]['debit_moyen'] / max_debit * 100)
                    values_pm.append(self.stats_pm[rate]['debit_moyen'] / max_debit * 100)
                else:
                    values_vm.append(0)
                    values_pm.append(0)
        
        for rate in self.test_rates:
            if rate in self.stats_vm and rate in self.stats_pm:
                max_eff = max(self.stats_vm[rate]['efficacite'], 
                             self.stats_pm[rate]['efficacite'])
                if max_eff > 0:
                    values_vm.append(self.stats_vm[rate]['efficacite'] / max_eff * 100)
                    values_pm.append(self.stats_pm[rate]['efficacite'] / max_eff * 100)
                else:
                    values_vm.append(0)
                    values_pm.append(0)
        
        # Graphique radar simplifié en barres
        x_pos = np.arange(len(categories))
        width = 0.35
        
        if values_vm and values_pm:
            ax4.bar(x_pos - width/2, values_vm[:len(categories)], width, 
                   label='VM', color='red', alpha=0.7)
            ax4.bar(x_pos + width/2, values_pm[:len(categories)], width,
                   label='Machine Physique', color='blue', alpha=0.7)
        
        ax4.set_title('Comparaison normalisée', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Performance relative (%)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories, rotation=45)
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # 5. Tableau de statistiques
        ax5 = axes[1, 1]
        ax5.axis('off')
        
        stats_text = "STATISTIQUES COMPARATIVES\n"
        stats_text += "=" * 40 + "\n\n"
        
        for rate in self.test_rates:
            stats_text += f"Test @ {rate} pkt/s:\n"
            if rate in self.stats_vm:
                stats_text += f"  VM:  {self.stats_vm[rate]['debit_moyen']:.2f} ± {self.stats_vm[rate]['ecart_type']:.2f} pkt/s\n"
            if rate in self.stats_pm:
                stats_text += f"  PM:  {self.stats_pm[rate]['debit_moyen']:.2f} ± {self.stats_pm[rate]['ecart_type']:.2f} pkt/s\n"
            
            if rate in self.stats_vm and rate in self.stats_pm:
                diff = self.stats_pm[rate]['debit_moyen'] - self.stats_vm[rate]['debit_moyen']
                diff_pct = (diff / self.stats_vm[rate]['debit_moyen']) * 100 if self.stats_vm[rate]['debit_moyen'] > 0 else 0
                stats_text += f"  Diff: {diff:+.2f} pkt/s ({diff_pct:+.1f}%)\n"
            stats_text += "\n"
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        # 6. Gain de performance
        ax6 = axes[1, 2]
        
        gains = []
        rates_with_data = []
        
        for rate in self.test_rates:
            if rate in self.stats_vm and rate in self.stats_pm:
                if self.stats_vm[rate]['debit_moyen'] > 0:
                    gain = ((self.stats_pm[rate]['debit_moyen'] - self.stats_vm[rate]['debit_moyen']) / 
                           self.stats_vm[rate]['debit_moyen']) * 100
                    gains.append(gain)
                    rates_with_data.append(int(rate))
        
        if gains:
            colors = ['green' if g > 0 else 'red' for g in gains]
            bars = ax6.bar(rates_with_data, gains, color=colors, alpha=0.7)
            
            # Ajout des valeurs sur les barres
            for bar, gain in zip(bars, gains):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2, height + (1 if height > 0 else -1),
                        f'{gain:+.1f}%', ha='center', 
                        va='bottom' if height > 0 else 'top', fontweight='bold')
        
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.set_title('Gain PM vs VM', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Charge (paquets/s)')
        ax6.set_ylabel('Gain (%)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "graphiques" / "comparaison_globale" / "synthese_complete.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self):
        """Génère le rapport de comparaison"""
        print("📄 Génération du rapport de comparaison...")
        
        rapport_path = self.output_dir / "rapports" / "rapport_comparaison_vm_pm.txt"
        
        with open(rapport_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT DE COMPARAISON VM vs MACHINE PHYSIQUE\n")
            f.write("ANALYSE DE PERFORMANCE IDS SURICATA\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("RÉSUMÉ EXÉCUTIF\n")
            f.write("-" * 20 + "\n\n")
            
            # Calcul des moyennes globales
            if self.stats_vm and self.stats_pm:
                mean_vm = np.mean([stats['debit_moyen'] for stats in self.stats_vm.values()])
                mean_pm = np.mean([stats['debit_moyen'] for stats in self.stats_pm.values()])
                
                f.write(f"• Performance moyenne VM: {mean_vm:.2f} pkt/s\n")
                f.write(f"• Performance moyenne PM: {mean_pm:.2f} pkt/s\n")
                
                if mean_vm > 0:
                    gain_global = ((mean_pm - mean_vm) / mean_vm) * 100
                    f.write(f"• Gain moyen PM vs VM: {gain_global:+.1f}%\n\n")
            
            f.write("DÉTAILS PAR TEST\n")
            f.write("-" * 20 + "\n\n")
            
            for rate in self.test_rates:
                f.write(f"Test @ {rate} paquets/s:\n")
                
                if rate in self.stats_vm:
                    f.write(f"  VM:\n")
                    f.write(f"    • Débit moyen: {self.stats_vm[rate]['debit_moyen']:.2f} pkt/s\n")
                    f.write(f"    • Écart-type: {self.stats_vm[rate]['ecart_type']:.2f} pkt/s\n")
                    f.write(f"    • Efficacité: {self.stats_vm[rate]['efficacite']:.1f}%\n")
                    f.write(f"    • CV: {self.stats_vm[rate]['cv']:.1f}%\n")
                
                if rate in self.stats_pm:
                    f.write(f"  Machine Physique:\n")
                    f.write(f"    • Débit moyen: {self.stats_pm[rate]['debit_moyen']:.2f} pkt/s\n")
                    f.write(f"    • Écart-type: {self.stats_pm[rate]['ecart_type']:.2f} pkt/s\n")
                    f.write(f"    • Efficacité: {self.stats_pm[rate]['efficacite']:.1f}%\n")
                    f.write(f"    • CV: {self.stats_pm[rate]['cv']:.1f}%\n")
                
                if rate in self.stats_vm and rate in self.stats_pm:
                    diff = self.stats_pm[rate]['debit_moyen'] - self.stats_vm[rate]['debit_moyen']
                    if self.stats_vm[rate]['debit_moyen'] > 0:
                        diff_pct = (diff / self.stats_vm[rate]['debit_moyen']) * 100
                        f.write(f"  Comparaison:\n")
                        f.write(f"    • Différence absolue: {diff:+.2f} pkt/s\n")
                        f.write(f"    • Différence relative: {diff_pct:+.1f}%\n")
                
                f.write("\n")
            
            f.write("CONCLUSIONS\n")
            f.write("-" * 20 + "\n\n")
            
            if self.stats_vm and self.stats_pm:
                # Détermination du meilleur environnement
                wins_pm = 0
                wins_vm = 0
                
                for rate in self.test_rates:
                    if rate in self.stats_vm and rate in self.stats_pm:
                        if self.stats_pm[rate]['debit_moyen'] > self.stats_vm[rate]['debit_moyen']:
                            wins_pm += 1
                        else:
                            wins_vm += 1
                
                if wins_pm > wins_vm:
                    f.write("• La Machine Physique présente de meilleures performances globales\n")
                elif wins_vm > wins_pm:
                    f.write("• La VM présente de meilleures performances globales\n")
                else:
                    f.write("• Les performances sont équivalentes entre VM et Machine Physique\n")
                
                f.write(f"• Nombre de tests gagnés - PM: {wins_pm}, VM: {wins_vm}\n")
    
    def analyze_comparison(self):
        """Analyse principale de comparaison"""
        print("🚀 Démarrage de l'analyse comparative VM vs Machine Physique...")
        
        # Vérification de la structure des dossiers
        vm_dir = Path("VM")
        pm_dir = Path("PM")
        
        if not vm_dir.exists():
            print(f"❌ Dossier VM non trouvé: {vm_dir}")
            return
        
        if not pm_dir.exists():
            print(f"❌ Dossier PM non trouvé: {pm_dir}")
            return
        
        # Chargement des données VM
        print("📁 Chargement des données VM...")
        for rate in self.test_rates:
            vm_file = vm_dir / f"{rate}.txt"
            if vm_file.exists():
                print(f"  → Chargement VM: {vm_file}")
                self.data_vm[rate] = self.load_throughput_data(vm_file, rate, 'VM')
            else:
                print(f"  ⚠️ Fichier VM non trouvé: {vm_file}")
                self.data_vm[rate] = None
        
        # Chargement des données PM
        print("📁 Chargement des données Machine Physique...")
        for rate in self.test_rates:
            pm_file = pm_dir / f"{rate}.txt"
            if pm_file.exists():
                print(f"  → Chargement PM: {pm_file}")
                self.data_pm[rate] = self.load_throughput_data(pm_file, rate, 'PM')
            else:
                print(f"  ⚠️ Fichier PM non trouvé: {pm_file}")
                self.data_pm[rate] = None
        
        # Vérification qu'au moins quelques données ont été chargées
        vm_loaded = sum(1 for df in self.data_vm.values() if df is not None)
        pm_loaded = sum(1 for df in self.data_pm.values() if df is not None)
        
        if vm_loaded == 0 and pm_loaded == 0:
            print("❌ Aucune donnée valide trouvée!")
            return
        
        print(f"✅ Données chargées - VM: {vm_loaded}/3, PM: {pm_loaded}/3")
        
        # Génération des analyses
        self.create_evolution_comparison()
        self.create_distribution_comparison()
        self.create_global_comparison()
        self.generate_comparison_report()
        
        print("\n" + "="*60)
        print("✅ ANALYSE COMPARATIVE TERMINÉE AVEC SUCCÈS!")
        print("="*60)
        print(f"📂 Résultats disponibles dans: {self.output_dir}")
        print(f"📊 Graphiques générés: {len(list((self.output_dir / 'graphiques').rglob('*.png')))}")
        print(f"📄 Rapports générés: {len(list((self.output_dir / 'rapports').rglob('*.*')))}")
        
        # Affichage du résumé
        if self.stats_vm or self.stats_pm:
            print("\n📈 RÉSUMÉ DES PERFORMANCES:")
            print("-" * 50)
            print(f"{'Test':<8} {'VM (pkt/s)':<12} {'PM (pkt/s)':<12} {'Gain (%)':<10}")
            print("-" * 50)
            
            for rate in self.test_rates:
                vm_val = f"{self.stats_vm[rate]['debit_moyen']:.2f}" if rate in self.stats_vm else "N/A"
                pm_val = f"{self.stats_pm[rate]['debit_moyen']:.2f}" if rate in self.stats_pm else "N/A"
                
                gain = "N/A"
                if rate in self.stats_vm and rate in self.stats_pm and self.stats_vm[rate]['debit_moyen'] > 0:
                    gain_val = ((self.stats_pm[rate]['debit_moyen'] - self.stats_vm[rate]['debit_moyen']) / 
                               self.stats_vm[rate]['debit_moyen']) * 100
                    gain = f"{gain_val:+.1f}%"
                
                print(f"{rate:<8} {vm_val:<12} {pm_val:<12} {gain:<10}")

def main():
    """Fonction principale"""
    print("🔄 Démarrage de l'analyse comparative VM vs Machine Physique")
    print("📋 Structure attendue:")
    print("   VM/1000.txt, VM/5000.txt, VM/25000.txt")
    print("   PM/1000.txt, PM/5000.txt, PM/25000.txt")
    print()
    
    # Création de l'analyseur
    analyzer = VMvsPMAnalyzer(output_dir="resultats_comparaison_vm_pm")
    
    # Lancement de l'analyse
    analyzer.analyze_comparison()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script d'analyse ULTIME des résultats de tests Snort 3
Version ULTIMATE EDITION - Histogrammes comparatifs avancés

Auteur: Assistant IA pour theTigerFox
Date: 2025-07-28
Version: ULTIMATE 3.0 - Full Histogram Edition

Ce script génère des histogrammes comparatifs détaillés pour analyser 
VRAIMENT les performances de Snort 3 à différents débits.

Dépendances:
- pandas
- matplotlib
- numpy
- ace_tools (optionnel)

Usage:
    python3 analyze_snort_results_ultimate.py /chemin/vers/results
    python3 analyze_snort_results_ultimate.py .  # pour le répertoire courant
"""

import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Gestion optionnelle d'ace_tools
try:
    import ace_tools
    ACE_TOOLS_AVAILABLE = True
except ImportError:
    ACE_TOOLS_AVAILABLE = False
    warnings.warn("ace_tools non disponible, le DataFrame sera affiché avec print()")

class SnortUltimateAnalyzer:
    """
    Analyseur ULTIME des résultats Snort 3 avec histogrammes comparatifs avancés
    
    Cette classe génère des graphiques histogramme vraiment utiles pour comparer
    les performances à différents débits avec des métriques pertinentes.
    """
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.data = []
        
        # Créer le dossier figures s'il n'existe pas
        self.figures_dir.mkdir(exist_ok=True)
        
        # Configuration matplotlib pour des graphiques époustouflants
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 11
        
        print("🚀 Initialisation de l'analyseur ULTIME Snort 3")
    
    def extract_pps_from_dirname(self, dirname: str) -> Optional[int]:
        """Extrait la valeur pps du nom de dossier (ex: '1000pps' -> 1000)"""
        match = re.search(r'(\d+)pps', dirname.lower())
        return int(match.group(1)) if match else None
    
    def parse_metrics_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Parse le fichier metrics.txt avec extraction précise des valeurs
        
        Format attendu:
        perf_monitor.packets_total:    69545
        latency.total_usecs_sum:      0
        latency.max_usecs:            0
        latency.avg_usecs_per_pkt:    0.00
        cpu.user_usec:                974415
        cpu.sys_usec:                 1270970
        cpu.total_usec:               2245385
        runtime_sec:                  303
        debit_avg_pps:                229.52
        """
        metrics = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Patterns précis pour extraction
            patterns = {
                'packets_total': r'perf_monitor\.packets_total:\s*(\d+)',
                'latency_total_usecs_sum': r'latency\.total_usecs_sum:\s*(\d+)',
                'latency_max_usecs': r'latency\.max_usecs:\s*(\d+)',
                'latency_avg_usecs_per_pkt': r'latency\.avg_usecs_per_pkt:\s*([\d.]+)',
                'cpu_user_usec': r'cpu\.user_usec:\s*(\d+)',
                'cpu_sys_usec': r'cpu\.sys_usec:\s*(\d+)',
                'cpu_total_usec': r'cpu\.total_usec:\s*(\d+)',
                'runtime_sec': r'runtime_sec:\s*(\d+)',
                'debit_avg_pps': r'debit_avg_pps:\s*([\d.]+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    value = match.group(1)
                    metrics[key] = float(value) if '.' in value else int(value)
                else:
                    print(f"⚠️  Valeur '{key}' non trouvée dans {filepath}")
                    metrics[key] = 0
                    
        except Exception as e:
            print(f"❌ Erreur lors de la lecture de {filepath}: {e}")
            
        return metrics
    
    def parse_console_output(self, filepath: Path) -> Dict[str, Any]:
        """
        Parse CORRECTEMENT la sortie console (sortie.txt) selon le format réel
        
        Le fichier contient des sections séparées par des lignes de tirets.
        Structure réelle:
        --------------------------------------------------
        codec
                            total: 69882        (100.000%)
                            other: 30           (  0.043%)
        ...
        --------------------------------------------------
        Module Statistics
        --------------------------------------------------
        detection
                         analyzed: 69882
        --------------------------------------------------
        """
        console_data = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # === EXTRACTION SECTION CODEC ===
            # Recherche de la section codec entre les séparateurs
            codec_match = re.search(r'codec\s*\n(.*?)(?:\n-{10,})', content, re.DOTALL)
            if codec_match:
                codec_data = {}
                codec_section = codec_match.group(1)
                
                # Extraction de chaque protocole avec pattern précis
                # Format: "                    arp: 68425        ( 97.915%)"
                protocol_lines = re.findall(r'\s*(\w+):\s*(\d+)\s*\(.*?\)', codec_section)
                
                for protocol, count in protocol_lines:
                    codec_data[protocol] = int(count)
                    
                console_data['codec'] = codec_data
                print(f"✅ Codec: {len(codec_data)} protocoles extraits: {list(codec_data.keys())}")
            else:
                print("⚠️  Section codec non trouvée")
                console_data['codec'] = {}
            
            # === EXTRACTION MODULES ===
            # Recherche dans la section "Module Statistics"
            module_section_match = re.search(r'Module Statistics\s*\n-+\s*(.*?)(?:\n[A-Z].*Statistics|\nSummary Statistics|$)', 
                                           content, re.DOTALL)
            
            if module_section_match:
                module_section = module_section_match.group(1)
                
                # Module Detection
                detection_match = re.search(r'detection\s*\n\s*analyzed:\s*(\d+)', module_section)
                console_data['detection_analyzed'] = int(detection_match.group(1)) if detection_match else 0
                
                # Module perf_monitor
                perf_monitor_match = re.search(r'perf_monitor\s*\n\s*packets:\s*(\d+)', module_section)
                console_data['perf_monitor_packets'] = int(perf_monitor_match.group(1)) if perf_monitor_match else 0
                
                # Module port_scan
                port_scan_packets_match = re.search(r'port_scan\s*\n\s*packets:\s*(\d+)', module_section)
                port_scan_trackers_match = re.search(r'port_scan\s*\n.*?trackers:\s*(\d+)', module_section, re.DOTALL)
                console_data['port_scan_packets'] = int(port_scan_packets_match.group(1)) if port_scan_packets_match else 0
                console_data['port_scan_trackers'] = int(port_scan_trackers_match.group(1)) if port_scan_trackers_match else 0
                
                # Module stream
                stream_flows_match = re.search(r'stream\s*\n\s*flows:\s*(\d+)', module_section)
                stream_prunes_match = re.search(r'stream\s*\n.*?total_prunes:\s*(\d+)', module_section, re.DOTALL)
                stream_idle_match = re.search(r'stream\s*\n.*?idle_prunes:\s*(\d+)', module_section, re.DOTALL)
                
                console_data['stream_flows'] = int(stream_flows_match.group(1)) if stream_flows_match else 0
                console_data['stream_total_prunes'] = int(stream_prunes_match.group(1)) if stream_prunes_match else 0
                console_data['stream_idle_prunes'] = int(stream_idle_match.group(1)) if stream_idle_match else 0
                
                # Module arp_spoof
                arp_spoof_match = re.search(r'arp_spoof\s*\n\s*packets:\s*(\d+)', module_section)
                console_data['arp_spoof_packets'] = int(arp_spoof_match.group(1)) if arp_spoof_match else 0
                
                # Module binder
                binder_raw_match = re.search(r'binder\s*\n\s*raw_packets:\s*(\d+)', module_section)
                binder_flows_match = re.search(r'binder\s*\n.*?new_flows:\s*(\d+)', module_section, re.DOTALL)
                binder_inspects_match = re.search(r'binder\s*\n.*?inspects:\s*(\d+)', module_section, re.DOTALL)
                
                console_data['binder_raw_packets'] = int(binder_raw_match.group(1)) if binder_raw_match else 0
                console_data['binder_new_flows'] = int(binder_flows_match.group(1)) if binder_flows_match else 0
                console_data['binder_inspects'] = int(binder_inspects_match.group(1)) if binder_inspects_match else 0
                
                print(f"✅ Modules extraits: detection={console_data['detection_analyzed']}, "
                      f"port_scan={console_data['port_scan_packets']}, stream={console_data['stream_flows']}")
            else:
                print("⚠️  Section Module Statistics non trouvée")
                for key in ['detection_analyzed', 'perf_monitor_packets', 'port_scan_packets', 
                           'port_scan_trackers', 'stream_flows', 'stream_total_prunes', 
                           'stream_idle_prunes', 'arp_spoof_packets', 'binder_raw_packets', 
                           'binder_new_flows', 'binder_inspects']:
                    console_data[key] = 0
                    
        except Exception as e:
            print(f"❌ Erreur lors de la lecture de {filepath}: {e}")
            
        return console_data
    
    def calculate_ultimate_metrics(self, data_row: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcule TOUTES les métriques comparatives utiles
        
        Ces métriques permettront de créer des histogrammes comparatifs significatifs
        """
        metrics = {}
        
        # === MÉTRIQUES DE BASE ===
        # Efficacité de débit (%)
        if data_row['rate_pps'] > 0:
            metrics['throughput_efficiency_pct'] = (data_row['debit_avg_pps'] / data_row['rate_pps']) * 100
            metrics['packet_loss_pct'] = 100 - metrics['throughput_efficiency_pct']
        else:
            metrics['throughput_efficiency_pct'] = 0
            metrics['packet_loss_pct'] = 100
        
        # === MÉTRIQUES CPU AVANCÉES ===
        metrics['cpu_total_sec'] = (data_row['cpu_user_usec'] + data_row['cpu_sys_usec']) / 1_000_000
        metrics['cpu_user_sec'] = data_row['cpu_user_usec'] / 1_000_000
        metrics['cpu_sys_sec'] = data_row['cpu_sys_usec'] / 1_000_000
        
        # Charge CPU réelle (% du temps total)
        if data_row['runtime_sec'] > 0:
            metrics['cpu_load_pct'] = (metrics['cpu_total_sec'] / data_row['runtime_sec']) * 100
            metrics['cpu_user_load_pct'] = (metrics['cpu_user_sec'] / data_row['runtime_sec']) * 100
            metrics['cpu_sys_load_pct'] = (metrics['cpu_sys_sec'] / data_row['runtime_sec']) * 100
        else:
            metrics['cpu_load_pct'] = 0
            metrics['cpu_user_load_pct'] = 0
            metrics['cpu_sys_load_pct'] = 0
        
        # === RATIOS CRITIQUES POUR HISTOGRAMMES ===
        # Paquets par seconde de CPU
        if metrics['cpu_total_sec'] > 0:
            metrics['packets_per_cpu_sec'] = data_row['packets_total'] / metrics['cpu_total_sec']
        else:
            metrics['packets_per_cpu_sec'] = 0
        
        # Paquets par seconde de runtime
        if data_row['runtime_sec'] > 0:
            metrics['packets_per_runtime_sec'] = data_row['packets_total'] / data_row['runtime_sec']
        else:
            metrics['packets_per_runtime_sec'] = 0
        
        # Ratio efficacité CPU/Runtime
        if metrics['packets_per_runtime_sec'] > 0:
            metrics['cpu_efficiency_ratio'] = metrics['packets_per_cpu_sec'] / metrics['packets_per_runtime_sec']
        else:
            metrics['cpu_efficiency_ratio'] = 0
        
        # Temps CPU par paquet (µs/paquet)
        if data_row['packets_total'] > 0:
            metrics['cpu_time_per_packet_usec'] = (data_row['cpu_user_usec'] + data_row['cpu_sys_usec']) / data_row['packets_total']
            metrics['cpu_user_time_per_packet_usec'] = data_row['cpu_user_usec'] / data_row['packets_total']
            metrics['cpu_sys_time_per_packet_usec'] = data_row['cpu_sys_usec'] / data_row['packets_total']
        else:
            metrics['cpu_time_per_packet_usec'] = 0
            metrics['cpu_user_time_per_packet_usec'] = 0
            metrics['cpu_sys_time_per_packet_usec'] = 0
        
        # Ratio User/System CPU
        if data_row['cpu_sys_usec'] > 0:
            metrics['cpu_user_sys_ratio'] = data_row['cpu_user_usec'] / data_row['cpu_sys_usec']
        else:
            metrics['cpu_user_sys_ratio'] = 0
        
        # === MÉTRIQUES DE MODULES ===
        # Efficacité de détection
        if data_row['rate_pps'] > 0:
            metrics['detection_efficiency_pct'] = (data_row['detection_analyzed'] / (data_row['rate_pps'] * data_row['runtime_sec'])) * 100
        else:
            metrics['detection_efficiency_pct'] = 0
        
        # Paquets détectés par seconde
        if data_row['runtime_sec'] > 0:
            metrics['detection_pps'] = data_row['detection_analyzed'] / data_row['runtime_sec']
        else:
            metrics['detection_pps'] = 0
        
        # Flux par seconde
        if data_row['runtime_sec'] > 0:
            metrics['flows_per_sec'] = data_row['stream_flows'] / data_row['runtime_sec']
        else:
            metrics['flows_per_sec'] = 0
        
        # Paquets par flux
        if data_row['stream_flows'] > 0:
            metrics['packets_per_flow'] = data_row['detection_analyzed'] / data_row['stream_flows']
        else:
            metrics['packets_per_flow'] = 0
        
        return metrics
    
    def collect_data(self):
        """Collecte les données avec parsing correct des fichiers"""
        print(f"🔍 Recherche des dossiers de résultats dans {self.results_dir}")
        
        pps_dirs = [d for d in self.results_dir.iterdir() 
                   if d.is_dir() and d.name.lower().endswith('pps')]
        
        if not pps_dirs:
            print("❌ Aucun dossier se terminant par 'pps' trouvé")
            return
        
        print(f"📁 Dossiers trouvés: {[d.name for d in pps_dirs]}")
        
        for pps_dir in sorted(pps_dirs, key=lambda x: self.extract_pps_from_dirname(x.name) or 0):
            rate_pps = self.extract_pps_from_dirname(pps_dir.name)
            if rate_pps is None:
                print(f"⚠️  Impossible d'extraire le débit de {pps_dir.name}")
                continue
                
            print(f"\n📊 Analyse du dossier {pps_dir.name} (débit: {rate_pps} pps)")
            
            metrics_file = pps_dir / "metrics.txt"
            sortie_file = pps_dir / "sortie.txt"
            
            # Vérification de l'existence des fichiers
            if not metrics_file.exists():
                print(f"❌ Fichier manquant: {metrics_file}")
                continue
            if not sortie_file.exists():
                print(f"❌ Fichier manquant: {sortie_file}")
                continue
            
            # Extraction des données
            metrics = self.parse_metrics_file(metrics_file)
            console = self.parse_console_output(sortie_file)
            
            # Compilation des données de base
            row_data = {
                'rate_pps': rate_pps,
                'packets_total': metrics.get('packets_total', 0),
                'runtime_sec': metrics.get('runtime_sec', 0),
                'debit_avg_pps': metrics.get('debit_avg_pps', 0),
                'latency_avg_usecs': metrics.get('latency_avg_usecs_per_pkt', 0),
                'latency_max_usecs': metrics.get('latency_max_usecs', 0),
                'cpu_user_usec': metrics.get('cpu_user_usec', 0),
                'cpu_sys_usec': metrics.get('cpu_sys_usec', 0),
                'detection_analyzed': console.get('detection_analyzed', 0),
                'perf_monitor_packets': console.get('perf_monitor_packets', 0),
                'port_scan_packets': console.get('port_scan_packets', 0),
                'port_scan_trackers': console.get('port_scan_trackers', 0),
                'stream_flows': console.get('stream_flows', 0),
                'stream_total_prunes': console.get('stream_total_prunes', 0),
                'stream_idle_prunes': console.get('stream_idle_prunes', 0),
                'arp_spoof_packets': console.get('arp_spoof_packets', 0),
                'binder_raw_packets': console.get('binder_raw_packets', 0),
                'binder_new_flows': console.get('binder_new_flows', 0),
                'binder_inspects': console.get('binder_inspects', 0),
                'codec_data': console.get('codec', {})
            }
            
            # Calcul des métriques avancées
            advanced_metrics = self.calculate_ultimate_metrics(row_data)
            row_data.update(advanced_metrics)
            
            self.data.append(row_data)
            print(f"✅ Données extraites pour {rate_pps} pps - Detection: {row_data['detection_analyzed']} paquets")
    
    def create_histogram_debit_comparison(self):
        """
        HISTOGRAMME 1: Comparaison des débits mesurés vs injectés
        
        Histogramme côte à côte montrant clairement l'écart entre
        le débit injecté et le débit réellement mesuré par Snort.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        rates = [d['rate_pps'] for d in self.data]
        measured_rates = [d['debit_avg_pps'] for d in self.data]
        efficiencies = [d['throughput_efficiency_pct'] for d in self.data]
        
        # Graphique 1: Histogramme comparatif débit injecté vs mesuré
        x = np.arange(len(rates))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, rates, width, label='Débit injecté (pps)', 
                       color='skyblue', alpha=0.8, edgecolor='navy')
        bars2 = ax1.bar(x + width/2, measured_rates, width, label='Débit mesuré (pps)', 
                       color='lightcoral', alpha=0.8, edgecolor='darkred')
        
        ax1.set_xlabel('Tests par débit', fontweight='bold')
        ax1.set_ylabel('Débit (pps)', fontweight='bold')
        ax1.set_title('Comparaison Histogramme: Débit Injecté vs Débit Mesuré\nSnort 3 - Performance réelle', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{r} pps' for r in rates])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Annotations avec valeurs exactes
        for i, (bar1, bar2, rate, measured) in enumerate(zip(bars1, bars2, rates, measured_rates)):
            ax1.annotate(f'{rate}', (bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                        ha='center', va='bottom', fontweight='bold', color='navy')
            ax1.annotate(f'{measured:.0f}', (bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                        ha='center', va='bottom', fontweight='bold', color='darkred')
        
        # Graphique 2: Histogramme efficacité en pourcentage
        bars3 = ax2.bar(range(len(rates)), efficiencies, color='gold', alpha=0.8, edgecolor='orange')
        ax2.set_xlabel('Tests par débit', fontweight='bold')
        ax2.set_ylabel('Efficacité (%)', fontweight='bold')
        ax2.set_title('Efficacité de Débit par Test\nPourcentage du débit théorique atteint', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(rates)))
        ax2.set_xticklabels([f'{r} pps' for r in rates])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 100)
        
        # Ligne de référence à 100%
        ax2.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Efficacité idéale (100%)')
        ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Seuil acceptable (80%)')
        ax2.legend()
        
        # Annotations efficacité
        for bar, eff in zip(bars3, efficiencies):
            ax2.annotate(f'{eff:.1f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontweight='bold', color='darkorange')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'histogram_debit_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Histogramme sauvegardé: histogram_debit_comparison.png")
    
    def create_histogram_cpu_analysis(self):
        """
        HISTOGRAMME 2: Analyse complète CPU avec ratios comparatifs
        
        Quatre histogrammes montrant les aspects CPU critiques:
        - Temps CPU User vs System
        - Charge CPU réelle en %
        - Paquets par seconde de CPU
        - Temps CPU par paquet
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        rates = [d['rate_pps'] for d in self.data]
        cpu_user_sec = [d['cpu_user_sec'] for d in self.data]
        cpu_sys_sec = [d['cpu_sys_sec'] for d in self.data]
        cpu_loads = [d['cpu_load_pct'] for d in self.data]
        packets_per_cpu_sec = [d['packets_per_cpu_sec'] for d in self.data]
        cpu_time_per_packet = [d['cpu_time_per_packet_usec'] for d in self.data]
        
        # Histogramme 1: Temps CPU User vs System
        x = np.arange(len(rates))
        width = 0.35
        
        bars1 = axes[0,0].bar(x - width/2, cpu_user_sec, width, label='CPU User (sec)', 
                             color='lightblue', alpha=0.8, edgecolor='blue')
        bars2 = axes[0,0].bar(x + width/2, cpu_sys_sec, width, label='CPU System (sec)', 
                             color='lightpink', alpha=0.8, edgecolor='red')
        
        axes[0,0].set_title('Temps CPU: User vs System\nComparaison par débit testé', fontweight='bold')
        axes[0,0].set_xlabel('Débits testés')
        axes[0,0].set_ylabel('Temps CPU (secondes)')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([f'{r} pps' for r in rates])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # Annotations temps CPU
        for bar1, bar2, user, sys in zip(bars1, bars2, cpu_user_sec, cpu_sys_sec):
            axes[0,0].annotate(f'{user:.1f}s', (bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='blue')
            axes[0,0].annotate(f'{sys:.1f}s', (bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='red')
        
        # Histogramme 2: Charge CPU réelle (%)
        bars3 = axes[0,1].bar(rates, cpu_loads, color='green', alpha=0.7, edgecolor='darkgreen')
        axes[0,1].set_title('Charge CPU Réelle\nPourcentage d\'utilisation processeur', fontweight='bold')
        axes[0,1].set_xlabel('Débit injecté (pps)')
        axes[0,1].set_ylabel('Charge CPU (%)')
        axes[0,1].grid(True, alpha=0.3, axis='y')
        axes[0,1].set_ylim(0, max(cpu_loads) * 1.2)
        
        # Lignes de référence
        axes[0,1].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Charge modérée (50%)')
        axes[0,1].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Charge élevée (80%)')
        axes[0,1].legend()
        
        # Annotations charge CPU
        for bar, load in zip(bars3, cpu_loads):
            axes[0,1].annotate(f'{load:.1f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='darkgreen')
        
        # Histogramme 3: Paquets par seconde de CPU (efficacité CPU)
        bars4 = axes[1,0].bar(rates, packets_per_cpu_sec, color='purple', alpha=0.7, edgecolor='darkmagenta')
        axes[1,0].set_title('Efficacité CPU\nPaquets traités par seconde de CPU', fontweight='bold')
        axes[1,0].set_xlabel('Débit injecté (pps)')
        axes[1,0].set_ylabel('Paquets/sec de CPU')
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Annotations efficacité CPU
        for bar, pps in zip(bars4, packets_per_cpu_sec):
            axes[1,0].annotate(f'{pps:.0f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='darkmagenta')
        
        # Histogramme 4: Temps CPU par paquet (µs/paquet)
        bars5 = axes[1,1].bar(rates, cpu_time_per_packet, color='orange', alpha=0.7, edgecolor='darkorange')
        axes[1,1].set_title('Temps CPU par Paquet\nCoût en µs CPU par paquet traité', fontweight='bold')
        axes[1,1].set_xlabel('Débit injecté (pps)')
        axes[1,1].set_ylabel('Temps CPU (µs/paquet)')
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        # Annotations temps par paquet
        for bar, time_pkt in zip(bars5, cpu_time_per_packet):
            axes[1,1].annotate(f'{time_pkt:.1f}µs', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='darkorange')
        
        fig.suptitle('Analyse CPU Complète - Histogrammes Comparatifs\nSnort 3 - Performance processeur détaillée', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'histogram_cpu_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Histogramme sauvegardé: histogram_cpu_analysis.png")
    
    def create_histogram_packets_cpu_ratios(self):
        """
        HISTOGRAMME 3: Ratios Paquets/CPU ultra détaillés
        
        Comparaison directe des ratios critiques:
        - Paquets/sec CPU vs Paquets/sec Runtime
        - Ratio efficacité CPU/Runtime
        - Comparaison User vs System CPU par paquet
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        rates = [d['rate_pps'] for d in self.data]
        packets_per_cpu_sec = [d['packets_per_cpu_sec'] for d in self.data]
        packets_per_runtime_sec = [d['packets_per_runtime_sec'] for d in self.data]
        cpu_efficiency_ratio = [d['cpu_efficiency_ratio'] for d in self.data]
        cpu_user_per_pkt = [d['cpu_user_time_per_packet_usec'] for d in self.data]
        cpu_sys_per_pkt = [d['cpu_sys_time_per_packet_usec'] for d in self.data]
        
        # Histogramme 1: Paquets/sec CPU vs Runtime (comparaison directe)
        x = np.arange(len(rates))
        width = 0.35
        
        bars1 = axes[0,0].bar(x - width/2, packets_per_cpu_sec, width, 
                             label='Paquets/sec CPU', color='cyan', alpha=0.8, edgecolor='teal')
        bars2 = axes[0,0].bar(x + width/2, packets_per_runtime_sec, width, 
                             label='Paquets/sec Runtime', color='yellow', alpha=0.8, edgecolor='gold')
        
        axes[0,0].set_title('Comparaison Paquets/seconde\nCPU vs Runtime - Efficacité relative', fontweight='bold')
        axes[0,0].set_xlabel('Débits testés')
        axes[0,0].set_ylabel('Paquets par seconde')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([f'{r} pps' for r in rates])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # Annotations comparaison
        for i, (bar1, bar2, cpu_pps, rt_pps) in enumerate(zip(bars1, bars2, packets_per_cpu_sec, packets_per_runtime_sec)):
            axes[0,0].annotate(f'{cpu_pps:.0f}', (bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='teal')
            axes[0,0].annotate(f'{rt_pps:.0f}', (bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='gold')
        
        # Histogramme 2: Ratio efficacité CPU/Runtime
        bars3 = axes[0,1].bar(rates, cpu_efficiency_ratio, color='magenta', alpha=0.7, edgecolor='purple')
        axes[0,1].set_title('Ratio Efficacité CPU/Runtime\nFacteur de performance CPU', fontweight='bold')
        axes[0,1].set_xlabel('Débit injecté (pps)')
        axes[0,1].set_ylabel('Ratio Efficacité')
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        # Ligne de référence ratio optimal
        axes[0,1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Ratio optimal (1.0)')
        axes[0,1].legend()
        
        # Annotations ratio
        for bar, ratio in zip(bars3, cpu_efficiency_ratio):
            axes[0,1].annotate(f'{ratio:.2f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='purple')
        
        # Histogramme 3: CPU User vs System par paquet
        bars4 = axes[1,0].bar(x - width/2, cpu_user_per_pkt, width, 
                             label='CPU User (µs/pkt)', color='lightsteelblue', alpha=0.8, edgecolor='navy')
        bars5 = axes[1,0].bar(x + width/2, cpu_sys_per_pkt, width, 
                             label='CPU System (µs/pkt)', color='mistyrose', alpha=0.8, edgecolor='crimson')
        
        axes[1,0].set_title('Temps CPU par Paquet\nUser vs System - Répartition détaillée', fontweight='bold')
        axes[1,0].set_xlabel('Débits testés')
        axes[1,0].set_ylabel('Temps CPU (µs/paquet)')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels([f'{r} pps' for r in rates])
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Annotations User/System
        for bar4, bar5, user, sys in zip(bars4, bars5, cpu_user_per_pkt, cpu_sys_per_pkt):
            axes[1,0].annotate(f'{user:.1f}', (bar4.get_x() + bar4.get_width()/2, bar4.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='navy')
            axes[1,0].annotate(f'{sys:.1f}', (bar5.get_x() + bar5.get_width()/2, bar5.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='crimson')
        
        # Histogramme 4: Ratio User/System global
        user_sys_ratios = [d['cpu_user_sys_ratio'] for d in self.data]
        bars6 = axes[1,1].bar(rates, user_sys_ratios, color='brown', alpha=0.7, edgecolor='maroon')
        axes[1,1].set_title('Ratio CPU User/System\nÉquilibre User vs Kernel', fontweight='bold')
        axes[1,1].set_xlabel('Débit injecté (pps)')
        axes[1,1].set_ylabel('Ratio User/System')
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        # Ligne de référence équilibre
        axes[1,1].axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Équilibre (1.0)')
        axes[1,1].legend()
        
        # Annotations ratio User/System
        for bar, ratio in zip(bars6, user_sys_ratios):
            axes[1,1].annotate(f'{ratio:.2f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='maroon')
        
        fig.suptitle('Ratios Paquets/CPU - Analyse Comparative Avancée\nSnort 3 - Métriques d\'efficacité détaillées', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'histogram_packets_cpu_ratios.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Histogramme sauvegardé: histogram_packets_cpu_ratios.png")
    
    def create_histogram_modules_comparison(self):
        """
        HISTOGRAMME 4: Comparaison détaillée des modules Snort
        
        Analyse comparative des performances des modules:
        - Detection vs Port Scan vs Stream
        - Efficacité par module
        - Ratios inter-modules
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        rates = [d['rate_pps'] for d in self.data]
        detection_analyzed = [d['detection_analyzed'] for d in self.data]
        port_scan_packets = [d['port_scan_packets'] for d in self.data]
        stream_flows = [d['stream_flows'] for d in self.data]
        arp_spoof_packets = [d['arp_spoof_packets'] for d in self.data]
        detection_pps = [d['detection_pps'] for d in self.data]
        flows_per_sec = [d['flows_per_sec'] for d in self.data]
        packets_per_flow = [d['packets_per_flow'] for d in self.data]
        
        # Histogramme 1: Comparaison volumes des modules principaux
        x = np.arange(len(rates))
        width = 0.2
        
        bars1 = axes[0,0].bar(x - width, detection_analyzed, width, 
                             label='Detection analyzed', color='lightcoral', alpha=0.8)
        bars2 = axes[0,0].bar(x, port_scan_packets, width, 
                             label='Port Scan packets', color='lightgreen', alpha=0.8)
        bars3 = axes[0,0].bar(x + width, arp_spoof_packets, width, 
                             label='ARP Spoof packets', color='lightblue', alpha=0.8)
        
        axes[0,0].set_title('Volume de Paquets par Module\nComparaison Detection, Port Scan, ARP Spoof', fontweight='bold')
        axes[0,0].set_xlabel('Débits testés')
        axes[0,0].set_ylabel('Nombre de paquets')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([f'{r} pps' for r in rates])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # Annotations modules
        for i, (bar1, bar2, bar3, det, port, arp) in enumerate(zip(bars1, bars2, bars3, 
                                                                  detection_analyzed, port_scan_packets, arp_spoof_packets)):
            if det > 0:
                axes[0,0].annotate(f'{det:,}', (bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                                  ha='center', va='bottom', fontsize=9, rotation=45)
            if port > 0:
                axes[0,0].annotate(f'{port:,}', (bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                                  ha='center', va='bottom', fontsize=9, rotation=45)
            if arp > 0:
                axes[0,0].annotate(f'{arp:,}', (bar3.get_x() + bar3.get_width()/2, bar3.get_height()),
                                  ha='center', va='bottom', fontsize=9, rotation=45)
        
        # Histogramme 2: Flux Stream et efficacité
        bars4 = axes[0,1].bar(x - width/2, stream_flows, width, 
                             label='Stream flows', color='gold', alpha=0.8)
        bars5 = axes[0,1].bar(x + width/2, flows_per_sec, width, 
                             label='Flows/sec', color='orange', alpha=0.8)
        
        axes[0,1].set_title('Analyse des Flux Stream\nVolume total vs Débit de création', fontweight='bold')
        axes[0,1].set_xlabel('Débits testés')
        axes[0,1].set_ylabel('Nombre de flux')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([f'{r} pps' for r in rates])
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        # Annotations flux
        for bar4, bar5, flows, fps in zip(bars4, bars5, stream_flows, flows_per_sec):
            axes[0,1].annotate(f'{flows}', (bar4.get_x() + bar4.get_width()/2, bar4.get_height()),
                              ha='center', va='bottom', fontweight='bold')
            axes[0,1].annotate(f'{fps:.1f}', (bar5.get_x() + bar5.get_width()/2, bar5.get_height()),
                              ha='center', va='bottom', fontweight='bold')
        
        # Histogramme 3: Efficacité Detection (pps)
        bars6 = axes[1,0].bar(rates, detection_pps, color='purple', alpha=0.7, edgecolor='darkmagenta')
        axes[1,0].set_title('Efficacité Module Detection\nPaquets analysés par seconde', fontweight='bold')
        axes[1,0].set_xlabel('Débit injecté (pps)')
        axes[1,0].set_ylabel('Detection pps')
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Annotations efficacité detection
        for bar, pps in zip(bars6, detection_pps):
            axes[1,0].annotate(f'{pps:.0f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='darkmagenta')
        
        # Histogramme 4: Paquets par flux (densité)
        bars7 = axes[1,1].bar(rates, packets_per_flow, color='teal', alpha=0.7, edgecolor='darkcyan')
        axes[1,1].set_title('Densité des Flux\nPaquets moyens par flux', fontweight='bold')
        axes[1,1].set_xlabel('Débit injecté (pps)')
        axes[1,1].set_ylabel('Paquets/flux')
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        # Annotations densité
        for bar, ppf in zip(bars7, packets_per_flow):
            axes[1,1].annotate(f'{ppf:.1f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color='darkcyan')
        
        fig.suptitle('Analyse Comparative des Modules Snort 3\nPerformance et efficacité détaillées', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'histogram_modules_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Histogramme sauvegardé: histogram_modules_comparison.png")
    
    def create_histogram_protocol_analysis(self):
        """
        HISTOGRAMME 5: Analyse détaillée des protocoles avec comparaisons
        
        Distribution des protocoles réseau détectés par le codec
        avec analyse comparative entre débits.
        """
        # Collecte de tous les protocoles
        all_protocols = set()
        for d in self.data:
            all_protocols.update(d['codec_data'].keys())
        
        if not all_protocols:
            print("⚠️  Aucune donnée de protocole trouvée pour l'histogramme")
            return
        
        # Sélection des protocoles les plus significatifs
        protocol_totals = {}
        for proto in all_protocols:
            protocol_totals[proto] = sum(d['codec_data'].get(proto, 0) for d in self.data)
        
        # Top 8 protocoles les plus fréquents
        top_protocols = sorted(protocol_totals.items(), key=lambda x: x[1], reverse=True)[:8]
        protocols = [p[0] for p in top_protocols]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        rates = [d['rate_pps'] for d in self.data]
        
        # Histogramme 1: Comparaison des top protocoles
        x = np.arange(len(protocols))
        width = 0.25
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        for i, rate in enumerate(rates):
            data_for_rate = next(d for d in self.data if d['rate_pps'] == rate)
            counts = [data_for_rate['codec_data'].get(proto, 0) for proto in protocols]
            
            offset = (i - len(rates)//2) * width
            bars = axes[0,0].bar(x + offset, counts, width, 
                               label=f'{rate} pps', alpha=0.8, color=colors[i % len(colors)])
            
            # Annotations pour valeurs significatives
            for j, (bar, count) in enumerate(zip(bars, counts)):
                if count > 1000:  # Annoter seulement les valeurs importantes
                    axes[0,0].annotate(f'{count:,}', 
                                      (bar.get_x() + bar.get_width()/2, bar.get_height()), 
                                      ha='center', va='bottom', rotation=45, fontsize=8)
        
        axes[0,0].set_title('Distribution des Top Protocoles\nComparaison par débit testé', fontweight='bold')
        axes[0,0].set_xlabel('Protocoles réseau')
        axes[0,0].set_ylabel('Nombre de paquets détectés')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(protocols, rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # Histogramme 2: Pourcentage de distribution pour chaque débit
        for i, rate in enumerate(rates):
            data_for_rate = next(d for d in self.data if d['rate_pps'] == rate)
            total_packets = sum(data_for_rate['codec_data'].values())
            
            if total_packets > 0:
                percentages = [(data_for_rate['codec_data'].get(proto, 0) / total_packets) * 100 
                              for proto in protocols]
                
                axes[0,1].bar(x + i * width - width, percentages, width, 
                             label=f'{rate} pps', alpha=0.8, color=colors[i % len(colors)])
        
        axes[0,1].set_title('Distribution Relative des Protocoles\nPourcentage par débit', fontweight='bold')
        axes[0,1].set_xlabel('Protocoles réseau')
        axes[0,1].set_ylabel('Pourcentage (%)')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(protocols, rotation=45, ha='right')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        # Histogramme 3: Focus sur les protocoles principaux (ARP, ETH, IPv4, IPv6)
        main_protocols = ['arp', 'eth', 'ipv4', 'ipv6', 'tcp', 'udp']
        existing_main = [p for p in main_protocols if p in all_protocols]
        
        if existing_main:
            x_main = np.arange(len(existing_main))
            
            for i, rate in enumerate(rates):
                data_for_rate = next(d for d in self.data if d['rate_pps'] == rate)
                counts_main = [data_for_rate['codec_data'].get(proto, 0) for proto in existing_main]
                
                axes[1,0].bar(x_main + i * width - width/2, counts_main, width, 
                             label=f'{rate} pps', alpha=0.8, color=colors[i % len(colors)])
            
            axes[1,0].set_title('Protocoles Principaux\nARP, ETH, IPv4, IPv6, TCP, UDP', fontweight='bold')
            axes[1,0].set_xlabel('Protocoles principaux')
            axes[1,0].set_ylabel('Nombre de paquets')
            axes[1,0].set_xticks(x_main)
            axes[1,0].set_xticklabels(existing_main)
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Histogramme 4: Évolution des protocoles avec le débit
        if 'arp' in all_protocols and 'ipv4' in all_protocols:
            arp_counts = [d['codec_data'].get('arp', 0) for d in self.data]
            ipv4_counts = [d['codec_data'].get('ipv4', 0) for d in self.data]
            tcp_counts = [d['codec_data'].get('tcp', 0) for d in self.data]
            udp_counts = [d['codec_data'].get('udp', 0) for d in self.data]
            
            x_evo = np.arange(len(rates))
            width_evo = 0.2
            
            axes[1,1].bar(x_evo - 1.5*width_evo, arp_counts, width_evo, 
                         label='ARP', alpha=0.8, color='gold')
            axes[1,1].bar(x_evo - 0.5*width_evo, ipv4_counts, width_evo, 
                         label='IPv4', alpha=0.8, color='lightblue')
            axes[1,1].bar(x_evo + 0.5*width_evo, tcp_counts, width_evo, 
                         label='TCP', alpha=0.8, color='lightcoral')
            axes[1,1].bar(x_evo + 1.5*width_evo, udp_counts, width_evo, 
                         label='UDP', alpha=0.8, color='lightgreen')
            
            axes[1,1].set_title('Évolution Protocoles vs Débit\nARP, IPv4, TCP, UDP', fontweight='bold')
            axes[1,1].set_xlabel('Débits testés')
            axes[1,1].set_ylabel('Nombre de paquets')
            axes[1,1].set_xticks(x_evo)
            axes[1,1].set_xticklabels([f'{r} pps' for r in rates])
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Analyse Détaillée des Protocoles Réseau\nCodec Snort 3 - Distribution comparative', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'histogram_protocol_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Histogramme sauvegardé: histogram_protocol_analysis.png")
    
    def create_histogram_performance_summary(self):
        """
        HISTOGRAMME 6: Résumé de performance avec métriques clés
        
        Vue d'ensemble avec les 4 métriques les plus importantes
        sous forme d'histogrammes comparatifs.
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        rates = [d['rate_pps'] for d in self.data]
        throughput_eff = [d['throughput_efficiency_pct'] for d in self.data]
        packet_loss = [d['packet_loss_pct'] for d in self.data]
        cpu_load = [d['cpu_load_pct'] for d in self.data]
        detection_eff = [d['detection_efficiency_pct'] for d in self.data]
        
        # Histogramme 1: Efficacité de débit (métrique principale)
        bars1 = axes[0,0].bar(rates, throughput_eff, color='green', alpha=0.7, edgecolor='darkgreen')
        axes[0,0].set_title('🎯 Efficacité de Débit\nPourcentage du débit théorique atteint', fontweight='bold')
        axes[0,0].set_xlabel('Débit injecté (pps)')
        axes[0,0].set_ylabel('Efficacité (%)')
        axes[0,0].set_ylim(0, 100)
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # Lignes de référence efficacité
        axes[0,0].axhline(y=100, color='blue', linestyle='--', alpha=0.7, label='Idéal (100%)')
        axes[0,0].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Bon (80%)')
        axes[0,0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Acceptable (50%)')
        axes[0,0].legend()
        
        # Annotations avec couleurs selon performance
        for bar, eff in zip(bars1, throughput_eff):
            color = 'green' if eff >= 80 else 'orange' if eff >= 50 else 'red'
            axes[0,0].annotate(f'{eff:.1f}%', 
                              (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color=color)
        
        # Histogramme 2: Perte de paquets (inverse de l'efficacité)
        bars2 = axes[0,1].bar(rates, packet_loss, color='red', alpha=0.7, edgecolor='darkred')
        axes[0,1].set_title('❌ Perte de Paquets\nPourcentage de paquets non traités', fontweight='bold')
        axes[0,1].set_xlabel('Débit injecté (pps)')
        axes[0,1].set_ylabel('Perte (%)')
        axes[0,1].set_ylim(0, 100)
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        # Lignes de référence pertes
        axes[0,1].axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Acceptable (<5%)')
        axes[0,1].axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Limite (20%)')
        axes[0,1].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Critique (50%)')
        axes[0,1].legend()
        
               # Annotations pertes
        for bar, loss in zip(bars2, packet_loss):
            color = 'green' if loss <= 5 else 'orange' if loss <= 20 else 'red'
            axes[0,1].annotate(f'{loss:.1f}%', 
                              (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color=color)
        
        # Histogramme 3: Charge CPU réelle
        bars3 = axes[1,0].bar(rates, cpu_load, color='blue', alpha=0.7, edgecolor='navy')
        axes[1,0].set_title('⚡ Charge CPU Réelle\nPourcentage d\'utilisation processeur', fontweight='bold')
        axes[1,0].set_xlabel('Débit injecté (pps)')
        axes[1,0].set_ylabel('Charge CPU (%)')
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Lignes de référence CPU
        axes[1,0].axhline(y=25, color='green', linestyle='--', alpha=0.7, label='Faible (25%)')
        axes[1,0].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Modérée (50%)')
        axes[1,0].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Élevée (80%)')
        axes[1,0].legend()
        
        # Annotations CPU
        for bar, load in zip(bars3, cpu_load):
            color = 'green' if load <= 25 else 'orange' if load <= 50 else 'red'
            axes[1,0].annotate(f'{load:.1f}%', 
                              (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color=color)
        
        # Histogramme 4: Efficacité Detection
        bars4 = axes[1,1].bar(rates, detection_eff, color='purple', alpha=0.7, edgecolor='darkmagenta')
        axes[1,1].set_title('🔍 Efficacité Détection\nPourcentage du trafic analysé', fontweight='bold')
        axes[1,1].set_xlabel('Débit injecté (pps)')
        axes[1,1].set_ylabel('Détection (%)')
        axes[1,1].set_ylim(0, max(detection_eff) * 1.1 if detection_eff else 100)
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        # Lignes de référence detection
        axes[1,1].axhline(y=95, color='green', linestyle='--', alpha=0.7, label='Excellent (95%)')
        axes[1,1].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Bon (80%)')
        axes[1,1].axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Limite (60%)')
        axes[1,1].legend()
        
        # Annotations detection
        for bar, det_eff in zip(bars4, detection_eff):
            color = 'green' if det_eff >= 95 else 'orange' if det_eff >= 80 else 'red'
            axes[1,1].annotate(f'{det_eff:.1f}%', 
                              (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              ha='center', va='bottom', fontweight='bold', color=color)
        
        fig.suptitle('📊 RÉSUMÉ DE PERFORMANCE SNORT 3 - Métriques Clés\nVue d\'ensemble comparative des 4 indicateurs critiques', 
                    fontsize=16, fontweight='bold')
        
        # Note explicative globale
        plt.figtext(0.02, 0.02, 
                   "🔥 CODE COULEUR: Vert=Excellent, Orange=Acceptable, Rouge=Problématique | "
                   "Métriques essentielles pour évaluer les performances Snort",
                   fontsize=11, style='italic', weight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'histogram_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Histogramme sauvegardé: histogram_performance_summary.png")
    
    def create_ultimate_comparison_histogram(self):
        """
        HISTOGRAMME 7 (ULTIME): Comparaison TOTALE avec tous les ratios critiques
        
        L'histogramme le plus complet avec 6 métriques comparatives essentielles
        pour une analyse exhaustive des performances.
        """
        fig = plt.figure(figsize=(24, 18))
        
        # Création d'une grille 3x2 pour 6 graphiques
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        rates = [d['rate_pps'] for d in self.data]
        
        # === MÉTRIQUES POUR COMPARAISON ULTIME ===
        debit_measured = [d['debit_avg_pps'] for d in self.data]
        packets_per_cpu_sec = [d['packets_per_cpu_sec'] for d in self.data]
        packets_per_runtime_sec = [d['packets_per_runtime_sec'] for d in self.data]
        cpu_time_per_packet = [d['cpu_time_per_packet_usec'] for d in self.data]
        detection_analyzed = [d['detection_analyzed'] for d in self.data]
        throughput_eff = [d['throughput_efficiency_pct'] for d in self.data]
        
        # GRAPHIQUE 1: Débit mesuré vs injecté (COMPARAISON DIRECTE)
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(rates))
        width = 0.35
        
        bars1a = ax1.bar(x - width/2, rates, width, label='Débit injecté', 
                        color='lightblue', alpha=0.8, edgecolor='blue')
        bars1b = ax1.bar(x + width/2, debit_measured, width, label='Débit mesuré', 
                        color='orange', alpha=0.8, edgecolor='darkorange')
        
        ax1.set_title('📈 DÉBIT: Injecté vs Mesuré\nComparaison directe des performances', 
                     fontweight='bold', fontsize=12)
        ax1.set_xlabel('Tests')
        ax1.set_ylabel('Débit (pps)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{r}pps' for r in rates])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Annotations avec écarts
        for i, (rate, measured) in enumerate(zip(rates, debit_measured)):
            loss = ((rate - measured) / rate) * 100
            ax1.annotate(f'-{loss:.1f}%', (i, max(rate, measured) + max(rates)*0.05),
                        ha='center', fontweight='bold', color='red')
        
        # GRAPHIQUE 2: Paquets/sec CPU vs Runtime (EFFICACITÉ COMPARATIVE)
        ax2 = fig.add_subplot(gs[0, 1])
        
        bars2a = ax2.bar(x - width/2, packets_per_cpu_sec, width, label='Paquets/sec CPU', 
                        color='green', alpha=0.8, edgecolor='darkgreen')
        bars2b = ax2.bar(x + width/2, packets_per_runtime_sec, width, label='Paquets/sec Runtime', 
                        color='red', alpha=0.8, edgecolor='darkred')
        
        ax2.set_title('⚡ EFFICACITÉ: CPU vs Runtime\nComparaison des débits de traitement', 
                     fontweight='bold', fontsize=12)
        ax2.set_xlabel('Tests')
        ax2.set_ylabel('Paquets/seconde')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{r}pps' for r in rates])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Annotations ratios
        for i, (cpu_pps, rt_pps) in enumerate(zip(packets_per_cpu_sec, packets_per_runtime_sec)):
            ratio = cpu_pps / rt_pps if rt_pps > 0 else 0
            ax2.annotate(f'×{ratio:.1f}', (i, max(cpu_pps, rt_pps) + max(packets_per_cpu_sec)*0.05),
                        ha='center', fontweight='bold', color='purple')
        
        # GRAPHIQUE 3: Temps CPU par paquet (COÛT UNITAIRE)
        ax3 = fig.add_subplot(gs[1, 0])
        
        bars3 = ax3.bar(rates, cpu_time_per_packet, color='purple', alpha=0.8, edgecolor='darkmagenta')
        ax3.set_title('🕐 COÛT CPU par Paquet\nTemps de traitement unitaire', 
                     fontweight='bold', fontsize=12)
        ax3.set_xlabel('Débit injecté (pps)')
        ax3.set_ylabel('Temps CPU (µs/paquet)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Annotations avec tendance
        for i, (bar, time_pkt) in enumerate(zip(bars3, cpu_time_per_packet)):
            ax3.annotate(f'{time_pkt:.1f}µs', 
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontweight='bold')
            
            # Indication de tendance
            if i > 0:
                prev_time = cpu_time_per_packet[i-1]
                trend = "↗" if time_pkt > prev_time else "↘" if time_pkt < prev_time else "→"
                ax3.annotate(trend, (bar.get_x() + bar.get_width()/2, bar.get_height() * 0.7),
                            ha='center', fontsize=16, color='red' if trend == "↗" else 'green')
        
        # GRAPHIQUE 4: Detection Analyzed (VOLUME DE TRAVAIL)
        ax4 = fig.add_subplot(gs[1, 1])
        
        bars4 = ax4.bar(rates, detection_analyzed, color='teal', alpha=0.8, edgecolor='darkcyan')
        ax4.set_title('🔍 VOLUME Détection\nPaquets analysés par le moteur', 
                     fontweight='bold', fontsize=12)
        ax4.set_xlabel('Débit injecté (pps)')
        ax4.set_ylabel('Paquets analysés')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Annotations avec pourcentage du débit injecté
        for bar, analyzed, rate in zip(bars4, detection_analyzed, rates):
            percentage = (analyzed / (rate * self.data[rates.index(rate)]['runtime_sec'])) * 100 if rate > 0 else 0
            ax4.annotate(f'{analyzed:,}\n({percentage:.1f}%)', 
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontweight='bold')
        
        # GRAPHIQUE 5: Efficacité globale (PERFORMANCE RELATIVE)
        ax5 = fig.add_subplot(gs[2, :])  # Graphique sur toute la largeur
        
        # Métriques normalisées pour comparaison (0-100)
        eff_debit_norm = throughput_eff  # Déjà en %
        eff_cpu_norm = [min((pps / 10000) * 100, 100) for pps in packets_per_cpu_sec]  # Normalisé arbitrairement
        eff_detection_norm = [(analyzed / (rate * runtime)) * 100 if rate > 0 else 0 
                             for analyzed, rate, runtime in zip(detection_analyzed, rates, 
                                                               [d['runtime_sec'] for d in self.data])]
        
        x_wide = np.arange(len(rates))
        width_wide = 0.25
        
        bars5a = ax5.bar(x_wide - width_wide, eff_debit_norm, width_wide, 
                        label='Efficacité Débit (%)', color='blue', alpha=0.7)
        bars5b = ax5.bar(x_wide, eff_cpu_norm, width_wide, 
                        label='Efficacité CPU (normalisée)', color='green', alpha=0.7)
        bars5c = ax5.bar(x_wide + width_wide, eff_detection_norm, width_wide, 
                        label='Efficacité Détection (%)', color='red', alpha=0.7)
        
        ax5.set_title('🏆 EFFICACITÉ GLOBALE COMPARATIVE\nTrois dimensions de performance normalisées', 
                     fontweight='bold', fontsize=14)
        ax5.set_xlabel('Débits testés', fontweight='bold')
        ax5.set_ylabel('Efficacité normalisée (%)', fontweight='bold')
        ax5.set_xticks(x_wide)
        ax5.set_xticklabels([f'{r} pps' for r in rates])
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim(0, 120)
        
        # Lignes de référence performance
        ax5.axhline(y=100, color='gold', linestyle='--', alpha=0.8, linewidth=2, label='Performance idéale')
        ax5.axhline(y=80, color='orange', linestyle='--', alpha=0.6, label='Performance acceptable')
        ax5.axhline(y=50, color='red', linestyle='--', alpha=0.6, label='Performance limite')
        
        # Annotations finales avec scores globaux
        for i, (debit_eff, cpu_eff, det_eff) in enumerate(zip(eff_debit_norm, eff_cpu_norm, eff_detection_norm)):
            score_global = (debit_eff + cpu_eff + det_eff) / 3
            ax5.annotate(f'Score: {score_global:.1f}', 
                        (i, max(debit_eff, cpu_eff, det_eff) + 10),
                        ha='center', fontweight='bold', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Titre et notes finales
        fig.suptitle('🔥 ANALYSE ULTIME SNORT 3 - COMPARAISON EXHAUSTIVE 🔥\n'
                    'Histogrammes comparatifs complets pour évaluation de performance', 
                    fontsize=18, fontweight='bold', color='darkblue')
        
        plt.figtext(0.02, 0.02, 
                   "💡 LÉGENDE: Ce graphique compile TOUTES les métriques critiques pour une analyse complète. "
                   "Score global = moyenne des 3 efficacités principales.",
                   fontsize=12, style='italic', weight='bold', color='darkgreen')
        
        plt.savefig(self.figures_dir / 'histogram_ultimate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("🔥 Histogramme ULTIME sauvegardé: histogram_ultimate_comparison.png")
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """
        Crée un DataFrame résumé COMPLET avec toutes les métriques calculées
        """
        df_data = []
        
        for d in self.data:
            row = {
                # === DONNÉES DE BASE ===
                'rate_pps': d['rate_pps'],
                'packets_total': d['packets_total'],
                'runtime_sec': d['runtime_sec'],
                'debit_avg_pps': d['debit_avg_pps'],
                'latency_avg_usecs': d['latency_avg_usecs'],
                'latency_max_usecs': d['latency_max_usecs'],
                
                # === CPU DÉTAILLÉ ===
                'cpu_user_usec': d['cpu_user_usec'],
                'cpu_sys_usec': d['cpu_sys_usec'],
                'cpu_total_sec': d['cpu_total_sec'],
                'cpu_load_pct': d['cpu_load_pct'],
                'cpu_time_per_packet_usec': d['cpu_time_per_packet_usec'],
                'cpu_user_sys_ratio': d['cpu_user_sys_ratio'],
                
                # === EFFICACITÉ ===
                'throughput_efficiency_pct': d['throughput_efficiency_pct'],
                'packet_loss_pct': d['packet_loss_pct'],
                'packets_per_cpu_sec': d['packets_per_cpu_sec'],
                'packets_per_runtime_sec': d['packets_per_runtime_sec'],
                'cpu_efficiency_ratio': d['cpu_efficiency_ratio'],
                
                # === MODULES ===
                'detection_analyzed': d['detection_analyzed'],
                'detection_efficiency_pct': d['detection_efficiency_pct'],
                'detection_pps': d['detection_pps'],
                'port_scan_packets': d['port_scan_packets'],
                'port_scan_trackers': d['port_scan_trackers'],
                'stream_flows': d['stream_flows'],
                'flows_per_sec': d['flows_per_sec'],
                'packets_per_flow': d['packets_per_flow'],
                'stream_total_prunes': d['stream_total_prunes'],
                'stream_idle_prunes': d['stream_idle_prunes'],
                'arp_spoof_packets': d['arp_spoof_packets'],
                'binder_raw_packets': d['binder_raw_packets'],
                'binder_new_flows': d['binder_new_flows'],
                'binder_inspects': d['binder_inspects']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return df
    
    def print_ultimate_summary_statistics(self, df: pd.DataFrame):
        """
        Affiche des statistiques ULTRA DÉTAILLÉES et analytiques
        """
        print("\n" + "🔥"*100)
        print("🚀 ANALYSE ULTIME DES PERFORMANCES SNORT 3 - RAPPORT COMPLET 🚀")
        print("🔥"*100)
        
        print(f"\n📋 CONFIGURATION DES TESTS:")
        print(f"   🎯 Nombre de débits testés: {len(df)}")
        print(f"   🎯 Débits injectés: {', '.join(map(str, sorted(df['rate_pps'].tolist())))} pps")
        print(f"   🎯 Durée totale des tests: {df['runtime_sec'].sum():.0f} secondes ({df['runtime_sec'].sum()/60:.1f} minutes)")
        print(f"   🎯 Paquets totaux traités: {df['packets_total'].sum():,}")
        print(f"   🎯 Temps CPU total consommé: {df['cpu_total_sec'].sum():.1f} secondes")
        
        print(f"\n📊 PERFORMANCES MOYENNES GLOBALES:")
        print(f"   📈 Débit moyen: {df['debit_avg_pps'].mean():.2f} pps")
        print(f"   📈 Efficacité débit moyenne: {df['throughput_efficiency_pct'].mean():.1f}%")
        print(f"   📈 Perte moyenne de paquets: {df['packet_loss_pct'].mean():.1f}%")
        print(f"   📈 Latence moyenne: {df['latency_avg_usecs'].mean():.3f} µs")
        print(f"   📈 Charge CPU moyenne: {df['cpu_load_pct'].mean():.1f}%")
        print(f"   📈 Temps CPU moyen par paquet: {df['cpu_time_per_packet_usec'].mean():.2f} µs/pkt")
        print(f"   📈 Paquets/sec CPU moyen: {df['packets_per_cpu_sec'].mean():.0f} pps")
        print(f"   📈 Efficacité détection moyenne: {df['detection_efficiency_pct'].mean():.1f}%")
        
        print(f"\n🏆 RECORDS DE PERFORMANCE:")
        best_throughput_idx = df['throughput_efficiency_pct'].idxmax()
        best_cpu_eff_idx = df['packets_per_cpu_sec'].idxmax()
        best_detection_idx = df['detection_efficiency_pct'].idxmax()
        lowest_loss_idx = df['packet_loss_pct'].idxmin()
        lowest_latency_idx = df['latency_avg_usecs'].idxmin()
        
        print(f"   🥇 RECORD Efficacité débit: {df.loc[best_throughput_idx, 'throughput_efficiency_pct']:.1f}% à {df.loc[best_throughput_idx, 'rate_pps']:.0f} pps")
        print(f"   🥇 RECORD Efficacité CPU: {df.loc[best_cpu_eff_idx, 'packets_per_cpu_sec']:.0f} pkt/sec à {df.loc[best_cpu_eff_idx, 'rate_pps']:.0f} pps")
        print(f"   🥇 RECORD Efficacité détection: {df.loc[best_detection_idx, 'detection_efficiency_pct']:.1f}% à {df.loc[best_detection_idx, 'rate_pps']:.0f} pps")
        print(f"   🥇 RECORD Perte minimale: {df.loc[lowest_loss_idx, 'packet_loss_pct']:.1f}% à {df.loc[lowest_loss_idx, 'rate_pps']:.0f} pps")
        print(f"   🥇 RECORD Latence minimale: {df.loc[lowest_latency_idx, 'latency_avg_usecs']:.3f} µs à {df.loc[lowest_latency_idx, 'rate_pps']:.0f} pps")
        
        print(f"\n⚠️  ANALYSE DES PROBLÈMES:")
        
        # Détection des goulots d'étranglement
        critical_loss = df[df['packet_loss_pct'] > 70]
        high_loss = df[df['packet_loss_pct'] > 50]
        low_cpu_high_loss = df[(df['cpu_load_pct'] < 20) & (df['packet_loss_pct'] > 50)]
        high_cpu_low_throughput = df[(df['cpu_load_pct'] > 50) & (df['throughput_efficiency_pct'] < 50)]
        
        if not critical_loss.empty:
            print(f"   🔴 CRITIQUE: Pertes très élevées (>70%) à {', '.join(map(str, critical_loss['rate_pps'].tolist()))} pps")
        
        if not high_loss.empty:
            print(f"   🟠 ATTENTION: Pertes élevées (>50%) à {', '.join(map(str, high_loss['rate_pps'].tolist()))} pps")
        
        if not low_cpu_high_loss.empty:
            print(f"   🔴 GOULOT NON-CPU: Charge CPU faible mais pertes élevées → Problème réseau/I/O")
        
        if not high_cpu_low_throughput.empty:
            print(f"   🔴 GOULOT CPU: Charge CPU élevée avec faible débit → Optimisation algorithmes nécessaire")
        
        # Analyse de la scalabilité
        if len(df) > 1:
            first_eff = df.iloc[0]['throughput_efficiency_pct']
            last_eff = df.iloc[-1]['throughput_efficiency_pct']
            scalability_trend = last_eff - first_eff
            
            if scalability_trend < -20:
                print(f"   📉 SCALABILITÉ MÉDIOCRE: Dégradation de {abs(scalability_trend):.1f}% entre débits min/max")
            elif scalability_trend > 5:
                print(f"   📈 BONNE SCALABILITÉ: Amélioration de {scalability_trend:.1f}% avec la charge")
            else:
                print(f"   📊 SCALABILITÉ STABLE: Variation de {scalability_trend:.1f}% entre débits")
        
        print(f"\n🔍 ANALYSE DÉTAILLÉE PAR DÉBIT:")
        for _, row in df.iterrows():
            # Évaluation de la performance globale
            score_debit = min(row['throughput_efficiency_pct'], 100)
            score_cpu = min((row['packets_per_cpu_sec'] / 1000) * 10, 100)  # Normalisé
            score_detection = min(row['detection_efficiency_pct'], 100)
            score_global = (score_debit + score_cpu + score_detection) / 3
            
            # Icône de performance
            if score_global >= 80:
                perf_icon = "🟢 EXCELLENT"
            elif score_global >= 60:
                perf_icon = "🟡 CORRECT"
            elif score_global >= 40:
                perf_icon = "🟠 PASSABLE"
            else:
                perf_icon = "🔴 CRITIQUE"
            
            print(f"\n   📌 {int(row['rate_pps'])} pps - {perf_icon} (Score: {score_global:.1f}/100)")
            print(f"      ├─ 📊 Performance: {row['debit_avg_pps']:.1f} pps mesurés ({row['throughput_efficiency_pct']:.1f}% efficacité)")
            print(f"      ├─ ❌ Pertes: {row['packet_loss_pct']:.1f}% de paquets perdus")
            print(f"      ├─ 🕐 Latence: {row['latency_avg_usecs']:.3f} µs (max: {row['latency_max_usecs']:.0f} µs)")
            print(f"      ├─ ⚡ CPU: {row['cpu_load_pct']:.1f}% charge ({row['cpu_time_per_packet_usec']:.2f} µs/pkt)")
            print(f"      ├─ 🚀 Efficacité CPU: {row['packets_per_cpu_sec']:.0f} pkt/sec CPU")
            print(f"      ├─ 🔍 Détection: {row['detection_analyzed']:,} paquets ({row['detection_efficiency_pct']:.1f}%)")
            print(f"      ├─ 🌊 Stream: {row['stream_flows']} flux ({row['flows_per_sec']:.1f} flux/sec)")
            print(f"      └─ 📦 Paquets/flux: {row['packets_per_flow']:.1f}")
        
        print(f"\n🎯 RECOMMANDATIONS D'OPTIMISATION:")
        
        # Recommandations basées sur l'analyse
        avg_cpu_load = df['cpu_load_pct'].mean()
        avg_loss = df['packet_loss_pct'].mean()
        avg_efficiency = df['throughput_efficiency_pct'].mean()
        
        if avg_cpu_load < 25 and avg_loss > 30:
            print(f"   💡 PRIORITÉ 1: Optimiser le réseau/I/O (CPU sous-utilisé: {avg_cpu_load:.1f}%)")
        elif avg_cpu_load > 70:
            print(f"   💡 PRIORITÉ 1: Optimiser les algorithmes CPU (charge élevée: {avg_cpu_load:.1f}%)")
        
        if avg_efficiency < 50:
            print(f"   💡 PRIORITÉ 2: Configuration Snort sous-optimale (efficacité: {avg_efficiency:.1f}%)")
        
        # Débit optimal recommandé
        best_efficiency_rate = df.loc[df['throughput_efficiency_pct'].idxmax(), 'rate_pps']
        print(f"   💡 RECOMMANDATION: Débit optimal pour production = {best_efficiency_rate:.0f} pps")
        
        # Capacité maximale estimée
        max_theoretical_pps = df['packets_per_cpu_sec'].max() * (df['cpu_load_pct'].min() / 100)
        if max_theoretical_pps > 0:
            print(f"   💡 ESTIMATION: Capacité théorique maximale ≈ {max_theoretical_pps:.0f} pps")
        
        print(f"\n📈 MÉTRIQUES TECHNIQUES AVANCÉES:")
        print(f"   🔧 Ratio User/System CPU moyen: {df['cpu_user_sys_ratio'].mean():.2f}")
        print(f"   🔧 Efficacité CPU/Runtime moyenne: {df['cpu_efficiency_ratio'].mean():.2f}")
        print(f"   🔧 Paquets par flux moyen: {df['packets_per_flow'].mean():.1f}")
        print(f"   🔧 Flux créés par seconde moyen: {df['flows_per_sec'].mean():.1f}")
        
        print(f"\n📋 RÉSUMÉ EXÉCUTIF:")
        if avg_efficiency > 80:
            verdict = "🟢 PERFORMANCE EXCELLENTE"
            recommendation = "Configuration prête pour production"
        elif avg_efficiency > 60:
            verdict = "🟡 PERFORMANCE CORRECTE"
            recommendation = "Optimisations mineures recommandées"
        elif avg_efficiency > 40:
            verdict = "🟠 PERFORMANCE PASSABLE"
            recommendation = "Optimisations importantes nécessaires"
        else:
            verdict = "🔴 PERFORMANCE CRITIQUE"
            recommendation = "Révision complète de la configuration requise"
        
        print(f"   🎖️  VERDICT: {verdict}")
        print(f"   🎖️  EFFICACITÉ GLOBALE: {avg_efficiency:.1f}%")
        print(f"   🎖️  RECOMMANDATION: {recommendation}")
        print(f"   🎖️  DÉBIT OPTIMAL: {best_efficiency_rate:.0f} pps")
        
        # Timestamp de l'analyse
        print(f"\n🕐 Analyse générée le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"👤 Utilisateur: theTigerFox")
        print("🔥"*100)
    
    def run_ultimate_analysis(self):
        """
        Lance l'analyse ULTIME complète avec tous les histogrammes
        """
        print("🔥🔥🔥 LANCEMENT DE L'ANALYSE ULTIME SNORT 3 🔥🔥🔥")
        print("="*80)
        
        # Collecte des données avec parsing correct
        self.collect_data()
        
        if not self.data:
            print("❌ Aucune donnée collectée. Vérifiez vos fichiers de résultats.")
            return
        
        print(f"\n✅ Données collectées pour {len(self.data)} configurations de débit")
        
        # Génération de TOUS les histogrammes
        print("\n🎨 Génération des histogrammes ULTIMES...")
        
        try:
            print("   📊 1/7 - Histogramme comparaison débits...")
            self.create_histogram_debit_comparison()
            
            print("   📊 2/7 - Histogramme analyse CPU...")
            self.create_histogram_cpu_analysis()
            
            print("   📊 3/7 - Histogramme ratios Paquets/CPU...")
            self.create_histogram_packets_cpu_ratios()
            
            print("   📊 4/7 - Histogramme comparaison modules...")
            self.create_histogram_modules_comparison()
            
            print("   📊 5/7 - Histogramme analyse protocoles...")
            self.create_histogram_protocol_analysis()
            
            print("   📊 6/7 - Histogramme résumé performance...")
            self.create_histogram_performance_summary()
            
            print("   📊 7/7 - Histogramme ULTIME comparaison...")
            self.create_ultimate_comparison_histogram()
            
            print(f"\n🎉 7 HISTOGRAMMES GÉNÉRÉS avec succès dans {self.figures_dir}")
            
        except Exception as e:
            print(f"⚠️  Erreur lors de la génération d'un histogramme: {e}")
            import traceback
            traceback.print_exc()
        
        # Création du DataFrame résumé complet
        print("\n📋 Création du résumé ULTRA-DÉTAILLÉ...")
        df = self.create_summary_dataframe()
        
        # Sauvegarde du CSV ultra-complet
        csv_path = self.results_dir / 'summary_ultimate.csv'
        df.to_csv(csv_path, index=False)
        print(f"💾 Résumé ULTIME sauvegardé: {csv_path}")
        
        # Affichage du DataFrame
        if ACE_TOOLS_AVAILABLE:
            try:
                ace_tools.display_dataframe_to_user(
                    name="🔥 ANALYSE ULTIME SNORT 3 - DONNÉES COMPLÈTES 🔥",
                    dataframe=df,
                    description="Analyse exhaustive avec TOUTES les métriques, ratios et comparaisons"
                )
            except Exception as e:
                print(f"⚠️  Erreur ace_tools: {e}")
                print("\n📊 DATAFRAME RÉSUMÉ ULTIME:")
                # Affichage des colonnes les plus importantes
                key_columns = ['rate_pps', 'debit_avg_pps', 'throughput_efficiency_pct', 
                             'packet_loss_pct', 'cpu_load_pct', 'packets_per_cpu_sec',
                             'detection_efficiency_pct']
                print(df[key_columns].to_string(index=False))
        else:
            print("\n📊 DATAFRAME RÉSUMÉ ULTIME (colonnes clés):")
            key_columns = ['rate_pps', 'debit_avg_pps', 'throughput_efficiency_pct', 
                         'packet_loss_pct', 'cpu_load_pct', 'packets_per_cpu_sec',
                         'detection_efficiency_pct']
            print(df[key_columns].to_string(index=False))
        
        # Statistiques résumées ULTRA-DÉTAILLÉES
        self.print_ultimate_summary_statistics(df)
        
        print(f"\n🎊🎊🎊 ANALYSE ULTIME TERMINÉE AVEC SUCCÈS ! 🎊🎊🎊")
        print(f"📁 FICHIERS GÉNÉRÉS dans {self.results_dir}:")
        print(f"   🎨 7 histogrammes PNG dans {self.figures_dir}/")
        print(f"   📄 summary_ultimate.csv avec TOUTES les métriques")
        print(f"   📊 Analyse comparative EXHAUSTIVE affichée ci-dessus")
        print(f"   🔥 Recommandations d'optimisation personnalisées")
        print("\n" + "🚀"*50 + " MISSION ACCOMPLIE " + "🚀"*50)

def main():
    """
    Fonction principale avec gestion d'arguments améliorée
    """
    if len(sys.argv) == 1:
        # Si aucun argument, utiliser le répertoire courant
        results_dir = "."
        print("ℹ️  Aucun répertoire spécifié, utilisation du répertoire courant")
    elif len(sys.argv) == 2:
        results_dir = sys.argv[1]
    else:
        print("Usage: python3 analyze_snort_results_ultimate.py [/chemin/vers/results]")
        print("Exemple: python3 analyze_snort_results_ultimate.py .")
        print("Exemple: python3 analyze_snort_results_ultimate.py /path/to/results")
        sys.exit(1)
    
    if not os.path.exists(results_dir):
        print(f"❌ Le répertoire {results_dir} n'existe pas")
        sys.exit(1)
    
    print(f"🎯 Analyse des résultats dans: {os.path.abspath(results_dir)}")
    
    analyzer = SnortUltimateAnalyzer(results_dir)
    analyzer.run_ultimate_analysis()

if __name__ == "__main__":
    main()
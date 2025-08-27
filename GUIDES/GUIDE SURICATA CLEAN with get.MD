# 📊 Workflow complet : Analyse du débit (paquets/s) avec Suricata et nping

## 🎯 Objectif

Mesurer le débit de traitement de paquets par **Suricata** en mode IDS passif, en utilisant la métrique `capture.afpacket.poll_data`.
Le trafic sera généré depuis une machine Windows via **nping**.
On produira ensuite des **graphiques PNG** de l’évolution et de la distribution du débit pour différents volumes de charge.

---

## 🌐 Environnement de test

* **Machine Debian 12** avec Suricata (IP : `192.168.100.135`)
* **Machine Windows** avec nping (IP : `192.168.100.20`)

---

## ✅ 1. Installation de Suricata (Debian 12)

```bash
sudo apt update
sudo apt install -y suricata jq python3-pandas python3-matplotlib
suricata --version   # vérifier version (6.x ou 7.x attendu)
```

---

## ⚙️ 2. Configuration de Suricata (mode IDS)

Éditer le fichier `/etc/suricata/suricata.yaml` :

```bash
sudo nano /etc/suricata/suricata.yaml
```

### Section `af-packet`

Adapter à ton interface réseau (ex: `wlp6s0`, `eth0`, etc.) :

```yaml
af-packet:
  - interface: wlp6s0
    threads: auto
    cluster-id: 99
    cluster-type: cluster_flow
    defrag: yes
    use-mmap: yes
    tpacket-v3: no
    block-size: 32768
    buffer-size: 32768
    copy-mode: none
```

### Section `outputs`

Activer uniquement les stats :

```yaml
outputs:
  - stats:
      enabled: yes
      filename: stats.log
      append: yes
      totals: yes
      threads: no

  - eve-log:
      enabled: no
```

### Redémarrer Suricata

```bash
sudo systemctl restart suricata
sudo systemctl status suricata
```

---

## 🚀 3. Génération de trafic avec nping (Windows)

Sur la machine Windows (PowerShell en mode admin) :

```powershell
nping --tcp -c 1000000 --rate 1000 192.168.100.135
nping --tcp -c 1000000 --rate 5000 192.168.100.135
nping --tcp -c 1000000 --rate 25000 192.168.100.135
```

Chaque test dure environ **2 minutes**.

---

## 🔎 4. Extraction des métriques `poll_data`

### Nettoyage avant un nouveau test

```bash
sudo rm -f /var/log/suricata/stats.log
sudo systemctl restart suricata
sleep 5
sudo systemctl stop suricata
```

### Script d’extraction : `extract_debit.sh`

```bash
cat << 'EOF' > ~/extract_debit.sh
#!/usr/bin/env bash
RATE=$1
LOG=/var/log/suricata/stats.log
OUT=~/tests/${RATE}.txt
mkdir -p ~/tests

awk -v out="$OUT" '
  /^Date: / { ts = $4; next }
  /capture\.afpacket\.poll_data/ {
    print ts "," $NF > out
  }
' "$LOG"

echo "→ Débit extrait dans $OUT"
EOF

chmod +x ~/extract_debit.sh
```

### Exécution après chaque test nping

```bash
~/extract_debit.sh 1000
~/extract_debit.sh 5000
~/extract_debit.sh 25000
```

---

## 📊 5. Analyse et graphes (Python)

Créer le script `analyze_debit.py` :

```bash
cat << 'EOF' > ~/analyze_debit.py
import pandas as pd
import matplotlib.pyplot as plt
import os

tests = {
    1000: "1000.txt",
    5000: "5000.txt",
    25000: "25000.txt"
}

def load_throughput(filepath):
    df = pd.read_csv(filepath, names=['temps','cum_pkts'], parse_dates=['temps'],
                     date_parser=lambda s: pd.to_datetime(s, format='%H:%M:%S'))
    df['debit'] = df['cum_pkts'].diff().fillna(0).astype(int)
    return df

moyennes = []
outdir = "/home/fox/tests"
os.makedirs(outdir, exist_ok=True)

for rate, fname in tests.items():
    df = load_throughput(f"{outdir}/{fname}")

    # Courbe évolution
    plt.figure()
    plt.plot(df['temps'], df['debit'], marker='o')
    plt.title(f"Évolution du débit @ {rate} pkt/s")
    plt.xlabel("Temps"); plt.ylabel("Paquets/s")
    plt.tight_layout()
    plt.savefig(f"{outdir}/evolution_{rate}.png")
    plt.close()

    # Histogramme
    plt.figure()
    plt.hist(df['debit'], bins=10, edgecolor='black')
    plt.title(f"Distribution du débit @ {rate} pkt/s")
    plt.xlabel("Paquets/s"); plt.ylabel("Occurrences")
    plt.tight_layout()
    plt.savefig(f"{outdir}/histogramme_{rate}.png")
    plt.close()

    # Moyenne
    moyennes.append({'rate_pkt_s': rate, 'debit_moyen': int(df['debit'].mean())})

dfm = pd.DataFrame(moyennes).sort_values('rate_pkt_s')

# Graphique global
plt.figure()
plt.plot(dfm['rate_pkt_s'], dfm['debit_moyen'], marker='o')
plt.title("Débit moyen vs rate nping")
plt.xlabel("Rate (pkt/s)"); plt.ylabel("Débit moyen (pkt/s)")
plt.tight_layout()
plt.savefig(f"{outdir}/debit_moyen_vs_rate.png")
plt.close()

print("=== Débits moyens observés ===")
print(dfm.to_string(index=False))
EOF

chmod +x ~/analyze_debit.py
```

### Lancer l’analyse

```bash
cd ~/tests
python3 ~/analyze_debit.py
```

---

## 📈 6. Résultats générés

* Graphes PNG :

  * `evolution_1000.png`, `evolution_5000.png`, `evolution_25000.png`
  * `histogramme_*.png`
  * `debit_moyen_vs_rate.png`
* Tableau récapitulatif : débits moyens par charge

---

## 📝 Récapitulatif du workflow

1. **Installer** Suricata et dépendances
2. **Configurer** l’interface réseau + activer `stats.log`
3. **Générer du trafic** avec nping depuis Windows
4. **Extraire les données** (`poll_data`) avec un script Bash
5. **Analyser et tracer** les graphes avec Python
6. **Interpréter** les performances de Suricata selon la charge simulée



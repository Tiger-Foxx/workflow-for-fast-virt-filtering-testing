\*\*Workflow complet : analyse du débit (paquets/s) avec Suricata et nping\*\*



\### 📊 Objectif



Mesurer le débit de traitement de paquets par Suricata en mode IDS passif, en utilisant `capture.afpacket.poll\_data`, avec un générateur de trafic `nping` depuis Windows. Produire des graphes PNG de l'évolution et distribution du débit pour différents volumes d'attaque.



\### 🌐 Environnement



\* \*\*Machine Linux (Ubuntu 22.04)\*\* avec Suricata (IP : `192.168.100.135`)

\* \*\*Machine Windows\*\* avec nping (IP : `192.168.100.20`)



---



\### ✅ 1. Installation de Suricata



```bash

sudo apt update

sudo apt install -y suricata jq

suricata --version  # vérifier version 6.x ou 7.x

```



---



\### ⚖️ 2. Configuration de Suricata (mode IDS)



Modifier `/etc/suricata/suricata.yaml` :



```bash

sudo nano /etc/suricata/suricata.yaml

```



\#### Section `af-packet:`



```yaml

af-packet:

&nbsp; - interface: wlp6s0

&nbsp;   threads: auto

&nbsp;   cluster-id: 99

&nbsp;   cluster-type: cluster\_flow

&nbsp;   defrag: yes

&nbsp;   use-mmap: yes

&nbsp;   tpacket-v3: no

&nbsp;   block-size: 32768

&nbsp;   buffer-size: 32768

&nbsp;   copy-mode: none

```



\#### Section `outputs:`



```yaml

outputs:

&nbsp; - stats:

&nbsp;     enabled: yes

&nbsp;     filename: stats.log

&nbsp;     append: yes

&nbsp;     totals: yes

&nbsp;     threads: no

&nbsp; - eve-log:

&nbsp;     enabled: no

```



\#### Redémarrer Suricata



```bash

sudo systemctl restart suricata

sudo systemctl status suricata

```



---



\### 🚀 3. Génération de trafic avec nping (Windows)



Dans PowerShell admin :



```powershell

nping --tcp -c 1000000 --rate 1000 192.168.100.135

nping --tcp -c 1000000 --rate 5000 192.168.100.135

nping --tcp -c 1000000 --rate 25000 192.168.100.135

```



Durée \\~2 minutes par commande.



---



\### 🔢 4. Extraction métrique `poll\_data`



\#### Nettoyage



```bash

sudo rm /var/log/suricata/stats.log

sudo systemctl restart suricata

sleep 5

sudo systemctl stop suricata

```



\#### Script Bash : `extract\_debit.sh`



```bash

cat << 'EOF' > ~/extract\_debit.sh

\#!/usr/bin/env bash

RATE=$1

LOG=/var/log/suricata/stats.log

OUT=~/tests/${RATE}.txt

mkdir -p ~/tests

awk -v out="$OUT" '

&nbsp; /^Date: / { ts = $4; next }

&nbsp; /capture\\.afpacket\\.poll\_data/ {

&nbsp;   print ts "," $NF > out

&nbsp; }

' "$LOG"

echo "→ Débit extrait dans $OUT"

EOF

chmod +x ~/extract\_debit.sh

```



\#### Exécution pour chaque test



```bash

~/extract\_debit.sh 1000

~/extract\_debit.sh 5000

~/extract\_debit.sh 25000

```



---



\### 🧲 5. Analyse Python : `analyze\_debit.py`



```bash

cat << 'EOF' > ~/analyze\_debit.py

import pandas as pd

import matplotlib.pyplot as plt

import ace\_tools as tools



tests = {

&nbsp;   1000: "1000.txt",

&nbsp;   5000: "5000.txt",

&nbsp;   25000: "25000.txt"

}



def load\_throughput(filepath):

&nbsp;   df = pd.read\_csv(filepath, names=\['temps','cum\_pkts'], parse\_dates=\['temps'], date\_parser=lambda s: pd.to\_datetime(s, format='%H:%M:%S'))

&nbsp;   df\['debit'] = df\['cum\_pkts'].diff().fillna(0).astype(int)

&nbsp;   return df



moyennes = \[]

for rate, fname in tests.items():

&nbsp;   df = load\_throughput(f"/home/fox/tests/{fname}")

&nbsp;   plt.figure()

&nbsp;   plt.plot(df\['temps'], df\['debit'], marker='o')

&nbsp;   plt.title(f"Évolution du débit @ {rate} pkt/s")

&nbsp;   plt.xlabel("Temps"); plt.ylabel("Paquets/s")

&nbsp;   plt.tight\_layout()

&nbsp;   plt.savefig(f"/home/fox/tests/evolution\_{rate}.png")

&nbsp;   plt.close()



&nbsp;   plt.figure()

&nbsp;   plt.hist(df\['debit'], bins=10, edgecolor='black')

&nbsp;   plt.title(f"Distribution débit @ {rate} pkt/s")

&nbsp;   plt.xlabel("Paquets/s"); plt.ylabel("Nb intervalles")

&nbsp;   plt.tight\_layout()

&nbsp;   plt.savefig(f"/home/fox/tests/histogramme\_{rate}.png")

&nbsp;   plt.close()



&nbsp;   moyennes.append({'rate\_pkt\_s': rate, 'debit\_moyen': int(df\['debit'].mean())})



dfm = pd.DataFrame(moyennes).sort\_values('rate\_pkt\_s')

plt.figure()

plt.plot(dfm\['rate\_pkt\_s'], dfm\['debit\_moyen'], marker='o')

plt.title("Débit moyen vs rate nping")

plt.xlabel("Rate (pkt/s)"); plt.ylabel("Débit moyen (pkt/s)")

plt.tight\_layout()

plt.savefig("/home/fox/tests/debit\_moyen\_vs\_rate.png")

plt.close()



tools.display\_dataframe\_to\_user("Débits moyens", dfm)

EOF

chmod +x ~/analyze\_debit.py

```



Lancer l'analyse :



```bash

cd ~/tests

python3 ~/analyze\_debit.py

```



---



\### 📈 6. Sorties générées



\* Graphes :



&nbsp; \* `evolution\_1000.png`, `evolution\_5000.png`, `evolution\_25000.png`

&nbsp; \* `histogramme\_\*.png`

&nbsp; \* `debit\_moyen\_vs\_rate.png`

\* Tableau final : moyennes de débit par charge



---



\### 📆 Récapitulatif



1\. Installer Suricata

2\. Configurer l'interface + `stats.log`

3\. Lancer le trafic avec `nping`

4\. Extraire `poll\_data` avec Bash

5\. Traiter les résultats avec Python

6\. Produire les graphes PNG et tableau final



Tu disposes d’un processus simple, régulier et réutilisable pour évaluer les performances de Suricata selon la charge réseau simulée.




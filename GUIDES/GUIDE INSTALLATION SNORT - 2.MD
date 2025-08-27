# Guide ultra‑complet Snort 3 sur Debian 12 + collecte débit & latence

Ce guide détaillé vous mène **de zéro** (désinstallation de Snort 2) à la **recette complète** d’un banc de test IDS avec Snort 3 en mode AF\_PACKET, jusqu’à l’extraction automatisée de vos métriques de **débit** (pps) et **latence** (µs/paquet). Il anticipe toutes les erreurs vues (manque de `<ctime>`, modules introuvables, conf Lua, etc.) et fournit tous les scripts nécessaires.

---

## Sommaire

1. [Préparatifs & désinstallation de Snort 2](#1)
2. [Installation des dépendances](#2)
3. [Compilation & installation de DAQ](#3)
4. [Compilation & installation de Snort 3 (avec patch \`<ctime>¨)](#4)
5. [Configuration de Snort 3](#5)
6. [Vérifications & tests de conf](#6)
7. [Lancement Snort en mode IDS AF\_PACKET](#7)
8. [Génération de trafic (nping & tcpreplay)](#8)
9. [Collecte & extraction des métriques](#9)
10. [Scripts d’automatisation (bash)](#10)
11. [Analyse & visualisation (optionnel)](#11)

---

<a name="1"></a>

## 1. Préparatifs & désinstallation de Snort 2

```bash
# 1.1 Supprimer toute trace de Snort 2.x
sudo apt remove --purge snort snort-common snort-common-libraries \
                  snort-rules-default oinkmaster libluajit-5.1-2
sudo apt autoremove --purge

# 1.2 Vérifier qu'il n'existe plus de binaire snort
which snort   # ne doit rien renvoyer
```

---

<a name="2"></a>

## 2. Installation des dépendances

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config git wget \
  libpcap-dev libpcre3-dev libdumbnet-dev zlib1g-dev liblzma-dev \
  libhwloc-dev libsqlite3-dev uuid-dev libcmocka-dev \
  libluajit-5.1-dev libunwind-dev libssl-dev \
  libnetfilter-queue-dev libmnl-dev flex bison \
  libgoogle-perftools-dev    # *pour* TCMalloc
```

> **Note** : `libgoogle-perftools-dev` est nécessaire si vous activez `--enable-tcmalloc`.

---

<a name="3"></a>
\## 3. Compiler & installer DAQ

```bash
mkdir -p ~/snort3-build && cd ~/snort3-build
git clone https://github.com/snort3/libdaq.git
cd libdaq
./bootstrap
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig
```

---

<a name="4"></a>

## 4. Compiler & installer Snort 3 (avec patch `<ctime>`)

1. **Télécharger + patcher**

   ```bash
   cd ~/snort3-build
   wget https://github.com/snort3/snort3/archive/refs/tags/3.1.28.0.tar.gz
   tar xzf 3.1.28.0.tar.gz && cd snort3-3.1.28.0

   # Appliquer le missing #include <ctime> dans text_log.cc
   sed -i '/#include "log.h"/a #include <ctime>' src/log/text_log.cc
   ```

2. **Configurer & compiler**

   ```bash
   ./configure_cmake.sh --prefix=/usr/local --enable-tcmalloc
   cd build
   make -j$(nproc)
   sudo make install
   sudo ldconfig
   ```

3. **Vérifier**

   ```bash
   /usr/local/bin/snort -V
   # → Snort++ Version 3.1.28.0
   ```

---

<a name="5"></a>
\## 5. Configuration de Snort 3

1. **Créer l’arborescence**

   ```bash
   sudo mkdir -p /usr/local/etc/snort /usr/local/etc/rules /var/log/snort
   sudo chown -R $USER /usr/local/etc/snort /usr/local/etc/rules /var/log/snort
   ```

2. **Règles communautaires**

   ```bash
   wget -qO- https://www.snort.org/downloads/community/snort3-community-rules.tar.gz \
     | tar xz -C /usr/local/etc/rules
   ```

3. **Le fichier `snort.lua`**

   ```bash
   cp /usr/local/etc/snort/snort.lua.example /usr/local/etc/snort/snort.lua
   nano /usr/local/etc/snort/snort.lua
   ```

   * **1. Defaults**

     ```lua
     HOME_NET     = '192.168.100.0/24'
     EXTERNAL_NET = 'any'
     ```
   * **5. Detection**

     ```lua
     ips = {
       variables = default_variables
     }
     ```
   * **7. Outputs** : remplacez toute la section 7 par :

     ```lua
     -- 7. configure outputs
     alert_fast = {
       file   = true,
       packet = false,
       limit  = 0
     }
     eve_log = {
       enabled   = true,
       filetype  = 'regular',
       filename  = 'eve.json',
       types     = { 'alert' }
     }
     stats = {
       enabled   = true,
       filename  = 'stats.log',
       interval  = 1,
       threads   = false,
       totals    = true
     }
     ```
   * **Perf\_monitor (optionnel)**

     ```lua
     perf_monitor = {
       base     = true,  cpu     = true,
       flow     = true,  flow_ip = true,
       format   = "csv",
       output   = "file",
       summary  = true,
       packets  = 10000,
       seconds  = 60
     }
     ```

---

<a name="6"></a>
\## 6. Vérifications & tests de conf

* **Test de syntaxe**

  ```bash
  sudo /usr/local/bin/snort -c /usr/local/etc/snort/snort.lua -T
  # → Configuration validated!
  ```

* **Lister outputs reconnus**

  ```bash
    snort --help-module alert_fast
    snort --help-module stats
    snort --help-module eve-log
  ```

* **Vérifier perf\_monitor**

  ```bash
  snort --help-module perf_monitor
  ```

---

<a name="7"></a>
\## 7. Lancement en mode IDS AF\_PACKET

1. **Préparer l’interface**

   ```bash
   sudo ip link set dev eno1 promisc on
   sudo ethtool -K eno1 gro off lro off
   ```

2. **Démarrer Snort**

   ```bash
   sudo /usr/local/bin/snort \
     -c /usr/local/etc/snort/snort.lua \
     -i eno1 \
     --daq afpacket --daq-var interface=eno1 \
     -A alert_fast -q
   ```
   ou
    ```bash
   sudo /usr/local/bin/snort \
     -c /usr/local/etc/snort/snort.lua \
     -i xenbr0 \
     --daq afpacket --daq-var interface=xenbr0 \
     -A alert_fast -q
   ```

   * `-q` supprime le bruit non-alerte.

> **En mode IPS inline** ajoutez `-Q` et chargez `inline.lua` dans la conf.

---

<a name="8"></a>
\## 8. Génération de trafic

### 8.1 Avec nping (Windows)

```powershell
# Surcharge 1 000 000 paquets (~2 min)
nping --tcp -c 1000000 --rate 1000   131.254.23.28
nping --tcp -c 1000000 --rate 5000   192.168.100.135
nping --tcp -c 1000000 --rate 25000  192.168.100.135
```

### 8.2 Avec tcpreplay (PCAP)

```bash
# Pré‑requis : installer tcpreplay
sudo apt install -y tcpreplay

# Envoi à 10 000 pps depuis un PCAP
sudo tcpreplay --intf1=eno1 --pps=10000 traffic.pcap
```

---

<a name="9"></a>
\## 9. Collecte & extraction des métriques

### 9.1 Débit (pps)

* **Console** : fin d’exécution

  ```
  Processed 1000000 packets (12345 pkts/sec avg) in 120.00 seconds
  ```
* **`stats.log`** : chaque seconde, cherchez `decoder.pkts_delta` ou `capture.afpacket.poll_data`.

### 9.2 Latence (µs/paquet)

Si **perf\_monitor** activé, lisez :

* `/var/log/snort/perf_monitor_base.csv` → `perf_monitor.packets`
* `/var/log/snort/perf_monitor_cpu.csv`  → `thread_0.cpu_user` + `thread_0.cpu_system`

Ou directement :

* `/var/log/snort/perf_monitor_base.csv` → `latency.total_packets`, `latency.total_usecs`, `latency.max_usecs`

**Calcul latence moyenne** :

```bash
# Méthode latency.* directement
awk -F, 'NR==2 {
    p=$27; u=$28; print "Lat moy =", u/p, "µs/paquet; max =", $29, "µs";
}' /var/log/snort/perf_monitor_base.csv
```

Ou via CPU-time :

```bash
paste -d, <(tail -n +2 /var/log/snort/perf_monitor_base.csv | cut -d, -f27) \
          <(tail -n +2 /var/log/snort/perf_monitor_cpu.csv  | awk -F, '{print $2+$3}') \
| awk -F, '{ sum_p+=$1; sum_c+=$2 } END { print "Lat moy =", sum_c/sum_p, "µs/paquet" }'
```

---

<a name="10"></a>
\## 10. Scripts d’automatisation

### 10.1 Nettoyer anciens logs

```bash
sudo rm -f /var/log/snort/perf_monitor_*.csv
sudo rm -f /var/log/snort/stats.log /var/log/snort/eve.json
```

### 10.2 Script `extract_metrics.sh`

```bash
cat << 'EOF' > ~/extract_metrics.sh
#!/usr/bin/env bash
# Usage: ./extract_metrics.sh TEST_NAME
T=$1
OUTDIR=~/tests/$T
mkdir -p "$OUTDIR"

# 1) Extraire débit console
grep "Processed" ~/snort_console.log > "$OUTDIR/debit_console.txt"

# 2) Extraire stats.log pps
grep -E 'decoder.pkts_delta|poll_data' /var/log/snort/stats.log \
  > "$OUTDIR/debit_par_seconde.txt"

# 3) Extraire perf_monitor latency
awk -F, 'NR==2 { printf "lat_moy=%.2f max=%s\n", $28/$27, $29 }' \
    /var/log/snort/perf_monitor_base.csv \
    > "$OUTDIR/latence_latency.csv"

# 4) Extraire perf_monitor CPU-latence
paste -d, <(tail -n +2 /var/log/snort/perf_monitor_base.csv | cut -d, -f27) \
          <(tail -n +2 /var/log/snort/perf_monitor_cpu.csv  | awk -F, '{print $2+$3}') \
| awk -F, '{ sum_p+=$1; sum_c+=$2 } END { printf "lat_cpu=%.2fµs/paquet\n", sum_c/sum_p }' \
 > "$OUTDIR/latence_cpu_calc.txt"

echo "→ Métriques extraites dans $OUTDIR"
EOF
chmod +x ~/extract_metrics.sh
```

**Workflow** :

1. Démarrez Snort en redirigeant la console vers `~/snort_console.log`.
2. Lancez votre test nping/tcpreplay.
3. Stoppez Snort (Ctrl +C).
4. Exécutez `./extract_metrics.sh 1000pps` → récupère tout dans `~/tests/1000pps/`.

---

<a name="11"></a>
\## 11. Analyse & visualisation (optionnel)

* **Python/Matplotlib** pour tracer débit vs pps et latence vs pps.
* Utilisez le dossier `~/tests/` pour organiser vos résultats.

---

**1. Bash script d’extraction des métriques perf\_monitor**
Créez un fichier `extract_perf.sh` qui :

* Supprime les anciens CSV (`perf_monitor_*.csv`)
* Lance Snort (en arrière‑plan)
* Attend l’arrêt manuel (Ctrl‑C), puis extrait :

  * **perf\_monitor.packets** (colonne `perf_monitor.packets` dans `perf_monitor_base.csv`)
  * **CPU total** = `thread_0.cpu_user + thread_0.cpu_system` (dans `perf_monitor_cpu.csv`)
  * **latency.total\_packets**, **latency.total\_usecs**, **latency.max\_usecs** (dans `perf_monitor_base.csv`)
* Stocke le tout dans un TXT horodaté dans `~/results`

```bash
cat << 'EOF' > ~/extract_perf.sh
#!/usr/bin/env bash
set -euo pipefail

# Usage
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <LABEL> [SRC_DIR] <RUNTIME_SEC>"
  echo "  LABEL       : nom du test (ex: 1000pps)"
  echo "  SRC_DIR     : dossier CSV (défaut: pwd)"
  echo "  RUNTIME_SEC : durée Snort (en secondes)"
  exit 1
fi

LABEL="$1"
if [ $# -eq 2 ]; then
  SRC_DIR="."
  RUNTIME_SEC="$2"
else
  SRC_DIR="$2"
  RUNTIME_SEC="$3"
fi

OUT_DIR="$HOME/results/$LABEL"
mkdir -p "$OUT_DIR"

BASE_CSV="$SRC_DIR/perf_monitor_base.csv"
CPU_CSV="$SRC_DIR/perf_monitor_cpu.csv"
[ -f "$BASE_CSV" ] || { echo "[!] $BASE_CSV introuvable"; exit 1; }
[ -f "$CPU_CSV"  ] || { echo "[!] $CPU_CSV introuvable"; exit 1; }

cp "$BASE_CSV" "$OUT_DIR/"
cp "$CPU_CSV"  "$OUT_DIR/"

# Récupération des en-têtes pour trouver les index de colonnes
IFS=, read -r -a hdr_base < <(head -1 "$BASE_CSV")
IFS=, read -r -a hdr_cpu  < <(head -1 "$CPU_CSV")

col_idx() {
  local name=$1; shift
  local -n arr=$1
  for i in "${!arr[@]}"; do
    if [[ "${arr[i]}" == "$name" ]]; then
      echo $(( i + 1 ))
      return
    fi
  done
  echo "0"
}

I_PKT=$( col_idx perf_monitor.packets hdr_base )
I_LPKT=$(col_idx latency.total_packets hdr_base )
I_LUSE=$(col_idx latency.total_usecs hdr_base )
I_LMAX=$(col_idx latency.max_usecs hdr_base )
I_CU=$(  col_idx thread_0.cpu_user hdr_cpu  )
I_CS=$(  col_idx thread_0.cpu_system hdr_cpu)

# Aggrégation perf_monitor_base.csv
read sum_pkts sum_usec max_usec < <(
  awk -F, -v P="$I_PKT" -v U="$I_LUSE" -v M="$I_LMAX" '
    NR>1 && NF>U {
      s1 += $P;
      s2 += $U;
      if ($M > mx) mx = $M;
    }
    END { print s1+0, s2+0, mx+0 }
  ' "$BASE_CSV"
)

# Dernière ligne CPU
read cpu_user cpu_sys < <(
  tail -n1 "$CPU_CSV" | awk -F, -v CU="$I_CU" -v CS="$I_CS" '{ print $CU, $CS }'
)
cpu_total=$(( cpu_user + cpu_sys ))

# Calculs
if (( sum_pkts > 0 )); then
  lat_avg=$(awk "BEGIN{printf \"%.2f\", $sum_usec/$sum_pkts}")
else
  lat_avg=0
fi
if (( RUNTIME_SEC > 0 )); then
  debit_avg=$(awk "BEGIN{printf \"%.2f\", $sum_pkts/$RUNTIME_SEC}")
else
  debit_avg=0
fi

# Écriture metrics.txt
cat > "$OUT_DIR/metrics.txt" << METRICS
# Résultats test $LABEL
# Durée Snort : ${RUNTIME_SEC}s
# Lignes base CSV: $(($(wc -l < "$BASE_CSV")-1))

# Perf Monitor Base agrégé
perf_monitor.packets_total:    $sum_pkts
latency.total_usecs_sum:      $sum_usec
latency.max_usecs:            $max_usec
latency.avg_usecs_per_pkt:    $lat_avg

# Perf Monitor CPU (dernière ligne)
cpu.user_usec:                $cpu_user
cpu.sys_usec:                 $cpu_sys
cpu.total_usec:               $cpu_total

# Runtime & débit
runtime_sec:                  $RUNTIME_SEC
debit_avg_pps:                $debit_avg
METRICS

echo "[✔] Metrics générées dans $OUT_DIR/metrics.txt"
EOF

chmod +x ~/extract_perf.sh
cat << 'EOF' > ~/extract_perf.sh
#!/usr/bin/env bash
set -euo pipefail

# Usage
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <LABEL> [SRC_DIR] <RUNTIME_SEC>"
  echo "  LABEL       : nom du test (ex: 1000pps)"
  echo "  SRC_DIR     : dossier CSV (défaut: pwd)"
  echo "  RUNTIME_SEC : durée Snort (en secondes)"
  exit 1
fi

LABEL="$1"
if [ $# -eq 2 ]; then
  SRC_DIR="."
  RUNTIME_SEC="$2"
else
  SRC_DIR="$2"
  RUNTIME_SEC="$3"
fi

OUT_DIR="$HOME/results/$LABEL"
mkdir -p "$OUT_DIR"

BASE_CSV="$SRC_DIR/perf_monitor_base.csv"
CPU_CSV="$SRC_DIR/perf_monitor_cpu.csv"
[ -f "$BASE_CSV" ] || { echo "[!] $BASE_CSV introuvable"; exit 1; }
[ -f "$CPU_CSV"  ] || { echo "[!] $CPU_CSV introuvable"; exit 1; }

cp "$BASE_CSV" "$OUT_DIR/"
cp "$CPU_CSV"  "$OUT_DIR/"

# Récupération des en-têtes pour trouver les index de colonnes
IFS=, read -r -a hdr_base < <(head -1 "$BASE_CSV")
IFS=, read -r -a hdr_cpu  < <(head -1 "$CPU_CSV")

col_idx() {
  local name=$1; shift
  local -n arr=$1
  for i in "${!arr[@]}"; do
    if [[ "${arr[i]}" == "$name" ]]; then
      echo $(( i + 1 ))
      return
    fi
  done
  echo "0"
}

I_PKT=$( col_idx perf_monitor.packets hdr_base )
I_LPKT=$(col_idx latency.total_packets hdr_base )
I_LUSE=$(col_idx latency.total_usecs hdr_base )
I_LMAX=$(col_idx latency.max_usecs hdr_base )
I_CU=$(  col_idx thread_0.cpu_user hdr_cpu  )
I_CS=$(  col_idx thread_0.cpu_system hdr_cpu)

# Aggrégation perf_monitor_base.csv
read sum_pkts sum_usec max_usec < <(
  awk -F, -v P="$I_PKT" -v U="$I_LUSE" -v M="$I_LMAX" '
    NR>1 && NF>U {
      s1 += $P;
      s2 += $U;
      if ($M > mx) mx = $M;
    }
    END { print s1+0, s2+0, mx+0 }
  ' "$BASE_CSV"
)

# Dernière ligne CPU
read cpu_user cpu_sys < <(
  tail -n1 "$CPU_CSV" | awk -F, -v CU="$I_CU" -v CS="$I_CS" '{ print $CU, $CS }'
)
cpu_total=$(( cpu_user + cpu_sys ))

# Calculs
if (( sum_pkts > 0 )); then
  lat_avg=$(awk "BEGIN{printf \"%.2f\", $sum_usec/$sum_pkts}")
else
  lat_avg=0
fi
if (( RUNTIME_SEC > 0 )); then
  debit_avg=$(awk "BEGIN{printf \"%.2f\", $sum_pkts/$RUNTIME_SEC}")
else
  debit_avg=0
fi

# Écriture metrics.txt
cat > "$OUT_DIR/metrics.txt" << METRICS
# Résultats test $LABEL
# Durée Snort : ${RUNTIME_SEC}s
# Lignes base CSV: $(($(wc -l < "$BASE_CSV")-1))

# Perf Monitor Base agrégé
perf_monitor.packets_total:    $sum_pkts
latency.total_usecs_sum:      $sum_usec
latency.max_usecs:            $max_usec
latency.avg_usecs_per_pkt:    $lat_avg

# Perf Monitor CPU (dernière ligne)
cpu.user_usec:                $cpu_user
cpu.sys_usec:                 $cpu_sys
cpu.total_usec:               $cpu_total

# Runtime & débit
runtime_sec:                  $RUNTIME_SEC
debit_avg_pps:                $debit_avg
METRICS

echo "[✔] Metrics générées dans $OUT_DIR/metrics.txt"
EOF

chmod +x ~/extract_perf.sh

```

**Usage** (après avoir lancé votre `nping` à la même occasion) :

```bash
# par exemple pour 1000pps
~/extract_perf.sh 1000pps
# Appuyez CTRL‑C quand vous voulez stopper Snort
```

---

**2. Script Python de visualisation**
Supposons que vous ayez un répertoire `~/results/` contenant :

```
~/results/
  ├── 1000pps/
  │    └── metrics.txt
  ├── 5000pps/
  │    └── metrics.txt
  └── ...
```

```python
import os, glob
import matplotlib.pyplot as plt

# 1) Charger tous les metrics.txt
base = os.path.expanduser('~/results')
data = {}
for d in os.listdir(base):
    txt = os.path.join(base, d, 'metrics.txt')
    if not os.path.isfile(txt): continue
    with open(txt) as f:
        vals = {}
        for line in f:
            k,v = line.strip().split(':',1)
            vals[k] = float(v)
        data[d] = vals

# 2) Trier par rate
rates = sorted(data.keys(), key=lambda s: int(s.replace('pps','')))
debit = [ data[r]['calculated_debit_avg'] for r in rates ]
lat_avg = [ data[r]['calculated_latency_avg'] for r in rates ]
lat_max = [ data[r]['latency_max_usecs'] for r in rates ]

# 3) Plot débit moyen
plt.figure()
plt.plot(rates, debit, marker='o')
plt.title('Débit moyen (pps)')
plt.xlabel('Rate testé')
plt.ylabel('pps traités/s')
plt.savefig('debit_moyen.png')

# 4) Plot latence moyenne & max
plt.figure()
plt.plot(rates, lat_avg, marker='o', label='moyenne')
plt.plot(rates, lat_max, marker='x', label='max')
plt.title('Latence CPU (µs/paquet)')
plt.xlabel('Rate testé')
plt.ylabel('µs')
plt.legend()
plt.savefig('latence_cpu.png')
```

---

### Notes & anticipations d’erreurs

* **Colonnes dynamiques :** on recherche le numéro de la colonne par son nom (pas de `cut -f27`).
* **plusieurs lignes CSV** : on ne conserve que la **dernière** (`tail -n1`), qui représente la fin de votre exécution.
* **Snort non lancé** : `pkill snort` supprime tout résidu.
* **Absence de perf\_monitor** : le script échouera – vérifiez d’abord que `perf_monitor` est bien activé.

Avec ce duo (**extract\_perf.sh** + **plot.py**) vous :

1. Lancez Snort automatiquement, arrêtez-le par Ctrl‑C.
2. Récupérez immédiatement toutes vos métriques clés dans un seul fichier TXT.
3. Lancez le script Python pour obtenir vos graphes PNG : débit moyen et latence CPU.

Vous êtes prêt pour vos séries de tests (1 000→2 000→5 000→15 000→25 000 pps) !



----


🎉 **Vous avez maintenant un guide complet et robuste**, de l’installation de Snort 3 sur Debian 12 à la collecte automatisée de vos métriques de débit et latence pour vos expérimentations IDS/IPS ! N’hésitez pas à me demander toute précision.

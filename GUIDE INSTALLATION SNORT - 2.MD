# Guide complet Snort 3 sur Debian 12 + métriques débit & latence

Ce guide reprend **de zéro** la désinstallation, l’installation, la configuration et le lancement de Snort 3 en mode IDS (AF\_PACKET), et la collecte :

* du **débit** (pkts/s)
* de la **latence moyenne** (µs de CPU par paquet)

Il tient compte des erreurs rencontrées (manque de `<ctime>`, modules introuvables, etc.) et des commandes d’aide correctes.

---

## Prérequis

* **Système** : Debian 12
* **Interface IDS** : remplacez `eno1` partout par votre interface réseau
* **User** : vous travaillez en `$USER` (sudo disponible)

---

## 1. Désinstaller toute version Snort 2.x

```bash
sudo apt remove --purge snort snort-common snort-common-libraries snort-rules-default oinkmaster libluajit-5.1-2
sudo apt autoremove --purge
which snort   # doit ne rien renvoyer
```

---

## 2. Installer les dépendances de compilation

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config git wget \
  libpcap-dev libpcre3-dev libdumbnet-dev zlib1g-dev liblzma-dev \
  libhwloc-dev libsqlite3-dev uuid-dev libcmocka-dev \
  libluajit-5.1-dev libunwind-dev libssl-dev \
  libnetfilter-queue-dev libmnl-dev flex bison \
  libgoogle-perftools-dev    # pour TCMalloc
```

> Le paquet `libgoogle-perftools-dev` fournit TCMalloc nécessaire si vous utilisez `--enable-tcmalloc`.

---

## 3. Compiler & installer DAQ

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

## 4. Compiler & installer Snort 3 (avec patch `<ctime>`)

1. **Récupérer la source** et appliquer le patch manquant :

   ```bash
   cd ~/snort3-build
   wget https://github.com/snort3/snort3/archive/refs/tags/3.1.28.0.tar.gz
   tar xzf 3.1.28.0.tar.gz && cd snort3-3.1.28.0

   # Appliquer manuellement l'#include <ctime>
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

## 5. Préparer la configuration

1. **Créer les répertoires**

   ```bash
   sudo mkdir -p /usr/local/etc/snort /usr/local/etc/rules /var/log/snort
   sudo chown -R $USER /usr/local/etc/snort /usr/local/etc/rules /var/log/snort
   ```

2. **Télécharger les règles Snort 3 Community**

   ```bash
   wget -qO- https://www.snort.org/downloads/community/snort3-community-rules.tar.gz \
     | tar xz -C /usr/local/etc/rules
   ```

3. **Copier & éditer le fichier principal**

   ```bash
   cp /usr/local/etc/snort/snort.lua.example /usr/local/etc/snort/snort.lua
   nano /usr/local/etc/snort/snort.lua
   ```

   Dans **la partie 1. configure defaults**, mettez :

   ```lua
   HOME_NET     = '192.168.100.0/24'
   EXTERNAL_NET = 'any'
   ```

   Dans **la partie 5. configure detection**, assurez-vous d’inclure les variables :

   ```lua
   ips = {
     variables = default_variables
   }
   ```

4. **Configurer les sorties**
   Remplacez **intégralement** votre section `-- 7. configure outputs` par **ceci** :

   ```lua
   ---------------------------------------------------------------------------
   -- 7. configure outputs
   ---------------------------------------------------------------------------

   -- alertes « fast » (fichier /var/log/snort/alert_fast.txt)
   alert_fast = {
     file   = true,
     packet = false,
     limit  = 0
   }

   -- EVE JSON alerts (fichier /var/log/snort/eve.json)
   eve_log = {
     enabled   = true,
     filetype  = 'regular',
     filename  = 'eve.json',
     types     = { 'alert' }
   }

   -- stats textuelles chaque seconde (stats.log)
   stats = {
     enabled   = true,
     filename  = 'stats.log',
     interval  = 1,
     threads   = false,
     totals    = true
   }
   ```

5. **(Optionnel) Activer perf\_monitor**
   juste après la section outputs :

   ```lua
   ---------------------------------------------------------------------------
   -- perf_monitor : CPU & flux
   ---------------------------------------------------------------------------
   perf_monitor = {
     base     = true,
     cpu      = true,
     flow     = true,
     flow_ip  = true,
     format   = "csv",
     output   = "file",
     summary  = true,
     packets  = 10000,
     seconds  = 60
   }
   ```

---

## 6. Vérifier la config & lister les modules

* **Vérifier la conf** :

  ```bash
  /usr/local/bin/snort -c /usr/local/etc/snort/snort.lua --dump-config
  ```

  — si ça s’affiche sans erreur, tout est OK.

* **Lister les plugins output** (pour s’assurer que `stats`, `eve-log` et `alert_fast` sont reconnus) :

  ```bash
  /usr/local/bin/snort --help-output | grep -E 'stats|eve|alert_fast'
  ```

* **Vérifier `perf_monitor`** :

  ```bash
  /usr/local/bin/snort --help-inspector perf_monitor
  ```

---

## 7. Lancer Snort 3 en mode IDS (AF\_PACKET)

1. **Configurer l’interface**

   ```bash
   sudo ip link set dev eno1 promisc on
   sudo ethtool -K eno1 gro off lro off
   ```

2. **Démarrer Snort**

   ```bash
   sudo /usr/local/bin/snort \
     -c /usr/local/etc/snort/snort.lua \
     -i eno1 \
     --daq-mode=afpacket --daq-var interface=eno1 \
     -A alert_fast -q
   ```

   * `-A alert_fast` : on utilise le logger fast
   * `-q` : silencieux sur le reste

---

## 8. Générer du trafic avec nping (Windows)

```powershell
# ~2 min => 1 000 000 paquets
nping --tcp -c 1000000 --rate 1000   192.168.100.135
nping --tcp -c 1000000 --rate 5000   192.168.100.135
nping --tcp -c 1000000 --rate 25000  192.168.100.135
```

---

## 9. Collecte des métriques

### 9.1 Débit (paquets/s)

* **En console**, Snort affiche à la fin :

  ```
  Processed 1000000 packets (12345 pkts/sec avg) in 120.00 seconds
  ```
* **Dans** `/var/log/snort/stats.log` chaque seconde vous trouverez une ligne comme :

  ```
  Date: ... (uptime: ...) 
  capture.afpacket.poll_data                  | Thread-0 | 1234
  decoder.pkts_delta                          | Thread-0 | 1234
  ```

  Le champ `decoder.pkts_delta` est théoriquement le nombre de paquets traités par seconde.

### 9.2 Latence moyenne (µs/paquet)

Si **perf\_monitor** est activé, vous aurez :

* `/var/log/snort/perf_monitor_base.csv` → colonne `perf_monitor.packets`
* `/var/log/snort/perf_monitor_cpu.csv`  → colonne `perf_monitor.cpu_time_usec`

Calculez la latence moyenne :

```bash
paste -d, /var/log/snort/perf_monitor_base.csv /var/log/snort/perf_monitor_cpu.csv \
  | awk -F, 'NR>1 {
      dp = $2 - prev_p; dc = $3 - prev_c;
      sum_p += dp; sum_c += dc;
    }
    { prev_p=$2; prev_c=$3 }
    END { printf "Latence moyenne = %.2f µs/paquet\n", sum_c/sum_p }'
```

---

🎉 Vous disposez maintenant d’un **workflow Debian 12** complet, robuste, sans confusions Suricata/Snort, prêt à mesurer **débit** et **latence** de votre Snort 3 en mode IDS inline.



# LANCEMENT ET VERIFICATION :

Voici les ajustements à faire pour **Snort 3** (Snort++), qui a une syntaxe de ligne de commande un peu différente de Snort 2 :

---

## 1. Vérifier votre fichier de conf (`snort.lua`)

Pour tester votre configuration **sans capter de trafic**, utilisez l’option `-T` (Test) :

```bash
sudo snort \
  -c /usr/local/etc/snort/snort.lua \
  -T
```

* **`-T`** : vérifie la syntaxe de tous vos fichiers inclus (variables, modules, règles…) et signale les erreurs ou manques.
* Si tout est correct, vous verrez un message du genre
  `Configuration validated! Snort exiting`.

---

## 2. Choisir le mode de capture (DAQ)

Avec Snort 3, on précise le moteur de capture (libDAQ) et ses variables :

```bash
--daq afpacket                 # capture Linux high‑speed via AF_PACKET
--daq-var interface=wlp6s0     # votre interface
```

Vous pouvez évidemment remplacer `afpacket` par `pcap`, `netmap`, etc., selon ce qui est supporté.

---

## 3. Sélectionner le format d’alerte

Snort 3 propose plusieurs « alert modes » (logger plugins) :

| Mode        | Description                     |
| ----------- | ------------------------------- |
| alert\_fast | format « fast » style Snort 2   |
| alert\_full | dump complet (« full »)         |
| alert\_json | JSON (utile pour ingestion ELK) |

Pour afficher les alertes directement dans votre terminal, on utilise par exemple `alert_fast`.

---

## 4. Lancer Snort en mode IDS (détection seule)

Voici la commande complète :

```bash
sudo /usr/local/bin/snort \
  -c /usr/local/etc/snort/snort.lua \
  -i eno1 \
  --daq afpacket \
  --daq-var interface=eno1 \
  -A alert_fast 

```

* **`-c /…/snort.lua`** : votre fichier principal
* **`-i wlp6s0`** : interface à surveiller
* **`--daq afpacket`** et **`--daq-var interface=wlp6s0`** : capture libDAQ AF\_PACKET
* **`-A alert_fast`** : imprime les alertes en « fast » format
* **`-q`** : mode silencieux pour le reste des logs (quiet)

Vous devriez voir des lignes du type :

```
07/15–16:00:00.000000 [**] [1:1000001:0] TEST-RULE [**] {TCP} 192.168.100.20:xxxxx -> 192.168.100.135:80
```

---

## 5. En mode IPS (inline)

Si vous voulez tester en **IPS** (bloquer les paquets) au lieu de IDS :

1. **Activez** la partie `inline.lua` dans votre `snort.lua` (ou bien chargez-le ensuite) :

   ```lua
   include 'inline.lua'
   ```

2. Lancez avec **`-Q`** (inline queue) :

   ```bash
   sudo snort \
     -c /usr/local/etc/snort/snort.lua \
     --daq afpacket \
     --daq-var interface=wlp6s0 \
     -i wlp6s0 \
     -A alert_fast \
     -Q
   ```

   ```bash
   sudo /usr/local/bin/snort \
  -c /usr/local/etc/snort/snort.lua \
  -i eno1 \
  --daq afpacket \
  --daq-var interface=eno1 \
  -A alert_fast \
  -l /var/log/snort 
   ```

   * **`-Q`** : active le mode inline (IPS). Sur Linux, vous devrez avoir configuré vos chaînes iptables/NFQUEUE pour rediriger les paquets vers Snort.

---

### Résumé rapide

1. **Tester** la config :

   ```bash
   sudo snort -c /usr/local/etc/snort/snort.lua -T
   ```
2. **IDS** (alertes seulement) :

   ```bash
   sudo snort -c /usr/local/etc/snort/snort.lua -i wlp6s0 \
     --daq afpacket --daq-var interface=wlp6s0 \
     -A alert_fast -q
   ```
3. **IPS** (bloquer inline) :

   ```bash
   sudo snort -c /usr/local/etc/snort/snort.lua \
     --daq afpacket --daq-var interface=wlp6s0 \
     -i wlp6s0 -A alert_fast -Q
   ```

Avec ce guide, vous bénéficiez de la syntaxe correcte pour Snort 3, vous savez comment **vérifier** votre configuration, choisir votre **DAQ**, et lancer Snort en **IDS** ou **IPS**.


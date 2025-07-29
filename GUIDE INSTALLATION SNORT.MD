Voici un guide détaillé, étape par étape, pour :



1\. \*\*Désinstaller\*\* proprement la version Snort 2.x installée via `apt`.

2\. \*\*Installer\*\* Snort 3 (Snort++) depuis les sources sur Debian/Ubuntu.

3\. \*\*Configurer\*\* Snort 3 en mode IDS inline (AF\\\_PACKET).



Les instructions sont issues de la documentation officielle et de plusieurs tutoriels de référence pour garantir leur fiabilité.



---



\## 1. Désinstaller la version actuelle (apt)



```bash

\# 1.1 Désinstaller Snort et ses dépendances

sudo apt remove --purge snort snort-common snort-common-libraries snort-rules-default oinkmaster libluajit-5.1-2

\# 1.2 Nettoyer les paquets orphelins

sudo apt autoremove --purge

```



> Vérifiez qu’aucun binaire `snort` n’est plus présent :

>

> ```bash

> which snort

> # ne doit rien renvoyer

> ```



---



\## 2. Installer les prérequis de compilation



Snort 3 utilise CMake et un certain nombre de bibliothèques. Installez-les :



```bash

sudo apt update

sudo apt install -y \\

&nbsp; build-essential cmake pkg-config libpcap-dev libpcre3-dev \\

&nbsp; libdnet-dev libdumbnet-dev zlib1g-dev liblzma-dev liblzma-dev \\

&nbsp; libhwloc-dev libsqlite3-dev uuid-dev libcmocka-dev \\

&nbsp; luajit libluajit-5.1-dev libunwind-dev libssl-dev \\

&nbsp; libnetfilter-queue-dev libmnl-dev flex bison git wget

```



---



\## 3. Compiler et installer DAQ (Data Acquisition Library)



Snort 3 s’appuie sur DAQ. Il faut le compiler :



```bash

mkdir ~/snort3-build \&\& cd ~/snort3-build

git clone https://github.com/snort3/libdaq.git

cd libdaq

./bootstrap

./configure --prefix=/usr/local

make -j$(nproc)

sudo make install

```



> Cette étape installe DAQ version 3.x, nécessaire pour l’acquisition AF\\\_PACKET (\[kifarunix.com]\[1]).



---



\## 4. Télécharger, compiler et installer Snort 3



1\. Récupérez la dernière version de Snort 3 (ici 3.1.28.0) :



&nbsp;  ```bash

&nbsp;  cd ~/snort3-build

&nbsp;  wget https://github.com/snort3/snort3/archive/refs/tags/3.1.28.0.tar.gz

&nbsp;  tar xzf 3.1.28.0.tar.gz

&nbsp;  cd snort3-3.1.28.0

&nbsp;  ```



2\. Configurez la compilation avec CMake :



&nbsp;  ```bash

&nbsp;  ./configure\_cmake.sh --prefix=/usr/local --enable-tcmalloc

&nbsp;  ```



3\. Compilez et installez :



&nbsp;  ```bash

&nbsp;  cd build

&nbsp;  make -j$(nproc)

&nbsp;  sudo make install

&nbsp;  sudo ldconfig

&nbsp;  ```



4\. Vérifiez l’installation :



&nbsp;  ```bash

&nbsp;  snort -V

&nbsp;  # Doit afficher Snort++ Version 3.1.28.0

&nbsp;  ```



---



\## 5. Configuration de base de Snort 3



1\. \*\*Mettre l’interface en promiscuité\*\* (pour voir tout le trafic) :



&nbsp;  ```bash

&nbsp;  sudo ip link set dev eno1 promisc on

&nbsp;  ```



2\. \*\*Désactiver l’offloading\*\* pour éviter la troncature des paquets :



&nbsp;  ```bash

&nbsp;  sudo ethtool -K eno1 gro off lro off

&nbsp;  ```



3\. \*\*Créer les répertoires de config\*\* :



&nbsp;  ```bash

&nbsp;  sudo mkdir -p /usr/local/etc/snort

&nbsp;  sudo mkdir -p /usr/local/etc/rules

&nbsp;  ```



4\. \*\*Récupérer les règles communautaires\*\* :



&nbsp;  ```bash

&nbsp;  sudo wget -qO- \\

&nbsp;    https://www.snort.org/downloads/community/snort3-community-rules.tar.gz \\

&nbsp;    | sudo tar xz -C /usr/local/etc/rules/

&nbsp;  ```



5\. \*\*Copier le fichier de configuration exemple\*\* et l’éditer :



&nbsp;  ```bash

&nbsp;  sudo cp /usr/local/etc/snort/snort\_defaults.lua.example /usr/local/etc/snort/snort\_defaults.lua

&nbsp;  sudo vim /usr/local/etc/snort/snort.lua

&nbsp;  ```



&nbsp;  Dans `snort.lua` :



&nbsp;  ```lua

&nbsp;  HOME\_NET = '192.168.100.0/24'

&nbsp;  EXTERNAL\_NET = 'any'

&nbsp;  RULE\_PATH = '/usr/local/etc/rules/snort3-community.rules'

&nbsp;  ```



---



\## 6. Lancer Snort 3 en mode IDS



```bash

sudo snort -c /usr/local/etc/snort/snort.lua -i eno1 \\

&nbsp; --daq-mode=afpacket --daq-var interface=eno1 \\

&nbsp; -Q    # IDS mode inline (observe)

```



\* `-c` : chemin du fichier de conf

\* `-i eno1` : interface à surveiller

\* `--daq-mode=afpacket` : utilise AF\\\_PACKET sous Linux

\* `-Q` : mode inline/IDS (pas de verdict ACCEPT/BLOCK)



---



\## 7. Vérification



\* Vous devez voir au démarrage :



&nbsp; ```

&nbsp; --== Initializing Snort ==

&nbsp; DAQ Modules:  afpacket; ...

&nbsp; Initialization Complete

&nbsp; ```

\* Envoi quelques paquets de test (nping ou ping) et vérifiez dans le log d’alertes :



&nbsp; ```bash

&nbsp; sudo tail -f /var/log/snort/alert

&nbsp; ```



---



> \*\*Sources\*\* :

> – « Install and Configure Snort 3 on Ubuntu 22.04 », Kifarunix (\[kifarunix.com]\[1])

> – Documentation officielle Snort 3 (\[kifarunix.com]\[1])



\[1]: https://kifarunix.com/install-and-configure-snort-3-on-ubuntu-22-04/ "Install and Configure Snort 3 on Ubuntu 22.04 - kifarunix.com"





Le message d’erreur indique que CMake ne trouve pas TCMalloc (la bibliothèque de Google Perftools) car vous avez activé `--enable-tcmalloc` sans avoir installé le paquet de développement nécessaire. Vous avez deux options :



---



\## Option 1 : Installer TCMalloc



1\. \*\*Installer le paquet de développement TCMalloc\*\*

&nbsp;  Sur Debian/Ubuntu, le paquet s’appelle généralement `libtcmalloc-minimal4` ou `libgoogle-perftools-dev` :



&nbsp;  ```bash

&nbsp;  sudo apt update

&nbsp;  sudo apt install -y libgoogle-perftools-dev

&nbsp;  ```



2\. \*\*Relancer la configuration CMake\*\*

&nbsp;  Depuis votre répertoire de build :



&nbsp;  ```bash

&nbsp;  cd ~/snort3-build/snort3-3.1.28.0

&nbsp;  rm -rf build

&nbsp;  ./configure\_cmake.sh --prefix=/usr/local --enable-tcmalloc

&nbsp;  cd build

&nbsp;  make -j$(nproc)

&nbsp;  sudo make install

&nbsp;  sudo ldconfig

&nbsp;  ```



---



\## Option 2 : Désactiver TCMalloc



Si vous ne souhaitez pas utiliser TCMalloc et accepter la libération de mémoire standard, supprimez le flag `--enable-tcmalloc` :



```bash

cd ~/snort3-build/snort3-3.1.28.0

rm -rf build

./configure\_cmake.sh --prefix=/usr/local

cd build

make -j$(nproc)

sudo make install

sudo ldconfig

```



---



Après cela, `snort -V` doit afficher la version installée sans erreur. Vous pourrez poursuivre la configuration et le lancement de Snort 3 en mode IDS comme expliqué précédemment.





À partir de l’instant où Snort 3 est compilé et installé, voici comment prendre le fichier de configuration d’exemple, le personnaliser et lancer Snort en mode IDS sur votre interface :



---



\## 1. Copier le fichier de configuration d’exemple



Les fichiers d’exemple sont fournis dans `/usr/local/etc/snort/`.

Nous allons renommer le `snort.lua` principal en `snort.lua.orig` et reprendre l’exemple :



```bash

cd /usr/local/etc/snort

\# Sauvegarde de l'actuel snort.lua

sudo cp snort.lua snort.lua.orig

\# Copier l’exemple de snort\_defaults.lua (si déjà installé)

\# Note : dans certains paquets, il s’appelle snort\_defaults.lua, il est déjà en place.

\# À présent, créez une copie du fichier de configuration de base :

sudo cp snort\_defaults.lua snort.lua

```



> Si vous avez un autre fichier `.lua` d’exemple (par ex. Snort-3.1.28.0 fournit un `snort.lua` exemple dans son repo), adaptez le chemin :

>

> ```bash

> sudo cp ~/snort3-3.1.28.0/etc/snort.lua.example /usr/local/etc/snort/snort.lua

> ```



---



\## 2. Éditer `snort.lua`



Ouvrez-le avec un éditeur en console (ici `nano`, installé de base) :



```bash

sudo nano /usr/local/etc/snort/snort.lua

```



Repérez et ajustez :



1\. \*\*Interface réseau\*\*

&nbsp;  Trouvez la clé `HOME\_NET` et `EXTERNAL\_NET`, par exemple :



&nbsp;  ```lua

&nbsp;  ips =

&nbsp;  {

&nbsp;      -- Sur quel(s) interface(s) Snort doit écouter :

&nbsp;      interface = { "wlp6s0" },



&nbsp;      -- Réseau protégé (HOME\_NET) :

&nbsp;      HOME\_NET = { "192.168.100.0/24" },



&nbsp;      -- Tout le reste est considéré comme `EXTERNAL\_NET` :

&nbsp;      EXTERNAL\_NET = { "any" },

&nbsp;  }

&nbsp;  ```



&nbsp;  Adaptez `"wlp6s0"` à votre interface, et `192.168.100.0/24` au réseau de votre IDS.



2\. \*\*Chemin vers les règles\*\*

&nbsp;  Assurez-vous que la variable `RULE\_PATH` pointe vers `/usr/local/etc/rules/snort3-community-rules` :



&nbsp;  ```lua

&nbsp;  -- Répertoire contenant vos .rules et sid-msg.map

&nbsp;  RULE\_PATH = '/usr/local/etc/rules/snort3-community-rules/'

&nbsp;  ```



3\. \*\*Activer le mode IDS\*\*

&nbsp;  Dans la section `runmode`, la valeur par défaut est déjà `‘autofp’` ou `‘inline’` — utilisez `‘autofp’` pour IDS :



&nbsp;  ```lua

&nbsp;  runmode = 'autofp',  -- IDS simple (pas d’inline)

&nbsp;  ```



4\. \*\*Logging minimal\*\*

&nbsp;  Pour ne générer que les alertes JSON, vérifiez la section `output` :



&nbsp;  ```lua

&nbsp;  output =

&nbsp;  {

&nbsp;      -- EVE JSON alerts seulement

&nbsp;      {

&nbsp;          module = 'eve-log',

&nbsp;          filetype = 'regular',

&nbsp;          filename = 'eve.json',

&nbsp;          types = { 'alert' },

&nbsp;      },

&nbsp;      -- Statistiques (facultatif, en plus de stats.log)

&nbsp;      {

&nbsp;          module = 'stats',

&nbsp;          filename = 'stats.log',

&nbsp;          interval = 1,       -- en secondes

&nbsp;          threads = false,

&nbsp;          totals = true,

&nbsp;      },

&nbsp;  }

&nbsp;  ```



5\. \*\*Sauvegarde\*\* et quittez (`Ctrl+O`, `Entrée`, `Ctrl+X`).



---



\## 3. Tester la configuration



Avant de lancer en live, testez la syntaxe :



```bash

sudo snort -c /usr/local/etc/snort/snort.lua --verify-config

```



Vous devez voir :



```

Configuration validated! Snort exiting

```



---



\## 4. Lancer Snort en IDS (pas daemon)



Pour démarrer Snort en avant-plan et observer les logs :



```bash

sudo snort -c /usr/local/etc/snort/snort.lua -i wlp6s0 -A console -q

```



\* `-i wlp6s0` : votre interface

\* `-A console` : affiche les alertes à l’écran

\* `-q` : réduit le bruit en console



> \*\*Pour lancer en arrière-plan\*\* (mode daemon), ajoutez `--daemon` et `--pidfile /var/run/snort.pid`.



---



\## 5. Vérifier que ça tourne



Dans un autre terminal :



```bash

ps aux | grep snort

\# ou

sudo netstat -tunlp | grep snort

```



Vous devez voir Snort à l’écoute sur votre interface.



---



\## 6. Générer du trafic et collecter les alertes



1\. \*\*Depuis Windows\*\* :



&nbsp;  ```powershell

&nbsp;  nping --tcp -c 100000 --rate 1000 192.168.100.135

&nbsp;  ```



2\. \*\*Sur Ubuntu\*\* : Taillez le fichier eve.json pour voir les alertes :



&nbsp;  ```bash

&nbsp;  sudo tail -f /var/log/suricata/eve.json   # si vous avez gardé le chemin par défaut

&nbsp;  # OU

&nbsp;  sudo tail -f /usr/local/var/log/snort/eve.json

&nbsp;  ```



&nbsp;  Vous verrez un JSON par alerte :



&nbsp;  ```json

&nbsp;  {

&nbsp;    "timestamp": "2025-07-15T20:00:01.123456+0200",

&nbsp;    "event\_type": "alert",

&nbsp;    "src\_ip": "192.168.100.20",

&nbsp;    "dest\_ip": "192.168.100.135",

&nbsp;    "alert": { "signature\_id": 2100498, "rev": 3, "signature": "ET SCAN Potential SSH Scan", … },

&nbsp;    …

&nbsp;  }

&nbsp;  ```



---



\## 7. Automatiser l’extraction du nombre d’alertes par seconde



Créez `extract\_alerts.sh` :



```bash

cat << 'EOF' > ~/extract\_alerts.sh

\#!/usr/bin/env bash

\# Usage: ./extract\_alerts.sh DURATION

DURATION=$1

OUT=~/tests/alerts\_per\_sec\_${DURATION}.txt



\# Lire eve.json, compter les alertes par seconde

jq -r --argjson dur "$DURATION" '

&nbsp; select(.event\_type=="alert") |

&nbsp; .timestamp\[11:19] as $ts |

&nbsp; $ts

' /usr/local/var/log/snort/eve.json \\

| uniq -c \\

| awk '{ print $2 "," $1 }' \\

> "$OUT"



echo "→ Alertes/s extraites dans $OUT"

EOF



chmod +x ~/extract\_alerts.sh

```



Puis, après avoir fait tourner Snort et nping pendant, par ex., `120` secondes :



```bash

~/extract\_alerts.sh 120

```



Vous obtiendrez un fichier CSV `alerts\_per\_sec\_120.txt` contenant :



```

HH:MM:SS,nb\_alertes

...

```



---



\## 8. (Option) Analyse Python



Vous pouvez réutiliser votre script Python `analyze\_debit.py`, en changeant simplement la colonne lue pour la deuxième valeur (`nb\_alertes`) et en remplaçant `cum\_pkts` par `alerts`. Le reste du workflow (évolution, histogramme, moyenne) reste identique.



---



Voilà, vous disposez maintenant d’un guide \*\*pas à pas\*\*, de la désinstallation/installation de Snort 3 jusqu’à la collecte et l’analyse du \*\*nombre d’alertes par seconde\*\* en mode IDS.




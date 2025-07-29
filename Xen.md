Puisque vous devez travailler sur votre propre Xen modifié, voici **la procédure complète pour compiler et installer Xen 4.18 manuellement** et ajouter vous‑même l’entrée dans GRUB, sans passer par les paquets Debian :

---

## 1. Installer les dépendances de compilation

```bash
sudo apt update
sudo apt install -y build-essential libncurses-dev libssl-dev libxml2-dev \
                   libyajl-dev asciidoc xsltproc python3-dev uuid-dev \
                   zlib1g-dev git
```

---

## 2. Compiler Xen depuis vos sources

1. Allez dans le répertoire source :

   ```bash
   cd ~/xen-4.18.0
   ```
2. Configurez la compilation (installe en `/usr`, config dans `/etc` etc.) :

   ```bash
   ./configure --prefix=/usr \
               --sysconfdir=/etc \
               --localstatedir=/var \
               --libexecdir=/usr/lib/xen \
               --with-python=/usr/bin/python3
   ```
3. Lancez la compilation (remplacez `4` par le nombre de cœurs à utiliser) :

   ```bash
   make -j4
   ```
4. (Optionnel) Exécutez les tests si vous voulez valider votre build :

   ```bash
   make check
   ```

---

## 3. Installer Xen et ses outils

1. Installez l’hyperviseur et les outils de gestion :

   ```bash
   sudo make install            # installe /usr/lib/xen-4.18/boot/xen.gz etc.
   sudo make install-tools      # installe xl, xm, xenstore-ls…
   ```
2. Vérifiez que le binaire hyperviseur est bien en place :

   ```bash
   ls -l /usr/lib/xen-4.18/boot/xen.gz
   ```
3. Pour plus de commodité, créez un lien :

   ```bash
   sudo ln -sf /usr/lib/xen-4.18/boot/xen.gz /boot/xen-4.18-amd64.gz
   ```

---

## 4. Créer manuellement une entrée GRUB pour votre Xen

Plutôt que de bricoler dans `/etc/grub.d/20_linux_xen`, on va ajouter une entrée **custom** via le script `/etc/grub.d/40_custom` :

1. **Récupérez l’UUID** de votre partition racine (nécessaire pour le paramètre `root=`) :

   ```bash
   blkid -s UUID -o value $(findmnt -n -o SOURCE /)
   ```

   Disons que l’UUID retournée est `37897315-f33f-4853-87b4-60c4acffc613`.

2. **Éditez** le fichier `/etc/grub.d/40_custom` :

   ```bash
   sudo nano /etc/grub.d/40_custom
   ```

   Collez-y, en adaptant l’UUID et le nom du kernel si besoin :

   ```bash
   #! /bin/sh
   exec tail -n +3 "$0"
   menuentry 'Debian, with Custom Xen 4.18' --class xen {
       multiboot   /boot/xen-4.18-amd64.gz placeholder
       module      /boot/vmlinuz-6.1.0-26-amd64 root=UUID=37897315-f33f-4853-87b4-60c4acffc613 ro quiet
       module      /boot/initrd.img-6.1.0-26-amd64
   }
   ```

3. **Rendez-le exécutable** :

   ```bash
   sudo chmod +x /etc/grub.d/40_custom
   ```

---

## 5. Mettre à jour GRUB et redémarrer

```bash
sudo update-grub
sudo reboot
```

Au démarrage, vous devriez voir une entrée **“Debian, with Custom Xen 4.18”** en plus des Linux classiques. Sélectionnez‑la (ou modifiez `GRUB_DEFAULT` dans `/etc/default/grub` pour en faire l’entrée par défaut) :

```bash
# /etc/default/grub
GRUB_DEFAULT="Debian, with Custom Xen 4.18"
sudo update-grub
```

---

## 6. Vérifier que Xen tourne

Une fois la machine redémarrée et que vous avez booté sur votre entrée Xen :

```bash
which xl
xl --version
xl info | grep xen_version
```

Vous y verrez bien la ligne `xen_version : 4.18`, et le binaire `xl` disponible.

---

**Voilà la procédure manuelle** :

* vous maîtrisez la compilation et l’installation,
* vous gérez vous‑même la ligne GRUB,
* vous êtes prêts à modifier le code Xen puis à recompiler et tester sans dépendre des paquets Debian.

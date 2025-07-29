@echo off
title Connexion SSH vers la machine Xen
REM Vérifier qu'AnyConnect est déjà connecté (tu devrais voir "Connected" dans l'UI)
REM Lance la connexion SSH
ssh -i "%USERPROFILE%\.ssh\id_rsa_widesys2" admin@131.254.23.28

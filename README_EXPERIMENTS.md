# VCM-EnergyMap — Base Layer avec bitstream réel et profilage

Ces fichiers **s'ajoutent** au dépôt `hadipardis/svc` (aucun fichier d'origine n'est modifié).
Couche d'amélioration (Enhancement Layer) : non utilisée.

## 1. Fichiers

| Fichier | Rôle |
|---|---|
| `entropy_codec.py` | Codage entropique réel (range coder `constriction`) : prior factorisé (z), gaussienne conditionnelle (y des trames P), contexte autorégressif + mélange de gaussiennes (y des trames I). Réutilise les modèles de probabilité du dépôt. |
| `bitstream.py` | Format de fichier `.vcmb` (en-tête de séquence + enregistrements par trame, champs préfixés par leur longueur). |
| `base_codec.py` | `BaseLayerCodec("enc"/"dec")` : encodeur et décodeur indépendants. Le décodeur n'a pas de réseau d'estimation de mouvement. |
| `profiling.py` | Chronométrage par étape (`time.perf_counter`), phases init/warmup/measured, séparation I/P, CSV + JSON. |
| `energy_profiling.py` | Mesure énergétique par blocs (CodeCarbon 3.3.2), baseline à vide, répétitions. |
| `test_bitstream.py` | Tests de validation (18 contrôles). |
| `test_base_profiled.py` | Pipeline : `encode`, `decode`, `evaluate`, `roundtrip`, `energy`. |

## 2. Dépendances

```
pip install constriction        # wheel Windows/Linux/macOS précompilé, aucun compilateur requis
```
`torchac` (PyPI) n'est **pas** utilisé : son API (`encode_float_cdf`…) diffère de celle attendue par le dépôt
(`ac.range_index_encode`, `pmf2cdf`, `flush`…) et il compile du C++ à l'installation.

## 3. Pourquoi les `compress()/decompress()` d'origine ne sont pas réutilisés

- `ac` n'est pas importé (`entropy_models.py` l.8) ; le module compilé n'est pas dans le dépôt.
- `MixtureModelConditional.compress/decompress` lèvent `NotImplementedError` (le codeur I utilise un mélange de gaussiennes).
- `ContextModel.get_cdf` convertit des CDF 16 bits en `int16` (`.short()`), ce qui peut déborder pour la valeur 65536 (CDF cumulée à 2^16).
- `ContextModel.compress` stocke `minmax` sur un octet (débordement possible au-delà de 255).
- `BitStreamIO` sépare les champs par une séquence magique de 4 octets pouvant apparaître dans la charge utile.

Les modules réseau (`encode`, `hyper_analysis`, `hyper_synthesis`, `PA`, `pred_prior`, `decode`, `DQ`, `MWNet`, `MCNet`…) sont **réutilisés tels quels** : seul le backend entropique est nouveau.

### Écart documenté par rapport au schéma de déploiement d'origine
Les tables de probabilité sont évaluées directement à partir de (moyenne, échelle[, π]) sur l'alphabet entier
`[lo, hi]` réellement présent dans le tenseur (envoyé en en-tête), au lieu d'une table de 64 niveaux d'échelle avec symbole d'échappement.
Cela ne modifie ni les poids ni la quantification (`symbole = round(y − moyenne)`). Conséquence : sur des modèles
mal entraînés (poids aléatoires), le débit réel peut être *inférieur* à l'estimation ; avec un modèle entraîné
l'écart doit être faible (à vérifier — voir §6).

## 4. Définitions du débit

| Métrique | Définition | Nature |
|---|---|---|
| `bpp_est` | −Σ log₂ P(symboles) avec la formule de `estimate_bpp` (borne 1e-9) | théorique |
| `bpp_payload` | 8 × octets du flux du range coder / (H·W) | mesuré |
| `bpp_frame` | 8 × (payload + en-tête de trame : longueurs + alphabets) / (H·W) | mesuré |
| `bpp_file` | 8 × taille du fichier (en-tête de séquence inclus) / (nb_trames·H·W) | mesuré |

H·W = taille originale (hors padding), comme `estimate_bpp(input=coding_frame)` dans le script d'origine.
Le flux du range coder est aligné sur 32 bits (≤ 3 octets de remplissage par trame).
Recommandation : rapporter `bpp_frame` (ou `bpp_file`) comme débit réel et `bpp_payload` comme borne basse.

## 5. Limites et comportements hérités à connaître

1. **Décodage séquentiel obligatoire.** Comme dans `test_base.py`, le tampon de flots de `MWNet` n'est pas
   réinitialisé aux trames I : un GOP n'est pas décodable seul. `--closed_gop` le réinitialise (écart volontaire, désactivé par défaut).
2. **Déterminisme flottant.** L'encodeur et le décodeur doivent exécuter les mêmes opérations sur la même
   machine, avec la même version de torch et le même nombre de threads. Le décodage inter-machines n'est pas garanti.
3. **Coût du décodage des trames I** : le contexte autorégressif impose un décodage position par position
   (une évaluation de réseau par position spatiale du latent).
4. **Reconstruction de référence dans l'encodeur** : l'encodeur doit reconstruire chaque trame pour prédire la suivante ;
   les étapes `*_synthesis` de l'encodeur sont donc du calcul de type décodeur, mesuré dans le temps d'encodage.
5. **Nommage des trames** : les scripts lisent `<prefix>%03d.png` à partir de `--start_index` (défaut 0).
   Avec ffmpeg : `ffmpeg -i video.yuv ... -start_number 0 prefix_%03d.png`
   (ffmpeg commence à 1 par défaut).

## 6. Procédure de validation (à exécuter avant toute expérience)

```powershell
# 1) Validation mécanique + équivalence avec la chaîne d'origine, AVEC le vrai checkpoint
python test_bitstream.py --checkpoint checkpoints\base\checkpoint_base_1.pth.tar `
       --frames_dir input\ParkScene --prefix ParkScene_ --no_frames 4 --gop 3
# attendu : "N/N checks passed", code de sortie 0
```
Contrôles : latents décodés = latents encodés (tous les sous-flux), reconstruction du décodeur bit-exacte,
écart vs `forward()` d'origine < `--tol_rec`, bpp estimé = `estimate_bpp` d'origine, flux corrompu rejeté/différent,
conteneur (varints, en-tête, troncature, charge utile contenant l'ancien séparateur magique).

Les tests T3–T5 ont été exécutés avec **poids aléatoires** sur une séquence synthétique (18/18 réussis, décodeur dans un processus séparé).
Ils prouvent la mécanique du bitstream, **pas** la qualité de compression, ni la bit-exactitude avec vos checkpoints
sur votre machine : relancer la commande ci-dessus.

## 7. Exécution

```powershell
# Encodage + décodage dans deux processus séparés + évaluation
python test_base_profiled.py roundtrip --checkpoint checkpoints\base\checkpoint_base_1.pth.tar --checkpoint_number 1 `
    --bitstream results\ParkScene_q1.vcmb --inp_path input\ParkScene --prefix ParkScene_ --out_path out\dec\ParkScene_q1 `
    --no_frames 32 --gop 32 --threads 4 --warmup_frames 2 --results results

# Séparément
python test_base_profiled.py encode   ...   # produit le .vcmb, <stem>_enc_frames.csv, <stem>_enc_profile.{json,csv}
python test_base_profiled.py decode   ...   # ne lit QUE le .vcmb ; produit PNG + <stem>_dec_profile.*
python test_base_profiled.py evaluate ...   # PSNR / MS-SSIM / BPP vs originaux

# Énergie (blocs entiers, en mémoire, sans E/S disque)
python test_base_profiled.py energy --checkpoint ... --bitstream results\x.vcmb --inp_path ... --prefix ... `
    --no_frames 8 --gop 8 --threads 4 --idle_seconds 60 --min_seconds 30 --repeats 5 --also_iframe
```

Sorties (`results/`) : `*_enc_frames.csv` (une ligne par trame : octets, bpp, bits par sous-flux), `*_eval_frames.csv`,
`*_eval_summary.json`, `*_{enc,dec}_profile.json` (+ `_raw.csv`, `_summary.csv`), `*_energy.json`.

## 8. Profilage : frontières de mesure

- Les étapes sont des **feuilles** (aucune n'en contient une autre) ; `frame_summary.coverage` = Σ étapes / temps mur
  (≈ 1 attendu ; vérifié à 98–100 % dans le test de fumée).
- Catégories : `compute` (réseaux, évaluation des PMF, range coder) / `io` (lecture/écriture d'images, lecture/écriture de flux).
  La sauvegarde PNG est mesurée à part et exclue du temps de calcul.
- Phases : `init` (construction + chargement du checkpoint, `init_times_s`), `warmup` (`--warmup_frames`, exclu des statistiques), `measured`.
- Sur CPU, PyTorch est synchrone : aucune synchronisation nécessaire ; `Profiler(device="cuda")` appelle `torch.cuda.synchronize()`.
- Écriture du fichier : le `.vcmb` est assemblé en mémoire et écrit à la fin (`bitstream_file_write_total` dans `init_times_s`) ;
  `bitstream_write` par trame = sérialisation de l'enregistrement seulement.
- Étapes : `M_*` = codage du mouvement (flot), `R_*` = résiduel, `I_*` = trame I ; suffixes `_analysis`, `_hyper_analysis`,
  `_z_quantize/_pmf/_rangecode|rangedecode`, `_cond_net` (hyper-synthèse + a priori temporel), `_y_*`, `_synthesis`
  (reconstruction), `I_y_context_net` (réseau de contexte, indissociable du décodage entropique des trames I).
  `motion_estimation` = PWC-Net (encodeur seul), `motion_prediction` = MWNet, `motion_compensation` = warping + MCNet.
  Le « codage entropique » = `*_pmf` + `*_rangecode` ; la « quantification » = `*_quantize`.

## 9. Énergie : ce qui est et n'est pas mesuré

- CodeCarbon **ne lit pas de wattmètre**. `energy_kind` dans le JSON vaut `measured_rapl` (compteurs RAPL) ou
  `modelled_tdp_x_load` (puissance estimée = TDP × charge CPU + modèle de RAM, cas habituel sous Windows). Une valeur modélisée est un **proxy**, jamais une mesure.
- Seuls des **blocs entiers** sont mesurés (séquence encodée, séquence décodée, trame I seule), répétés ≥ `--min_seconds`
  (CodeCarbon échantillonne toutes les `--measure_power_secs` ; viser ≥ 10× cet intervalle). **Aucune répartition de l'énergie entre modules n'est faite.**
  L'énergie d'une trame P peut être *estimée par différence* entre blocs (séquence − trame I) avec propagation des incertitudes ; à étiqueter comme dérivée.
- Baseline à vide mesurée (`--idle_seconds`) ; les valeurs `net_*` retirent `P_idle × durée`. Les valeurs brutes (`gross_*`) sont toujours fournies.
- Périmètre = machine entière (`tracking_mode="machine"`) : fermer les autres applications, secteur branché, plan d'alimentation fixé.
- Un seul tracker à la fois (garde dans `EnergyMeter`). Initialisation et warm-up hors tâche mesurée.
- Pour une mesure défendable : wattmètre externe (prise ou USB) en parallèle, CodeCarbon servant de comptabilité temps/énergie.
- Rappel : une réduction de débit n'est pas une économie d'énergie ; ne pas l'assimiler sans mesure.

## 10. Versions testées (environnement de développement de ces fichiers)

Linux, Python 3.12, torch 2.4.1 (CPU, 1 thread), constriction 0.5.0, codecarbon 3.3.2, **poids aléatoires**.
Les résultats de `results/_smoke_test_RANDOM_WEIGHTS_not_results/` ne sont **pas** des résultats expérimentaux.
Votre environnement cible (Windows, Python 3.10.11, torch 2.4.1 CPU) n'a pas été testé : lancer la validation du §6.

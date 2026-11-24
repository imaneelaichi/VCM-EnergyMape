# VCM-EnergyMap V5.1 = V5 (configuration YAML) + V4.1 (profilage énergétique)

Originaux **non modifiés** : V4 (`experiment3_frame_components_v4.py`, `energy_profiling.py`), V4.1 (`experiment3_frame_components_v41.py`,
`energy_profiling_v41.py`, `test_energy_v41.py`), V5 (`experiment3_frame_components_v5.py`, `exp_config.py`, `experiment_config.yaml`).
Codec VCM, checkpoints, modèles, architecture : **aucune modification**.

## 1. Audit des conflits (V4 → V5 vs V4 → V4.1)

Comparaison ligne à ligne : V5 modifie 36 blocs de V4, V4.1 en modifie 33. Zones modifiées par les DEUX :

| Zone de V4 | V5 | V4.1 | Résolution dans V5.1 |
|---|---|---|---|
| en-tête, docstring, `VERSION`, `RUN_NAME_DEFAULT` (l.4-62) | nouveau docstring, `VERSION=v5` | nouveau docstring, `VERSION=v41` | docstring unique ; `VERSION="v51"`, `exp3_v51_RAPL_10rep` ; import `exp_config_v51` |
| `record()` (l.645…) | contrôle I/P observé + colonnes `config_id/checkpoint/gop/resolution` | `energy_breakdown`, temps par phase, colonnes débit | **les deux** : ordre = contrôle I/P → net via `energy_breakdown` → temps → débit |
| `run_repetition()` | noms de fichiers `{checkpoint}_rep…`, tâches `{config_id}_r…` | `perf_counter` autour de l'appel codec, `record_bytes` | les deux, sans chevauchement de lignes |
| `build_parser()` | supprimé (paramètres issus du YAML) | ajout de `--fps` | `--fps` **déplacé dans le YAML** (`video.fps`) ; `fps_assumed` calculé par `exp_config_v51.resolve` |
| `main()` (l.1395) | devient `run_experiment(a, trace)` | `fps_assumed` depuis `sys.argv`, contrôle `--fps` | `a.fps/a.fps_assumed` fournis par `to_namespace` ; contrôle `--fps>0` conservé |
| bloc `definitions`/`limitations` (`emit`) | ajoute des définitions config | ajoute des définitions temps/débit | cumul |

Autres conflits traités :
* **Nom des modules** : `energy_profiling` (V4) / `energy_profiling_v41` / `exp_config` → modules V5.1 autonomes `energy_profiling_v51`, `exp_config_v51` (V4/V4.1/V5 peuvent coexister dans `svc_code`).
* **Dossiers de résultats** : V5 `exp3_v5`, V4.1 `exp3_v41_…` → `results\VCM_EnergyMap\exp3_v51\<id>_gop…` (jamais de collision avec V3/V4/V4.1/V5).
* **Double mesure / double comptage** : un seul `EnergyMeter` par processus (garde `_ACTIVE`) ; `--probe` n'utilise aucun tracker ; `--batch` = un processus par configuration, séquentiel ; garde `heavy_guard` contre une autre expérience concurrente ; net total = somme des nets (aucune double soustraction) ; énergie par composant = part de temps de l'énergie de l'image, la somme des composants = énergie brute de l'image (testé).
* **Une seule configuration par expérience** : `namespace_consistency` compare le moteur au YAML résolu (26 paramètres) avant toute exécution ; copie du YAML + `config_resolved.json` dans le dossier de résultats.

## 2. Fichiers livrés

| Fichier | Origine | Rôle |
|---|---|---|
| `experiment3_frame_components_v51.py` | V4.1 + couche V5 | moteur + CLI (`--list/--probe/--run/--batch/--check_energy`) |
| `energy_profiling_v51.py` | copie de `energy_profiling_v41.py` (nom seul changé + docstring) | `EnergyMeter`, `energy_breakdown`, `idle_power_stats` |
| `exp_config_v51.py` | `exp_config.py` V5 + fps + cohérence | lecture/validation YAML, entrées, checkpoints |
| `experiment_config_v51.yaml` | `experiment_config.yaml` V5 | C1–C4 conservées ; `results_group: exp3_v51` ; `video.fps` |
| `test_energy_v51.py` | `test_energy_v41.py` + tests config | 28 tests unitaires |
| `test_simulated_pipeline_v51.py` | nouveau | test de bout en bout avec composants simulés |

## 3. Modifications — fichier, lignes, nature

### `energy_profiling_v51.py`
| Lignes | Élément | Modification |
|---|---|---|
| 2–5 | docstring | « revision V5.1 », provenance (seul changement par rapport à `energy_profiling_v41.py`) |
| 60–286 | `EnergyMeter` | identique à V4/V4.1 : vérification RAPL réelle, classification CPU mesuré / RAM estimée |
| 297–298 | `NET_ABS_TOL_J`, `NET_REL_TOL` | de V4.1 |
| 301–302 | `_f` | de V4.1 |
| 305–337 | `energy_breakdown` | de V4.1 : brut/net par composant, net direct, cohérence, négatifs conservés |
| 340–353 | `idle_power_stats` | de V4.1 |

### `exp_config_v51.py` (depuis `exp_config.py` V5)
| Lignes | Élément | Modification |
|---|---|---|
| 1–3 | docstring | V5.1 |
| 102–137 | `resolve` | V5 ; + lecture de `video.fps` (131–136) : `fps_assumed=True` si absent (30 supposé), erreur si invalide |
| 145–203 | `inspect_checkpoint` | V5, inchangé (aucune hypothèse de « qualité ») |
| 206–221 | `compare_checkpoints` | V5, inchangé |
| 230–250, 253–283 | `input_plan`, `prepare_inputs` | V5, inchangés (cache dédié, jamais d'écrasement) |
| 287–293 | `estimate_params` | V5, inchangé |
| 301–418 | `static_checks` | V5 ; + contrôle « cadence fps » (335–340 : FAIL si ≤ 0, WARN si supposée) |
| 421–451 | `cross_checks` | V5, inchangé (détecteur identique, dossiers distincts, checkpoints) |
| 466–487 | `to_namespace` | V5 ; + `fps`, `fps_assumed` (484) |
| 490–509 | `namespace_consistency` | NOUVELLE : moteur == YAML résolu (26 paramètres) |

### `experiment3_frame_components_v51.py` (2245 lignes)
| Lignes | Fonction / zone | Origine | Modification |
|---|---|---|---|
| 1–25 | docstring | fusion | description V5.1 |
| 51–56 | `VERSION`, `import exp_config_v51`, `REQUIRED` | fusion | `v51`, modules V5.1 |
| 70–96 | `GROUPS`, `GROUP_SHORT`, `STAGE_IO_GROUP` | V4.1 | prétraitement / flux binaire / post-traitement séparés |
| 113–116 | `TIME_SPLIT_KEYS`, `BITRATE_KEYS`, `FRAME_KEYS`, `TYPE_KEYS` | V4.1 | statistiques temps + débit |
| 120–133, 136–147 | `bitrate_metrics`, `sequence_bitrate` | V4.1 | bits, bpp, bit/s (image, séquence) |
| 150–161 | `stage_group` | V4.1 | répartition des étapes d'E/S |
| 296–349 | `energy_label`, `energy_source_block`, `rapl_problem` | V4 | vérification RAPL réelle, inchangée |
| 355–366 | `load_modules` | V4.1 | importe `energy_profiling_v51` (`energy_breakdown`, `idle_power_stats`) |
| 469–501 | `collect_repro_metadata` | V4 + V5 | checkpoint issu du YAML |
| 565–588 | `Experiment.measure_idle` | V4 | baselines enregistrées dans `idle_baselines.csv` |
| 591–598 | `Experiment.type_check_summary` | V5 | bilan I/P observé |
| 600–633 | `Experiment.init_codecs` | V5 | résolution réelle des entrées == résolution demandée |
| 636–752 | `Experiment.record` | **fusion** | I/P observé (648) ; `energy_breakdown` (661) ; temps par phase (685–687) ; débit (691) ; colonnes de temps (709) ; `energy_net_consistent` (731) |
| 755–841 | `Experiment.run_repetition` | **fusion** | `perf_counter` autour de l'appel codec (786, 818) ; noms `{checkpoint}_rep…` |
| 844–965 | `Experiment.validate` | V4.1 | contrôle `net_consistency` (939) |
| 1071–1084 | `bitrate_block` | V4.1 | bloc `bitrate` de `summary.json` |
| 1087–1138 | `repeat_summary` | V4.1 | débit de séquence, temps par phase |
| 1144–1326 | `make_figures` | V4 | inchangée |
| 1330–1396 | `print_summary` | V4.1 + V5 | tableaux temps et débit ; nom de configuration |
| 1400–1408 | `estimate_seconds` | V4 | inchangée (paramètres issus du YAML) |
| 1411–1477 | `check_energy` | V4 | module énergie V5.1 |
| 1480–1889 | `run_experiment` (ex-`main`) | **fusion** | `out_dir` issu du YAML ; `detector_params.json` (1562) ; `codec_runtime` (1581) ; `frame_type_check` (1714) ; `idle_power_stats` et `bitrate` dans `summary.json` (1795–1798) ; définitions (1828…) |
| 1892–1908 | `FrameTypeMismatch`, `stage_signature`, `observed_frame_type` | V5 | I/P observé sans hypothèse sur les noms d'étapes |
| 1950–1960 | `heavy_guard` | V5 | refuse probe/run/batch si une autre expérience tourne |
| 1963–1975 | `verify_input_files` | V5 | résolution vérifiée sur les fichiers |
| 1978–2054 | `probe_config` | V5 | + bit/s à la cadence fps |
| 2057–2076 | `build_trace` | V5 | + `fps`, `fps_assumed` |
| 2079–2125 | `validate_ids`, `prepare_and_probe` | V5 | validation, préparation des entrées |
| 2128–2169 | `batch_run` | V5 | lot : un processus par configuration, lot refusé si la validation échoue |
| 2172–2245 | `main` | V5 | CLI `--list/--probe/--run/--batch/--check_energy` ; `namespace_consistency` avant `--run` (2238) |
| — | `build_parser` | supprimé | plus aucun paramètre expérimental dans le code |

### `experiment_config_v51.yaml`
`results_group: exp3_v51` ; `defaults.video.fps: null` (30 supposé tant que non renseigné) ; C1–C4, valeurs autorisées, détecteur, énergie, estimation, lot : inchangés.

## 4. Dépendances

Standard : argparse, csv, hashlib, importlib, json, math, os, pathlib, subprocess, sys, tempfile, time… Tierces : `pyyaml`, `Pillow`, `numpy`, `matplotlib`, `psutil`,
`codecarbon==3.3.2`, `torch`, `torchvision` (le module de configuration n'importe torch/PIL/yaml que dans les fonctions qui en ont besoin).
Projet VCM : `test_base_profiled.py`, `profiling.py`, `base_codec.py`, `bitstream.py`, `entropy_codec.py`. Tests : `pandas` (test simulé seulement).

## 5. Ce qui est testé / non testé

| Testé ici (composants simulés, CodeCarbon 3.3.2 réel en mode `cpu_load` sous Linux) | NON validé : à faire sur votre PC |
|---|---|
| imports (analyse AST), compilation, 28 tests unitaires | le **vrai codec VCM** (checkpoints réels, GOP 2/4/8/16, 768×432 / 640×360 / 384×216) |
| configurations invalides (GOP, résolution, checkpoint, images, détecteur, énergie, fps, dossier existant) et valides | **RAPL via Windows EMI** : `--check_energy` puis garde-fou au démarrage |
| `--list`, validation, `--probe` (+ cas GOP incohérent), `--run`, `--batch`, lot refusé si une validation échoue | durées réelles, estimation du lot |
| temps séparés, bpp, bit/s, net / net direct, négatifs conservés, pas de double comptage, séparation des dossiers | structure I/P réelle du codec (règle « mêmes étapes que l'image 0 ») |
| `--check_energy` honnête (échec attendu hors EMI) | cadence réelle `video.fps` |

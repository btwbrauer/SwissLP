# Class Weights Usage

## Übersicht

Class Weights können verwendet werden, um unausgeglichene Datensätze zu kompensieren. Standardmäßig sind Class Weights **deaktiviert**, um das Verhalten der Teammitglieder-Notebooks zu replizieren (nur vollständige Daten, perfekt ausgeglichene Klassen).

## Standard-Verhalten (wie Teammitglieder)

Standardmäßig werden nur vollständige Einträge verwendet (alle Dialekte vorhanden), was zu perfekt ausgeglichenen Klassen führt:

```yaml
data:
  dialects:
    - "ch_ag"
    - "ch_lu"
    # ...
  use_class_weights: false  # Default
```

**Ergebnis:**
- Nur ~2,535 Sätze (nur vollständige Einträge)
- Perfekt ausgeglichene Klassen (z.B. 2,022 pro Klasse im Training)
- Keine Class Weights nötig

## Class Weights aktivieren

Um Class Weights zu aktivieren, setze `use_class_weights: true` in der Config:

```yaml
data:
  dialects:
    - "ch_ag"
    - "ch_lu"
    # ...
  use_class_weights: true  # Aktiviert Class Weights
```

**Wann werden Class Weights angewendet?**

Class Weights werden nur angewendet, wenn:
1. `use_class_weights: true` in der Config gesetzt ist
2. Das Dataset tatsächlich unausgeglichen ist (Imbalance Ratio > 1.2)

**Wie funktionieren Class Weights?**

- Berechnung: `weight[klasse] = n_samples / (n_classes * count_per_class)`
- Seltene Klassen erhalten höhere Gewichte
- Häufige Klassen erhalten niedrigere Gewichte
- Automatische Normalisierung für Stabilität

## Beispiel-Config

```yaml
model:
  model_name: ZurichNLP/swissbert
  num_labels: 8

data:
  data_path: ./data/sentences_ch_de_transcribed.json
  dialects:
    - "ch_ag"
    - "ch_lu"
    - "ch_be"
    - "ch_zh"
    - "ch_vs"
    - "ch_bs"
    - "ch_gr"
    - "ch_sg"
  use_class_weights: true  # Aktiviert Class Weights

training:
  learning_rate: 2e-5
  num_epochs: 3
  # ...
```

## Wann sollte man Class Weights verwenden?

**Verwende Class Weights, wenn:**
- Du alle verfügbaren Daten nutzen möchtest (auch unvollständige Einträge)
- Das Dataset unausgeglichen ist
- Du bessere Performance für seltene Klassen erzielen möchtest

**Verwende KEINE Class Weights, wenn:**
- Du nur vollständige Einträge nutzt (wie Teammitglieder)
- Die Klassen bereits ausgeglichen sind
- Du das Standard-Verhalten replizieren möchtest

## Technische Details

Class Weights werden automatisch berechnet in `src/utils/dataset.py`:
- Funktion: `compute_class_weights()`
- Anwendung: Automatisch im `BaseTrainer` wenn aktiviert
- Implementierung: Gewichtete `CrossEntropyLoss` Funktion


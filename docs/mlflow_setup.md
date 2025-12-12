# MLflow Setup und Verwendung

## Übersicht

MLflow wird für Experiment-Tracking verwendet. Alle Trainings-Runs werden automatisch zu MLflow geloggt.

## MLflow Server (Empfohlen)

Ein MLflow Server läuft bereits auf `http://localhost:5000` mit SQLite-Backend.

**Vorteile:**
- ✅ Keine Deprecation-Warnungen
- ✅ SQLite-Backend (nicht deprecated)
- ✅ Zentrale Datenbank
- ✅ Mehrere Nutzer können gleichzeitig zugreifen

**UI öffnen:**
Öffne einfach im Browser: `http://localhost:5000`

**Status prüfen:**
```bash
curl http://localhost:5000/health
```

## Lokale MLflow UI (Nicht empfohlen)

Wenn du lokal `mlflow ui` startest, bekommst du Deprecation-Warnungen:

```
FutureWarning: Filesystem tracking backend (e.g., './mlruns') is deprecated.
```

**Warum?**
- Filesystem-Backend ist deprecated
- Sollte durch SQLite oder Server ersetzt werden

**Lösung:**
Nutze den MLflow Server auf `http://localhost:5000` statt lokal `mlflow ui` zu starten.

## Konfiguration

### Standard (Server)

Der Code verbindet sich automatisch mit dem MLflow Server:

```python
# Automatisch: http://localhost:5000
# Keine Konfiguration nötig
```

### Custom Tracking URI

Falls du einen anderen MLflow Server nutzen möchtest:

```bash
export MLFLOW_TRACKING_URI="http://your-server:5000"
```

### Lokales SQLite-Backend (Alternative)

Falls du kein Server nutzen möchtest, aber SQLite verwenden willst:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

Dann starte MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Was wird geloggt?

MLflow loggt automatisch:

- **Parameters**: Model-Config, Hyperparameter, Data-Splits
- **Metrics**: Training Loss, Validation Metrics (Accuracy, F1, etc.)
- **Artifacts**: Trainierte Modelle, Checkpoints, Config-Files
- **Experiment Metadata**: Run Name, Timestamp, Experiment Name

## Troubleshooting

### "Address already in use"

Der MLflow Server läuft bereits. Nutze einfach `http://localhost:5000` im Browser.

### Deprecation-Warnungen

Diese erscheinen nur, wenn du lokal `mlflow ui` startest. Nutze stattdessen den Server.

### Verbindungsfehler

Prüfe ob der Server läuft:
```bash
curl http://localhost:5000/health
```

Falls nicht, starte den Server (siehe System-Konfiguration).


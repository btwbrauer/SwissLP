# Minimale Konfiguration für Pyright + Nix + direnv

## Notwendige Dateien (Minimum)

### 1. `flake.nix` - MUSS vorhanden sein
```nix
{
  outputs = { self, nixpkgs, ... }:
    {
      devShells.default = pkgs.mkShell {
        packages = [
          pythonEnv    # Python-Umgebung mit allen Paketen
          pkgs.pyright # Pyright LSP
        ];
      };
    };
}
```

**Wichtig:**
- `pythonEnv` muss in `packages` sein (setzt Python in PATH)
- `pkgs.pyright` muss in `packages` sein (für LSP-Support)

### 2. `.envrc` - MUSS vorhanden sein
```
use flake
```

**Wichtig:**
- Aktiviert die direnv-Umgebung automatisch
- Setzt Python und alle Pakete in PATH

## Optionale Dateien

### `pyrightconfig.json` - OPTIONAL
- **Nicht nötig** wenn Pyright Python aus PATH findet
- Nur für spezielle Einstellungen (z.B. `typeCheckingMode`, `reportMissingImports`)
- visual-odometry und ShapingRL haben **keine** pyrightconfig.json

### `pyproject.toml` - OPTIONAL
- Für Projekt-Metadaten und Dependencies-Liste
- **Nicht nötig** für Pyright (Pyright nutzt die Nix-Umgebung)

## Wie es funktioniert

1. **direnv** liest `.envrc` → aktiviert `use flake`
2. **Nix** baut die Umgebung aus `flake.nix`
3. **Python** wird in PATH gesetzt (von `pythonEnv` in `packages`)
4. **Pyright** findet Python automatisch aus PATH
5. **Pyright** findet Pakete automatisch aus der Python-Umgebung

## Warum es in anderen Projekten funktioniert

- ✅ `flake.nix` enthält `pythonEnv` und `pyright`
- ✅ `.envrc` enthält `use flake`
- ✅ **Keine** `pyrightconfig.json` (Pyright nutzt Auto-Detection)
- ✅ **Keine** `.vscode/settings.json` (nicht nötig)

## Troubleshooting

Wenn Pyright Imports nicht findet:
1. Prüfe: `which python` → sollte Nix-Store-Pfad zeigen
2. Prüfe: `python -c "import transformers"` → sollte funktionieren
3. Editor neu laden (Cmd/Ctrl+Shift+P → "Reload Window")
4. direnv-Erweiterung in Cursor/VS Code aktiviert?


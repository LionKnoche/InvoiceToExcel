"""
FACHBEGRIFFE - Machine Learning Grundlagen:

Datenstrukturen:
- Feature Matrix (X_train): Eingabedaten, Zeilen = Samples, Spalten = Features
- Target Vector (y_train): Zielwerte/Labels für jedes Sample
- Shape: Dimensionen eines Arrays (z.B. (3, 2) = 3 Zeilen, 2 Spalten)
- 2D-Array/Matrix: Daten mit Zeilen und Spalten
- 1D-Array/Vektor: Daten mit nur Indizes

Machine Learning Konzepte:
- Supervised Learning: Modell lernt aus Input-Output-Paaren
- Training/Fitting: Modell lernt aus Daten (model.fit())
- Classification: Modell ordnet Daten in Kategorien ein
- Pattern Learning: Modell erkennt Muster in den Daten
- Generalization: Modell kann auf neue, ähnliche Daten anwenden
- Inference: Vorhersage mit trainiertem Modell
- Predict: Modell gibt Vorhersagen zurück (model.predict())
- Algorithm: Mathematische Methode zum Lernen (z.B. RandomForestClassifier)
- Model Instance: Konkrete Ausprägung nach dem Training
- Re-training: Modell neu trainieren
- Ground Truth: Die Labels in y_train sind die "Wahrheit"

Preprocessing:
- Preprocessing: Datenvorverarbeitung vor dem Training
- Standardisierung: Normalisierung auf Mittelwert 0, Standardabweichung 1
- Feature Scaling: Skalierung der Features auf ähnliche Bereiche
- Transform: Daten mit trainiertem Scaler normalisieren

Artifacts & Serialization:
- Artifacts: Gespeicherte Outputs (trainierte Modelle, Preprocessing-Pipelines)
- Serialization: Objekte in Dateien speichern (joblib.dump())
- Deserialization: Objekte aus Dateien laden (joblib.load())
- Artifact Store: Verzeichnis/System, in dem Artifacts gespeichert werden

CLI & Entwicklung:
- CLI (Command Line Interface): Programm über Terminal steuern
- argparse: Bibliothek für Kommandozeilenargumente
- Parsing: Text/Argumente in strukturierte Daten umwandeln
- Namespace Object: Objekt mit Attributen (args.features)
- Virtual Environment (venv): Isolierte Python-Umgebung für ein Projekt
- Traceback/Stack Trace: Fehlerpfad durch den Code
- ModuleNotFoundError: Paket nicht installiert
"""

import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse

# Beispiel: Ein einfaches Modell trainieren (supervised learning)
X_train = np.array([[1, 2], [3, 4], [5, 6]])
y_train = np.array([0, 1, 0])

# Modell trainieren
# supervised learning
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# Preprocessor trainieren
scaler = StandardScaler()
scaler.fit(X_train)

# Artifacts speichern (das sind die "Artefakte")
joblib.dump(model, 'model.pkl')  # Trainiertes Modell
joblib.dump(scaler, 'scaler.pkl')  # Preprocessing-Pipeline

# Später wieder laden
loaded_model = joblib.load('model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Terminal-Interface für Vorhersagen
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML-Modell Vorhersage")
    parser.add_argument("--features", type=float, nargs="+", 
                       required=True, help="Features als Liste: --features 2 3")
    args = parser.parse_args()
    
    # Neue Daten vorbereiten
    X_new = np.array([args.features])
    
    # Mit Scaler transformieren
    X_new_scaled = loaded_scaler.transform(X_new)
    
    # Vorhersage machen
    prediction = loaded_model.predict(X_new_scaled)
    
    print(f"Features: {args.features}")
    print(f"Vorhersage: {prediction[0]}")
import joblib
import json
import os

pedestrian_id = 0 # ID of the pedestrian to track (so far only one pedestrian is tracked)

# --- NEU: Basis-Ausgabeverzeichnis definieren und einmalig erstellen ---
# Der Pfad, wo die finalen JSON-Dateien gespeichert werden sollen
output_base_dir = 'Trajectory Prediction/test_data/atlas_json/mmcp/'
os.makedirs(output_base_dir, exist_ok=True) # Erstellt alle notwendigen Zwischenverzeichnisse, falls sie nicht existieren

# Basis-Input-Verzeichnis
input_base_dir = 'Trajectory Prediction/test_data/pedestrian_gps/'

# Optional: Stellen Sie sicher, dass das Input-Basisverzeichnis existiert, bevor Sie es auflisten
if not os.path.isdir(input_base_dir):
    print(f"Fehler: Das Eingabeverzeichnis '{input_base_dir}' existiert nicht.")
    exit() # Beendet das Skript, wenn das Eingabeverzeichnis fehlt

# --- Hauptschleife durch die Frame-Ordner ---
for frame_folder in os.listdir(input_base_dir):
    # Vollständiger Pfad zum aktuellen Frame-Ordner im Input-Verzeichnis
    current_frame_input_path = os.path.join(input_base_dir, frame_folder)

    # Sicherstellen, dass es sich um einen Ordner und nicht um eine Datei handelt
    if not os.path.isdir(current_frame_input_path):
        continue # Überspringt, wenn es keine Datei ist

    # Vollständiger Pfad zur 'ped_gps_per_frame'-Datei
    ped_gps_file_path = os.path.join(current_frame_input_path, 'ped_gps_per_frame')

    # Der vollständige Pfad für die Atlas JSON-Ausgabedatei
    output_json_file_path = os.path.join(output_base_dir, f"{frame_folder}_atlas.json")

    coordinate_dict = {}
    try:
        # Laden der Daten
        coordinate_dict = joblib.load(ped_gps_file_path)
    except FileNotFoundError:
        print(f"Warnung: '{ped_gps_file_path}' nicht gefunden. Überspringe diesen Ordner.")
        continue # Gehe zum nächsten Ordner in der Schleife
    except Exception as e:
        print(f"Fehler beim Laden von '{ped_gps_file_path}': {e}. Überspringe diesen Ordner.")
        continue

    atlas_trajectory_points = []

    sorted_coordinate_dict = sorted(coordinate_dict.items())
    for frame_id, coords_tuple in sorted_coordinate_dict:
        # Überprüfen, ob coords_tuple wirklich ein Tupel mit 2 Elementen ist
        if not isinstance(coords_tuple, (list, tuple)) or len(coords_tuple) != 2:
            print(f"Warnung: Unerwartetes Koordinatenformat für Frame {frame_id} im Ordner {frame_folder}. Überspringe Punkt.")
            continue
        
        x_coord, y_coord = coords_tuple # Entpacke das Tupel der Koordinaten

        # Erstelle das innere Dictionary für den Trajektorienpunkt
        track_point_data = {
            "f": int(frame_id),      # Frame ID (als Integer)
            "p": pedestrian_id,      # Personen ID (die feste ID für diesen Pedestrian)
            "x": float(x_coord),     # X-Koordinate (als Float)
            "y": float(y_coord)      # Y-Koordinate (als Float)
        }
        
        # Erstelle das äußere Dictionary mit dem Schlüssel "track"
        atlas_entry = {"track": track_point_data}

        # Füge das fertige Dictionary zur Liste hinzu
        atlas_trajectory_points.append(atlas_entry)

    # --- Ausgabe auf der Konsole (zu Debugging-Zwecken) ---
    print(f"\n--- Ausgabe für '{frame_folder}' im Atlas JSON Format (zeilenweise) ---")
    if not atlas_trajectory_points:
        print("Keine Trajektorienpunkte für diesen Ordner generiert.")
    else:
        for entry in atlas_trajectory_points:
            json_string = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
            print(json_string)

    # --- Speichern der Trajektoriendaten ---
    # Die ursprüngliche os.makedirs-Zeile für den Input-Pfad kann hier entfernt werden,
    # da die Input-Ordner bereits existieren müssen, damit os.listdir funktioniert.
    # os.makedirs(f'Trajectory Prediction/test_data/pedestrian_gps/{frame_folder}/', exist_ok=True) # <-- DIESE ZEILE HIER ENTFERNEN ODER KORRIGIEREN

    try:
        # Öffne die Datei im Schreibmodus
        with open(output_json_file_path, 'w', encoding='utf-8') as f:
            # Schreibe jeden Trajektorienpunkt als eine eigene JSON-Zeile
            for entry in atlas_trajectory_points:
                json_string = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
                f.write(json_string + '\n')

        print(f"\nErfolgreich Trajektoriendaten in '{output_json_file_path}' gespeichert.")
    except IOError as e:
        print(f"\nFehler beim Speichern der Datei '{output_json_file_path}': {e}")
    except Exception as e:
        print(f"\nEin unerwarteter Fehler ist aufgetreten beim Speichern von '{output_json_file_path}': {e}")

    pedestrian_id += 1 # Erhöhe die ID für den nächsten Pedestrian

print("\n--- Verarbeitung abgeschlossen ---")
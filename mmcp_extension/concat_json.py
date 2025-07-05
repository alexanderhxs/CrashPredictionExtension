import os

# Bestimme das Verzeichnis, in dem sich dieses Skript befindet
script_dir = os.path.dirname(os.path.abspath(__file__))

# Bestimme den ROOT_DIR.
# Basierend auf Ihrem Copy-Pfad scheint 'Crash-Prediction' das eigentliche Stammverzeichnis zu sein,
# von dem aus relative Pfade (wie 'Trajectory Prediction/...') beginnen.
# Wir müssen von script_dir so weit hochgehen, bis wir bei 'Crash-Prediction' sind.

# Wenn das Skript direkt in 'Crash-Prediction' liegt:
# ROOT_DIR = script_dir

# Wenn das Skript in einem Unterordner von 'Crash-Prediction' liegt, z.B. 'Crash-Prediction/scripts':
# ROOT_DIR = os.path.dirname(script_dir)

# Wenn das Skript in einem Unterordner von 'Crash-Prediction/Trajectory Prediction' liegt:
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(script_dir))) # 3x hoch

# AM ROBUSTESTEN: Suchen Sie nach dem spezifischen Ordnernamen 'Crash-Prediction'
current_dir = script_dir
while not os.path.basename(current_dir) == 'Crash-Prediction' and current_dir != os.path.dirname(current_dir):
    current_dir = os.path.dirname(current_dir)

if os.path.basename(current_dir) == 'Crash-Prediction':
    ROOT_DIR = current_dir
    print(f"ROOT_DIR automatisch erkannt als: {ROOT_DIR}")
else:
    print("Warnung: 'Crash-Prediction' Verzeichnis nicht gefunden im Pfadbaum.")
    print(f"Aktuelles Skript-Verzeichnis: {script_dir}")
    print("Bitte passen Sie ROOT_DIR manuell an, um auf Ihr Projekt-Stammverzeichnis zu zeigen.")
    # Fallback, wenn 'Crash-Prediction' nicht gefunden wird, dann versuchen Sie den allgemeinen Ansatz
    # der 2 Ebenen hochgeht, falls das Skript in einem Unterordner wie 'Trajectory Prediction/test_data/atlas_json/mmcp' liegt
    # oder passen Sie den Pfad zu Ihrem Skript an, wenn es sich woanders befindet.
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(script_dir))) # Geht 3 Ebenen hoch, falls Skript in .../mmcp liegt

# Der relative Pfad zu Ihren Daten vom ROOT_DIR aus
# Basierend auf 'C:\Dokumente\Studium\HiWi\code\multi-modal-crash-prediction\Crash-Prediction\Trajectory Prediction\test_data\atlas_json\mmcp'
# und ROOT_DIR = 'C:\Dokumente\Studium\HiWi\code\multi-modal-crash-prediction\Crash-Prediction'
# ist der relative Pfad: 'Trajectory Prediction\test_data\atlas_json\mmcp'
relative_data_path_from_root = os.path.join('Trajectory Prediction', 'test_data', 'atlas_json', 'mmcp')
input_data_directory = os.path.join(ROOT_DIR, relative_data_path_from_root)


# Der Name der Ausgabedatei für die Pfadliste
output_txt_file_name = 'mmcp_atlas_paths.txt'
# Der Pfad zur Ausgabedatei (neben diesem Skript)
output_txt_file_path = os.path.join(script_dir, output_txt_file_name)


print(f"Suche nach '_atlas.json'-Dateien in: {input_data_directory}")
print(f"Speichere Pfade in: {output_txt_file_path}")

atlas_file_paths = []

if not os.path.isdir(input_data_directory):
    print(f"Fehler: Eingabeverzeichnis '{input_data_directory}' nicht gefunden.")
    print("Bitte überprüfen Sie die Variable 'input_data_directory' und 'ROOT_DIR' im Skript.")
else:
    for filename in os.listdir(input_data_directory):
        if filename.endswith('_atlas.json') and os.path.isfile(os.path.join(input_data_directory, filename)):
            
            relative_path = os.path.join('datasets','mmcp',filename)
            atlas_file_paths.append(relative_path + ',')
            print(f"Gefunden und hinzugefügt: {relative_path}")

    if atlas_file_paths:
        try:
            with open(output_txt_file_path, 'w', encoding='utf-8') as f:
                for path in atlas_file_paths:
                    f.write(path + '\n')
            print(f"\nErfolgreich {len(atlas_file_paths)} Dateipfade in '{output_txt_file_path}' gespeichert.")
        except IOError as e:
            print(f"Fehler beim Schreiben der Datei '{output_txt_file_path}': {e}")
    else:
        print("\nKeine '_atlas.json'-Dateien im angegebenen Verzeichnis gefunden.")

print("\n--- Skript abgeschlossen ---")


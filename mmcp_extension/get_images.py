import cv2
import os


def extract_frames_from_mp4(video_path, output_folder):
    """
    Extrahiert alle Frames aus einer MP4-Videodatei und speichert sie als PNG-Bilder.

    Args:
        video_path (str): Der vollständige Pfad zur MP4-Eingabedatei.
        output_folder (str): Der Pfad zum Ordner, in dem die extrahierten Frames gespeichert werden sollen.
                             Der Ordner wird erstellt, falls er nicht existiert.
    """
    # Überprüfen, ob die Eingabedatei existiert
    if not os.path.exists(video_path):
        print(f"Fehler: Videodatei nicht gefunden unter '{video_path}'")
        return

    # Ausgabeordner erstellen, falls er nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Ausgabeordner '{output_folder}' erstellt.")
    else:
        print(f"Warnung: Ausgabeordner '{output_folder}' existiert bereits. Vorhandene Dateien könnten überschrieben werden.")


    print(f"Öffne Videodatei: '{video_path}'")
    # Videodatei öffnen
    cap = cv2.VideoCapture(video_path)

    # Überprüfen, ob die Videodatei erfolgreich geöffnet wurde
    if not cap.isOpened():
        print(f"Fehler: Konnte Videodatei '{video_path}' nicht öffnen.")
        return

    frame_count = 0
    print("Beginne Frame-Extraktion...")

    while True:
        # Frame lesen
        ret, frame = cap.read()

        # Wenn ret False ist, bedeutet das, dass keine Frames mehr übrig sind (Ende des Videos)
        if not ret:
            break

        # Dateiname für den Frame erstellen (mit führenden Nullen für einfache Sortierung)
        # Beispiel: frame_00000.png, frame_00001.png, ...
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")

        # Frame als PNG-Datei speichern
        success = cv2.imwrite(frame_filename, frame)

        if success:
            # Optional: Fortschritt ausgeben (z.B. alle 100 Frames)
            if frame_count % 100 == 0:
                 print(f"Frame {frame_count} gespeichert.")
            frame_count += 1
        else:
            print(f"Fehler beim Speichern von Frame {frame_count}.")
            # Optional: Abbruch bei Speicherfehler
            # break


    # Videodatei schließen und Ressourcen freigeben
    cap.release()

    print(f"Frame-Extraktion abgeschlossen.")
    print(f"Gesamt {frame_count} Frames extrahiert nach '{output_folder}'")

# --- ANWENDUNG ---

input_base_directory = '../Crash-Prediction-Data/'

# Definiere das Basis-Verzeichnis, wohin die Frame-Ordner gespeichert werden sollen
# Für jede MP4-Datei wird ein Unterordner erstellt.
output_base_directory_for_frames = './Trajectory Prediction/test_data/'

# Stelle sicher, dass das Basis-Ausgabeverzeichnis existiert
os.makedirs(output_base_directory_for_frames, exist_ok=True)

# Überprüfe, ob das Eingabeverzeichnis existiert
if not os.path.isdir(input_base_directory):
    print(f"Fehler: Eingabeverzeichnis nicht gefunden: {input_base_directory}")
else:
    print(f"Suche nach MP4-Dateien in: {input_base_directory}")
    # Gehe durch alle Elemente im Eingabeverzeichnis
    for item_name in os.listdir(input_base_directory):
        # Erstelle den vollständigen Pfad zum Element
        full_input_path = os.path.join(input_base_directory, item_name)

        # Prüfe, ob das Element eine Datei ist UND mit '.mp4' endet (Groß-/Kleinschreibung ignorieren)
        if os.path.isfile(full_input_path) and item_name.lower().endswith('.mp4'):
            print(f"\n--- Verarbeite Datei: {item_name} ---")

            # Erzeuge den Namen des Ausgabe-Unterordners (ohne Dateiendung)
            # z.B. '2024-08-22-15-06-55_head_front_camera_color_image_raw_compressed'
            output_folder_name = os.path.splitext(item_name)[0]

            # Erstelle den vollständigen Pfad für den Ausgabe-Ordner für diese spezifische Datei
            output_directory_for_current_video = os.path.join(output_base_directory_for_frames, output_folder_name)

            # Erstelle den Ausgabe-Ordner für diese Datei, falls er noch nicht existiert
            os.makedirs(output_directory_for_current_video, exist_ok=True)

            # Rufe die Funktion extract_frames_from_mp4 für diese Datei auf
            extract_frames_from_mp4(full_input_path, output_directory_for_current_video)

        # Optional: Wenn du andere Dateitypen oder Ordner ignorieren möchtest, kannst du hier eine else-Bedingung hinzufügen
        # else:
        #     print(f"Ignoriere: {item_name} (keine MP4-Datei)")

    print("\n--- Verarbeitung abgeschlossen ---")
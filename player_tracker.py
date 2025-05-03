import os
import cv2
import numpy as np
import json
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import supervision as sv
import argparse
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Ścieżki
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
CORNERS_DATA_FILE = os.path.join(PARENT_DIR, 'pitch_corners.json')
LOG_DIR = os.path.join(PARENT_DIR, 'log')

# Stałe
PLAYER_CLASS_ID = 2
CONFIDENCE_THRESHOLD = 0.4

def select_file():
    """Otwiera okno dialogowe do wyboru pliku wideo."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Wybierz plik wideo",
        filetypes=[("Pliki wideo", "*.mp4 *.avi *.mkv"), ("Wszystkie pliki", "*.*")]
    )
    root.destroy()
    return file_path

def load_corners_data(corners_data_file):
    """Wczytuje dane o rogach boiska z pliku JSON."""
    if os.path.exists(corners_data_file):
        try:
            with open(corners_data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Błąd wczytywania danych o rogach boiska: {str(e)}")
    return {}

def get_video_name(video_path):
    """Zwraca nazwę pliku wideo."""
    return os.path.basename(video_path)

def pixel_to_meters(pixel_point, corners, pitch_width, pitch_length, frame_shape):
    """Przelicza pozycję pikseli na pozycję w metrach na boisku."""
    if not all(corner is not None for corner in corners):
        return None
    src_points = np.float32([corners])
    dst_points = np.float32([
        [0, 0],
        [pitch_width, 0],
        [pitch_width, pitch_length],
        [0, pitch_length]
    ])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    pixel_point = np.float32([[pixel_point]])
    meter_point = cv2.perspectiveTransform(pixel_point, M)[0][0]
    return meter_point[0], meter_point[1]

def create_tracker(tracker_type):
    # Sprawdź, czy trackery są w module legacy (OpenCV 4.5+)
    if tracker_type == 'CSRT':
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
        else:
            return cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        else:
            return cv2.TrackerKCF_create()
    elif tracker_type == 'MOSSE':
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
            return cv2.legacy.TrackerMOSSE_create()
        else:
            return cv2.TrackerMOSSE_create()
    elif tracker_type == 'MEDIANFLOW':
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMedianFlow_create'):
            return cv2.legacy.TrackerMedianFlow_create()
        else:
            return cv2.TrackerMedianFlow_create()
    else:
        # Domyślnie CSRT
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
        else:
            return cv2.TrackerCSRT_create()

def track_players(video_path, output_dir, tracker_type="CSRT"):
    """
    Śledzi piłkarzy w wideo po ręcznym zaznaczeniu.
    
    Args:
        video_path (str): Ścieżka do pliku wideo
        output_dir (str): Ścieżka do katalogu z danymi wyjściowymi
        tracker_type (str): Typ algorytmu śledzenia (CSRT, KCF, MOSSE, MEDIANFLOW)
    """
    # Upewnij się, że katalog wyjściowy istnieje
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Otwórz wideo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Nie można otworzyć pliku wideo: {video_path}")
        return
    
    # Pobierz właściwości wideo
    video_name = get_video_name(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Domyślne FPS jeśli nie można odczytać z wideo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Wczytaj dane o rogach boiska
    corners_data = load_corners_data(CORNERS_DATA_FILE)
    
    # Sprawdź czy mamy dane dla tego wideo
    if video_name not in corners_data or '0' not in corners_data[video_name]:
        print(f"Brak danych o rogach boiska dla wideo: {video_name}")
        return
    
    # Wymiary boiska
    pitch_width = corners_data[video_name]['dimensions']['width']
    pitch_length = corners_data[video_name]['dimensions']['length']
    
    # Odczytaj pierwszą klatkę
    ret, frame = cap.read()
    if not ret:
        print("Nie można odczytać pierwszej klatki wideo.")
        return
    
    # Instrukcja dla użytkownika
    print("Instrukcje:")
    print("1. Zaznacz zawodnika prostokątem i naciśnij ENTER")
    print("2. Wybierz 'y', aby kontynuować zaznaczanie kolejnych zawodników lub 'n', aby zakończyć")
    
    # Lista zawodników do śledzenia
    players = []
    player_id = 0
    
    # Pętla zaznaczania zawodników
    while True:
        # Wyświetl ramkę z zaznaczonymi już zawodnikami
        display_frame = frame.copy()
        for p in players:
            x, y, w, h = [int(v) for v in p['bbox']]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"ID: {p['id']}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Zaznacz zawodnika", display_frame)
        cv2.waitKey(1)
        
        # Pozwól użytkownikowi zaznaczyć zawodnika
        bbox = cv2.selectROI("Zaznacz zawodnika", display_frame, fromCenter=False, showCrosshair=True)
        
        if bbox[2] > 0 and bbox[3] > 0:  # Sprawdź czy zaznaczono obszar
            tracker = create_tracker(tracker_type)
            tracker.init(frame, bbox)
            
            # Wylicz środek i promień (dla zgodności z istniejącym kodem)
            center_x = int(bbox[0] + bbox[2]/2)
            center_y = int(bbox[1] + bbox[3]/2)
            radius = max(int(bbox[2]/2), int(bbox[3]/2))
            
            # Dodaj zawodnika do listy
            players.append({
                'id': player_id,
                'bbox': bbox,
                'tracker': tracker,
                'data': [],
                'center_x': center_x,
                'center_y': center_y,
                'radius': radius
            })
            player_id += 1
        
        # Zapytaj czy chcesz dodać kolejnego zawodnika
        cv2.destroyWindow("Zaznacz zawodnika")
        print(f"Dodano zawodnika ID: {player_id-1}. Czy chcesz dodać kolejnego? (y/n)")
        choice = input().strip().lower()
        if choice == 'n':
            break
    
    # Przewiń do początku wideo
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    
    print(f"Rozpoczynam śledzenie {len(players)} zawodników w wideo {video_name}...")
    
    # Główna pętla przetwarzania klatek
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Aktualizacja paska postępu co 100 klatek
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            print(f"Postęp: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Wczytaj aktualne rogi boiska dla tej klatki (lub użyj domyślnych)
        current_corners = corners_data.get(video_name, {}).get(str(frame_count), corners_data[video_name]['0'])
        
        # Zaktualizuj pozycje zawodników
        display_frame = frame.copy()
        timestamp = frame_count / fps
        
        for player in players:
            success, bbox = player['tracker'].update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                center_x = int(x + w/2)
                center_y = int(y + h/2)
                radius = max(int(w/2), int(h/2))
                
                # Przelicz pozycję na metry
                pixel_pos = (center_x, center_y)
                meter_pos = pixel_to_meters(pixel_pos, current_corners, pitch_width, pitch_length, (frame_width, frame_height))
                
                if meter_pos is None:
                    meter_x, meter_y = -1, -1
                else:
                    meter_x, meter_y = meter_pos
                
                # Zapisz dane
                player['data'].append({
                    'frame': int(frame_count),
                    'time': float(timestamp),
                    'x': float(center_x),
                    'y': float(center_y),
                    'radius': float(radius),
                    'class_id': int(PLAYER_CLASS_ID),
                    'PozycjaX_Metry': float(meter_x),
                    'PozycjaY_Metry': float(meter_y)
                })
                
                # Rysuj na obrazie
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID: {player['id']}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Jeśli śledzenie zawiodło, zapisz pustą ramkę
                player['data'].append({
                    'frame': int(frame_count),
                    'time': float(timestamp),
                    'x': -1.0,
                    'y': -1.0,
                    'radius': -1.0,
                    'class_id': int(PLAYER_CLASS_ID),
                    'PozycjaX_Metry': -1.0,
                    'PozycjaY_Metry': -1.0
                })
        
        # Wyświetl obraz ze śledzeniem
        cv2.imshow("Śledzenie zawodników", display_frame)
        key = cv2.waitKey(1)
        if key == 27:  # Esc
            break
    
    # Zamknij plik wideo
    cap.release()
    cv2.destroyAllWindows()
    
    # Zapisz dane w formacie JSON dla każdego zawodnika
    for player in players:
        player_id = player['id']
        output_json = os.path.join(output_dir, f"player_{player_id}_intermediate.json")
        with open(output_json, 'w') as f:
            json.dump(player['data'], f, indent=2)
        print(f"Dane zawodnika {player_id} zapisane w: {output_json}")
    
    print("Śledzenie zawodników zakończone.")
    return players

def main():
    parser = argparse.ArgumentParser(description='Śledzenie piłkarzy w wideo po ręcznym zaznaczeniu')
    parser.add_argument('--video', type=str, help='Ścieżka do pliku wideo')
    parser.add_argument('--output', type=str, default='log', help='Ścieżka do katalogu wyjściowego')
    parser.add_argument('--tracker', type=str, default='CSRT', 
                        choices=['CSRT', 'KCF', 'MOSSE', 'MEDIANFLOW'],
                        help='Algorytm śledzenia do użycia')
    
    args = parser.parse_args()
    
    # Upewnij się, że folder log istnieje
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Jeśli nie podano ścieżki do wideo, otwórz okno dialogowe
    if not args.video:
        args.video = select_file()
        
        if not args.video:
            print("Nie wybrano pliku wideo. Kończę.")
            return
    
    # Rozpocznij śledzenie zawodników
    track_players(args.video, args.output, args.tracker)

def load_pitch_data(pitch_file):
    with open(pitch_file, "r") as f:
        data = json.load(f)
    key = list(data.keys())[0]
    dims = data[key]["dimensions"]
    return dims["width"], dims["length"]

def load_players_positions(players_dir):
    players = []
    for fname in os.listdir(players_dir):
        if fname.startswith("player_") and fname.endswith(".json"):
            with open(os.path.join(players_dir, fname), "r") as f:
                data = json.load(f)
            # Szukamy ostatniej pozycji, gdzie x i y >= 0
            for row in reversed(data):
                if row.get("x", -1) >= 0 and row.get("y", -1) >= 0:
                    # Konwersja na float (naprawa problemu z float32)
                    players.append({
                        "id": int(fname.split("_")[1]),
                        "x": float(row.get("PozycjaX_Metry", 0)),
                        "y": float(row.get("PozycjaY_Metry", 0))
                    })
                    break
    return players

def draw_static_map(pitch_width, pitch_length, players):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.add_patch(plt.Rectangle((0, 0), pitch_width, pitch_length, fill=False, edgecolor='green', lw=2))
    ax.plot([0, pitch_width], [pitch_length/2, pitch_length/2], 'g--')
    for player in players:
        ax.plot(player['x'], player['y'], 'ro', markersize=10)
        ax.text(player['x']+1, player['y']+1, f"ID: {player['id']}", color='black')
    ax.set_xlim(0, pitch_width)
    ax.set_ylim(0, pitch_length)
    ax.set_xlabel('Szerokość boiska (m)')
    ax.set_ylabel('Długość boiska (m)')
    ax.set_title('Pozycje zawodników na boisku')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def draw_interactive_map(pitch_width, pitch_length, players):
    if not PLOTLY_AVAILABLE:
        print("Plotly nie jest zainstalowane! Zainstaluj: pip install plotly")
        return
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=pitch_width, y1=pitch_length, line=dict(color="green", width=3))
    fig.add_shape(type="line", x0=0, y0=pitch_length/2, x1=pitch_width, y1=pitch_length/2, line=dict(color="green", dash="dash"))
    fig.add_trace(go.Scatter(
        x=[p['x'] for p in players],
        y=[p['y'] for p in players],
        mode='markers+text',
        marker=dict(size=15, color='red'),
        text=[f"ID: {p['id']}" for p in players],
        textposition="top center"
    ))
    fig.update_layout(
        title="Pozycje zawodników na boisku",
        xaxis=dict(range=[0, pitch_width], title="Szerokość (m)"),
        yaxis=dict(range=[0, pitch_length], title="Długość (m)"),
        yaxis_scaleanchor="x",
        width=800,
        height=500
    )
    fig.show()

if __name__ == "__main__":
    main()
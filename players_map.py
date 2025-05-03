import os
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

PITCH_FILE = "pitch_corners.json"
PLAYERS_DIR = "logs"

def load_pitch_data(pitch_file):
    with open(pitch_file, "r") as f:
        data = json.load(f)
    key = list(data.keys())[0]
    dims = data[key]["dimensions"]
    return dims["width"], dims["length"]

def load_players_tracks(players_dir):
    tracks = {}
    for fname in os.listdir(players_dir):
        if fname.startswith("player_") and fname.endswith(".json"):
            with open(os.path.join(players_dir, fname), "r") as f:
                data = json.load(f)
            player_id = int(fname.split("_")[1])
            track = []
            for row in data:
                x = row.get("PozycjaX_Metry", -1)
                y = row.get("PozycjaY_Metry", -1)
                time = row.get("time", None)
                if x is not None and y is not None and x >= 0 and y >= 0:
                    track.append((float(x), float(y), float(time) if time is not None else None))
                else:
                    track.append(None)
            tracks[player_id] = track
    return tracks

def get_max_frames(tracks):
    return max(len(track) for track in tracks.values())

def draw_pitch(ax, width, length):
    # Boisko
    ax.add_patch(plt.Rectangle((0, 0), width, length, fill=False, edgecolor='green', lw=3))
    # Linia środkowa
    ax.plot([0, width], [length/2, length/2], 'g-', lw=2)
    # Koło środkowe
    center_circle = plt.Circle((width/2, length/2), 9.15, color='green', fill=False, lw=2)
    ax.add_patch(center_circle)
    # Punkt środkowy
    ax.plot(width/2, length/2, 'go', markersize=6)
    # Pola karne
    penalty_area_w = 40.3
    penalty_area_h = 16.5
    ax.add_patch(plt.Rectangle(((width-penalty_area_w)/2, 0), penalty_area_w, penalty_area_h, fill=False, edgecolor='green', lw=2))
    ax.add_patch(plt.Rectangle(((width-penalty_area_w)/2, length-penalty_area_h), penalty_area_w, penalty_area_h, fill=False, edgecolor='green', lw=2))
    # Punkty karne
    ax.plot(width/2, 11, 'go', markersize=6)
    ax.plot(width/2, length-11, 'go', markersize=6)
    # Bramki
    goal_w = 7.32
    ax.add_patch(plt.Rectangle(((width-goal_w)/2, -2.44), goal_w, 2.44, fill=False, edgecolor='black', lw=2))
    ax.add_patch(plt.Rectangle(((width-goal_w)/2, length), goal_w, 2.44, fill=False, edgecolor='black', lw=2))
    # Rogi
    for x in [0, width]:
        for y in [0, length]:
            corner = plt.Circle((x, y), 1, color='green', fill=False, lw=1)
            ax.add_patch(corner)

def animate_players(pitch_width, pitch_length, tracks):
    player_ids = sorted(tracks.keys())
    colors = plt.cm.get_cmap('tab10', len(player_ids))

    # Przygotuj figury: boisko + panel statystyk
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])
    ax_stats.axis('off')

    draw_pitch(ax, pitch_width, pitch_length)
    ax.set_xlim(0, pitch_width)
    ax.set_ylim(0, pitch_length)
    ax.set_aspect(pitch_length / pitch_width)
    ax.set_xlabel('Szerokość boiska (m)')
    ax.set_ylabel('Długość boiska (m)')
    ax.set_title('Animacja ruchu zawodników na boisku')

    scatters = []
    texts = []
    for i, pid in enumerate(player_ids):
        pos = tracks[pid][0] if tracks[pid][0] is not None else (-10, -10, None)
        scat = ax.plot([pos[0]], [pos[1]], 'o', markersize=10, color=colors(i), label=f"ID: {pid}")[0]
        txt = ax.text(pos[0]+1, pos[1]+1, f"ID: {pid}", color=colors(i))
        scatters.append(scat)
        texts.append(txt)

    max_frames = get_max_frames(tracks)
    # Przygotuj tablicę na dystans i prędkość
    distances = {pid: 0.0 for pid in player_ids}
    prev_pos = {pid: None for pid in player_ids}
    prev_time = {pid: None for pid in player_ids}
    speeds = {pid: 0.0 for pid in player_ids}

    def update(frame):
        ax_stats.clear()
        ax_stats.axis('off')
        stats_lines = ["Statystyki zawodników:"]
        for i, pid in enumerate(player_ids):
            if frame < len(tracks[pid]) and tracks[pid][frame] is not None:
                x, y, t = tracks[pid][frame]
                scatters[i].set_data([x], [y])
                texts[i].set_position((x+1, y+1))
                # Dystans
                if prev_pos[pid] is not None:
                    dx = x - prev_pos[pid][0]
                    dy = y - prev_pos[pid][1]
                    dt = t - prev_time[pid] if (t is not None and prev_time[pid] is not None) else None
                    dist = np.sqrt(dx**2 + dy**2)
                    distances[pid] += dist
                    # Prędkość
                    if dt and dt > 0:
                        speeds[pid] = dist / dt
                prev_pos[pid] = (x, y)
                prev_time[pid] = t
            else:
                scatters[i].set_data([-10], [-10])
                texts[i].set_position((-10, -10))
                speeds[pid] = 0.0
            stats_lines.append(f"ID {pid}: {distances[pid]:.1f} m, {speeds[pid]:.2f} m/s")
        ax.set_title(f"Animacja ruchu zawodników na boisku (klatka {frame})")
        # Wyświetl statystyki
        ax_stats.text(0, 1, "\n".join(stats_lines), fontsize=12, va='top', family='monospace')
        return scatters + texts

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=30, blit=False, repeat=True)
    ax.legend()
    plt.tight_layout()
    plt.show()

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
    pitch_width, pitch_length = load_pitch_data(PITCH_FILE)
    tracks = load_players_tracks(PLAYERS_DIR)
    if not tracks:
        print("Brak danych zawodników!")
        exit(1)
    animate_players(pitch_width, pitch_length, tracks)
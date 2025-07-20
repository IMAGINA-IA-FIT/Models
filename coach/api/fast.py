import mediapipe as mp
import cv2
import json
import numpy as np
import gdown
import os
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware


import warnings
warnings.filterwarnings("ignore")



def log_msg(message, prop=None):
    log_message_debug = {
        "severity": "DEBUG",
        "message": message,
    }
    if prop is not None:
        if isinstance(prop, dict):
            log_message_debug["custom_property"] = json.dumps(prop)
        else:
            log_message_debug["custom_property"] = str(prop)
    print(json.dumps(log_message_debug))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Función para calcular ángulo entre tres puntos
def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radians * 180.0 / np.pi)
    if angulo > 180.0:
        angulo = 360 - angulo
    return angulo

# Obtener métricas usando MediaPipe Pose
def obtener_metricas(frame, frame_id):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = pose.process(rgb_frame)
    metricas = {}

    if resultados.pose_landmarks:
        landmarks = resultados.pose_landmarks.landmark

        # Coordenadas necesarias
        cadera_d = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        rodilla_d = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        tobillo_d = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

        hombro_d = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        hombro_i = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        cadera_i = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        rodilla_i = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        tobillo_i = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

        # Ángulos adicionales
        angulo_rodilla_derecha = calcular_angulo(cadera_d, rodilla_d, tobillo_d)
        angulo_rodilla_izquierda = calcular_angulo(cadera_i, rodilla_i, tobillo_i)
        angulo_espalda = calcular_angulo(hombro_d, cadera_d, rodilla_d)  # Inclinación torso
        angulo_cadera = calcular_angulo(hombro_d, cadera_d, tobillo_d)   # Alineación hombro-cadera-tobillo

        metricas = {
            "frame": frame_id,
            "angulo_rodilla_derecha": angulo_rodilla_derecha,
            "angulo_rodilla_izquierda": angulo_rodilla_izquierda,
            "angulo_espalda": angulo_espalda,
            "angulo_cadera": angulo_cadera
        }

    return metricas

# Detectar repeticiones simples por ángulo mínimo de rodilla (ejemplo rudimentario)
def detectar_repeticiones(data, umbral=90):
    repeticiones = []
    en_repeticion = False
    inicio = 0

    for i, frame in enumerate(data):
        angulo = min(frame['angulo_rodilla_derecha'], frame['angulo_rodilla_izquierda'])

        if angulo < umbral and not en_repeticion:
            inicio = frame['frame']
            en_repeticion = True
        elif angulo >= umbral and en_repeticion:
            fin = frame['frame']
            repeticiones.append((inicio, fin))
            en_repeticion = False

    return repeticiones

# Agrupar por fases y obtener resumen por fase
def agrupar_por_fases(data, repeticiones):
    resumen = {}
    for idx, (ini, fin) in enumerate(repeticiones):
        rango = fin - ini
        if rango <= 0:
            continue
        fases = {
            'fase1': [], 'fase2': [], 'fase3': [], 'fase4': []
        }
        for frame in data:
            if ini <= frame['frame'] <= fin:
                pos = frame['frame'] - ini
                if pos < 0.25 * rango:
                    fases['fase1'].append(frame)
                elif pos < 0.5 * rango:
                    fases['fase2'].append(frame)
                elif pos < 0.75 * rango:
                    fases['fase3'].append(frame)
                else:
                    fases['fase4'].append(frame)

        resumen[f'rep{idx+1}'] = {}
        for fase, frames in fases.items():
            if not frames:
                continue
            resumen[f'rep{idx+1}'][fase] = {
                'angulo_rodilla': {
                    'min': min(min(f['angulo_rodilla_derecha'], f['angulo_rodilla_izquierda']) for f in frames),
                    'max': max(max(f['angulo_rodilla_derecha'], f['angulo_rodilla_izquierda']) for f in frames)
                },
                'angulo_cadera': {
                    'min': min(f['angulo_cadera'] for f in frames),
                    'max': max(f['angulo_cadera'] for f in frames)
                },
                'angulo_espalda': {
                    'min': min(f['angulo_espalda'] for f in frames),
                    'max': max(f['angulo_espalda'] for f in frames)
                }
            }
    return resumen

def analizar_video_local(ruta):
    video = cv2.VideoCapture(ruta)
    resultados = []
    frame_id = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        metricas_frame = obtener_metricas(frame, frame_id)
        if metricas_frame:
            resultados.append(metricas_frame)

        frame_id += 1

    video.release()
    return resultados

# Descargar video desde Google Drive y analizarlo
def analizar_video_desde_drive(drive_url, output_json="resumen_por_reps.json"):
    # Descargar el video temporalmente
    video_temp_path = "temp_video.mp4"
    gdown.download(drive_url, video_temp_path, quiet=False)

    # Procesar el video
    resultados = analizar_video_local(video_temp_path)
    reps = detectar_repeticiones(resultados)
    resumen_reps = agrupar_por_fases(resultados, reps)

    # Guardar el resumen en un archivo JSON
    with open(output_json, "w") as f:
        json.dump(resumen_reps, f, indent=2)

    # Eliminar el archivo temporal
    os.remove(video_temp_path)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175"],  # ajusta según tu front
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def output_models():
    # Ejemplo de uso
    drive_link = "https://drive.google.com/uc?id=1tvT9i8Jo9zQ3a312NtVLRpZ0XkrAUPgR"  # Reemplaza YOUR_FILE_ID con el ID del archivo en Drive
    analizar_video_desde_drive(drive_link)
    return {"data": "Video analizado y resumen guardado en resumen_por_reps.json"}


@app.get("/status")
def status():
    return {"status": "cobranza model is up and running"}

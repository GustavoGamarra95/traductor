import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean
import os
import csv
from tensorflow.keras.models import load_model
import joblib


class DetectorMovimientosManos:
    def __init__(self, umbral_distancia=50, umbral_tiempo=30, modo_solo_procesar=False):
        # Configuración de MediaPipe para manos y pose
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Inicializar detectores
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Parámetros de memoria
        self.memoria_movimientos = {"izquierda": [], "derecha": []}
        self.umbral_distancia = umbral_distancia
        self.umbral_tiempo = umbral_tiempo

        # Modo solo procesar (sin cargar el modelo)
        self.modo_solo_procesar = modo_solo_procesar

        # Verificar si el archivo CSV existe
        self.csv_file_path = 'movimientos_manos.csv'
        if not os.path.exists(self.csv_file_path):
            print("El archivo CSV no existe. Generando un nuevo archivo...")
            self.csv_file = open(self.csv_file_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            # Escribir encabezados
            self.csv_writer.writerow([
                'Frame', 'Mano', 'Wrist_X', 'Wrist_Y',
                'Thumb_Tip_X', 'Thumb_Tip_Y',
                'Index_Tip_X', 'Index_Tip_Y',
                'Middle_Tip_X', 'Middle_Tip_Y',
                'Ring_Tip_X', 'Ring_Tip_Y',
                'Pinky_Tip_X', 'Pinky_Tip_Y',
                'Accion'
            ])
            self.csv_file.close()
        else:
            print("El archivo CSV ya existe.")

        # Cargar el modelo de Deep Learning y los objetos de preprocesamiento (si no está en modo solo procesar)
        if not self.modo_solo_procesar:
            self.model = load_model('modelo_movimientos_manos.h5')
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            self.onehot_encoder = joblib.load('onehot_encoder.pkl')

    def __del__(self):
        # Cerrar el archivo CSV al finalizar
        if hasattr(self, 'csv_file'):
            self.csv_file.close()

    def procesar_video(self, ruta_video):
        if not os.path.exists(ruta_video):
            print(f"Error: El archivo {ruta_video} no existe.")
            return

        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video en {ruta_video}.")
            return
        else:
            print(f"Video abierto correctamente. Resolución: {cap.get(3)}x{cap.get(4)}")

        # Abrir el archivo CSV en modo de escritura
        self.csv_file = open(self.csv_file_path, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        frame_actual = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break

            # Procesar frame para detección de manos y pose
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = self.hands.process(rgb_frame)
            results_pose = self.pose.process(rgb_frame)

            # Dibujar puntos de la pose
            if results_pose.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Procesar detección de manos
            if results_hands.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks,
                                                      results_hands.multi_handedness):
                    # Determinar si es mano izquierda o derecha
                    mano = "izquierda" if handedness.classification[0].label == "Left" else "derecha"

                    # Extraer puntos clave (incluyendo Frame y Mano_izquierda)
                    puntos_clave = self._extraer_puntos_clave(hand_landmarks, frame.shape, frame_actual, mano)

                    # Obtener la posición de la mano y los dedos
                    posicion_mano = self._obtener_posicion_mano(puntos_clave)
                    posicion_dedos = self._obtener_posicion_dedos(puntos_clave)

                    # Predecir la acción (si no está en modo solo procesar)
                    if not self.modo_solo_procesar:
                        accion = self._predecir_accion(puntos_clave)
                    else:
                        accion = "Sin acción"

                    # Guardar datos en el CSV
                    self._guardar_en_csv(frame_actual, mano, puntos_clave, accion)

                    # Mostrar texto en la pantalla
                    self._mostrar_texto(frame, mano, posicion_mano, posicion_dedos, accion)

            # Mostrar frame
            cv2.imshow('Detector de Movimientos y Pose', frame)

            # Salir con tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_actual += 1

        cap.release()
        cv2.destroyAllWindows()

    def _extraer_puntos_clave(self, hand_landmarks, shape_frame, frame_actual, mano):
        altura, anchura, _ = shape_frame
        puntos = []

        # Definir los puntos clave de interés
        puntos_interes = [
            self.mp_hands.HandLandmark.WRIST,
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]

        # Extraer coordenadas X e Y de cada punto clave
        for punto in puntos_interes:
            landmark = hand_landmarks.landmark[punto]
            x = int(landmark.x * anchura)
            y = int(landmark.y * altura)
            puntos.append(x)  # Agregar coordenada X
            puntos.append(y)  # Agregar coordenada Y

        # Agregar características adicionales: Frame y Mano_izquierda
        puntos.append(frame_actual)  # Frame
        puntos.append(1 if mano == "izquierda" else 0)  # Mano_izquierda (1 si es izquierda, 0 si es derecha)

        return puntos

    def _obtener_posicion_mano(self, puntos_clave):
        # Extraer solo las coordenadas X e Y de los puntos clave (ignorar Frame y Mano_izquierda)
        coordenadas = puntos_clave[:12]  # Las primeras 12 características son las coordenadas X e Y
        x_promedio = sum(coordenadas[i] for i in range(0, len(coordenadas), 2)) / (len(coordenadas) // 2)
        y_promedio = sum(coordenadas[i] for i in range(1, len(coordenadas), 2)) / (len(coordenadas) // 2)
        return (int(x_promedio), int(y_promedio))

    def _obtener_posicion_dedos(self, puntos_clave):
        # Extraer solo las coordenadas X e Y de los puntos clave (ignorar Frame y Mano_izquierda)
        coordenadas = puntos_clave[:12]  # Las primeras 12 características son las coordenadas X e Y

        dedos = {
            "Pulgar": (coordenadas[0], coordenadas[1]),  # Wrist
            "Índice": (coordenadas[2], coordenadas[3]),  # Thumb_Tip
            "Medio": (coordenadas[4], coordenadas[5]),   # Index_Tip
            "Anular": (coordenadas[6], coordenadas[7]),  # Middle_Tip
            "Meñique": (coordenadas[8], coordenadas[9])  # Ring_Tip
        }
        return dedos

    def _mostrar_texto(self, frame, mano, posicion_mano, posicion_dedos, accion):
        margen = 50
        y_offset = 50

        # Mostrar posición de la mano
        texto_mano = f"Mano {mano}: {posicion_mano}"
        if mano == "izquierda":
            cv2.putText(frame, texto_mano, (margen, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            ancho_texto = cv2.getTextSize(texto_mano, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
            x = frame.shape[1] - ancho_texto - margen
            cv2.putText(frame, texto_mano, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar posición de los dedos
        y_offset += 30
        for dedo, pos in posicion_dedos.items():
            texto_dedo = f"{dedo}: {pos}"
            if mano == "izquierda":
                cv2.putText(frame, texto_dedo, (margen, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                ancho_texto = cv2.getTextSize(texto_dedo, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
                x = frame.shape[1] - ancho_texto - margen
                cv2.putText(frame, texto_dedo, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            y_offset += 30

        # Mostrar acción predicha
        texto_accion = f"Acción {mano}: {accion}"
        if mano == "izquierda":
            cv2.putText(frame, texto_accion, (margen, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            ancho_texto = cv2.getTextSize(texto_accion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
            x = frame.shape[1] - ancho_texto - margen
            cv2.putText(frame, texto_accion, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    def _predecir_accion(self, puntos_clave):
        # Convertir a un array de numpy y asegurar que tenga la forma correcta
        puntos_clave = np.array(puntos_clave).reshape(1, -1)

        # Verificar que el número de características coincida con el esperado
        if puntos_clave.shape[1] != self.scaler.n_features_in_:
            raise ValueError(
                f"Se esperaban {self.scaler.n_features_in_} características, pero se obtuvieron {puntos_clave.shape[1]}."
            )

        # Normalizar los puntos clave
        puntos_clave = self.scaler.transform(puntos_clave)

        # Predecir la acción
        prediccion = self.model.predict(puntos_clave)
        accion = self.label_encoder.inverse_transform([np.argmax(prediccion)])
        return accion[0]

    def _guardar_en_csv(self, frame_actual, mano, puntos_clave, accion):
        # Aplanar la lista de puntos clave
        datos_fila = [frame_actual, mano]
        datos_fila.extend(puntos_clave)
        datos_fila.append(accion)


        self.csv_writer.writerow(datos_fila)


def main():

    modo_solo_procesar = True
    # modo_solo_procesar = True

    detector = DetectorMovimientosManos(modo_solo_procesar=modo_solo_procesar)

    ruta_video = r"videos/Profesionesnúmeroscoloresfamiliavocabulariosvarios.mp4"
    print(f"Intentando abrir el video en: {ruta_video}")

    detector.procesar_video(ruta_video)


if __name__ == "__main__":
    main()
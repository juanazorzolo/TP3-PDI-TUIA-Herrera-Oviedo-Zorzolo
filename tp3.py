import cv2
import numpy as np

# Función para detectar un área donde se tiran los dados basada en un rango de colores HSV
def detectar_area(frame, rango_inferior_hsv, rango_superior_hsv):
    # Convertir el frame de BGR a HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Crear una máscara binaria según el rango de colores HSV
    mascara = cv2.inRange(frame_hsv, rango_inferior_hsv, rango_superior_hsv)
    # Crear un kernel para operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Aplicar cierre para unir áreas cercanas
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    # Aplicar apertura para eliminar ruido
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    # Encontrar contornos en la máscara
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        # Obtener el contorno más grande
        contorno_mayor = max(contornos, key=cv2.contourArea)
        # Obtener el rectángulo delimitador del contorno
        x, y, ancho, alto = cv2.boundingRect(contorno_mayor)
        return (x, y, ancho, alto), mascara
    return None, mascara

# Función para detectar movimiento en una región de interés (ROI)
def detectar_movimiento(frame_anterior, frame_actual, roi):
    x, y, ancho, alto = roi
    # Recortar la región de interés de ambos frames
    roi_anterior = frame_anterior[y:y+alto, x:x+ancho]
    roi_actual = frame_actual[y:y+alto, x:x+ancho]
    # Calcular la diferencia absoluta entre las dos imágenes
    diferencia = cv2.absdiff(roi_anterior, roi_actual)
    # Convertir la diferencia a escala de grises
    diferencia_gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)
    # Aplicar un umbral binario para destacar las diferencias
    _, umbral = cv2.threshold(diferencia_gris, 25, 255, cv2.THRESH_BINARY)
    # Calcular el nivel de movimiento como la cantidad de píxeles no cero
    nivel_movimiento = cv2.countNonZero(umbral)
    return nivel_movimiento, umbral

# Ruta del video a analizar
ruta_video = "videos-dados/tirada_4.mp4"
cap = cv2.VideoCapture(ruta_video)

# Obtener las dimensiones del video
ancho_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir el rango de colores HSV para segmentar el área deseada
rango_inferior_hsv = np.array([35, 50, 80])
rango_superior_hsv = np.array([85, 255, 255])

frame_anterior = None  # Almacena el frame anterior para comparar movimiento
frame_guardado = False  # Bandera para guardar solo un frame por video

while cap.isOpened():
    ret, frame_actual = cap.read()
    if not ret:
        break

    # Detectar área basada en el rango de colores
    area, mascara = detectar_area(frame_actual, rango_inferior_hsv, rango_superior_hsv)

    if area:
        x, y, ancho, alto = area
        # Dibujar un rectángulo alrededor del área detectada
        cv2.rectangle(frame_actual, (x, y), (x+ancho, y+alto), (255, 0, 0), 2)

        if frame_anterior is not None:
            # Detectar movimiento dentro del área seleccionada
            nivel_movimiento, umbral = detectar_movimiento(frame_anterior, frame_actual, area)

            if nivel_movimiento < 1000 and not frame_guardado:  # Ajustar el umbral si es necesario
                cv2.putText(frame_actual, "Dados quietos", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # Guardar el área de interés (ROI) si los dados están quietos
                roi_frame = frame_actual[y:y+alto, x:x+ancho]
                ruta_frame = "dado_quieto_frame.png"
                cv2.imwrite(ruta_frame, roi_frame)
                print(f"Frame guardado en {ruta_frame}")
                frame_guardado = True

                # Redimensionar y mostrar el ROI
                roi_redimensionado = cv2.resize(roi_frame, (int(ancho/3), int(alto/3)))
                cv2.imshow("ROI - Dados quietos", roi_redimensionado)

            elif nivel_movimiento >= 1000:
                cv2.putText(frame_actual, "Dados en movimiento", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Mostrar la máscara de movimiento
            umbral_mostrar = cv2.resize(umbral, (int(ancho_video/3), int(alto_video/3)))
            cv2.imshow("Movimiento", umbral_mostrar)

    # Actualizar el frame anterior
    frame_anterior = frame_actual.copy()

    # Redimensionar y mostrar el frame y la máscara
    frame_mostrar = cv2.resize(frame_actual, (int(ancho_video/3), int(alto_video/3)))
    mascara_mostrar = cv2.resize(mascara, (int(ancho_video/3), int(alto_video/3)))

    cv2.imshow("Frame", frame_mostrar)
    cv2.imshow("Mascara", mascara_mostrar)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

"""
FUNCIONES DE UTILIDAD
"""
# Mostrar imagenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=True, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        if blocking:
            plt.show(block=True)

# Eliminar carpetas y su contenido para reiniciar la ejecucion
def eliminar_carpetas_contenido(carpeta1, carpeta2, carpeta3):
    carpetas = [carpeta1, carpeta2, carpeta3]
    for carpeta in carpetas:
        if os.path.exists(carpeta):
            shutil.rmtree(carpeta)
            print(f"Carpeta '{carpeta}' y su contenido han sido eliminados.")
        else:
            print(f"La carpeta '{carpeta}' no existe.")


"""
CREAR CARPETAS NECESARIAS
"""
eliminar_carpetas_contenido('capturas_dados', 'dados_detectados', 'videos_finales')
os.makedirs("capturas_dados", exist_ok=True) # imagenes de los dados quietos
os.makedirs("dados_detectados", exist_ok=True) # imagenes con los dados y números detectados
carpeta_salida = "videos_finales"
os.makedirs(carpeta_salida, exist_ok=True) # videos finales


"""
OBTENER FRAME CON LOS DADOS QUIETOS
"""
# Detecta el área donde se tiran los dados basada en un rango de colores HSV
def detectar_area(frame, rango_inferior_hsv, rango_superior_hsv):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(frame_hsv, rango_inferior_hsv, rango_superior_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        contorno_mayor = max(contornos, key=cv2.contourArea)
        x, y, ancho, alto = cv2.boundingRect(contorno_mayor)
        return (x, y, ancho, alto), mascara
    return None, mascara

# Detecta movimiento en una región de interés (ROI)
def detectar_movimiento(frame_anterior, frame_actual, roi):
    x, y, ancho, alto = roi
    roi_anterior = frame_anterior[y:y+alto, x:x+ancho]
    roi_actual = frame_actual[y:y+alto, x:x+ancho]
    diferencia = cv2.absdiff(roi_anterior, roi_actual)
    diferencia_gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(diferencia_gris, 25, 255, cv2.THRESH_BINARY)
    nivel_movimiento = cv2.countNonZero(umbral)
    return nivel_movimiento, umbral

archivos_videos = [f for f in os.listdir("videos-dados") if f.endswith(('.mp4'))]
indices_frames_guardados = []  # Almacenar el índice del frame guardado por video
for archivo_video in archivos_videos:
    ruta_video = os.path.join("videos-dados", archivo_video)
    cap = cv2.VideoCapture(ruta_video)
    
    # Dimensiones del video
    ancho_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Parámetros iniciales
    rango_inferior_hsv = np.array([35, 50, 80])
    rango_superior_hsv = np.array([85, 255, 255])
    frame_anterior = None
    frame_guardado = False
    calibracion_movimiento = []  # Niveles de movimiento registrados en el periodo de calibración
    contador_frames = 0  # Contador de frames procesados

    while cap.isOpened():
        ret, frame_actual = cap.read()
        if not ret:
            break

        contador_frames += 1
        area, mascara = detectar_area(frame_actual, rango_inferior_hsv, rango_superior_hsv)
        if area:
            x, y, ancho, alto = area
            cv2.rectangle(frame_actual, (x, y), (x + ancho, y + alto), (255, 0, 0), 2)

            if frame_anterior is not None:
                nivel_movimiento, umbral = detectar_movimiento(frame_anterior, frame_actual, area)

                # Periodo de calibración inicial
                if len(calibracion_movimiento) < 60:  # Frames iniciales para calibrar el movimiento
                    calibracion_movimiento.append(nivel_movimiento)
                    cv2.putText(frame_actual, "Calibrando...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    umbral_movimiento_calibrado = max(1000, np.mean(calibracion_movimiento) * 2)

                    if nivel_movimiento < 1000 and not frame_guardado:
                        cv2.putText(frame_actual, "Dados quietos", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Guardar la imagen con el tamaño original del video
                        nombre_guardado = f"{os.path.splitext(archivo_video)[0]}_imagen.png"
                        ruta_frame = os.path.join("capturas_dados", nombre_guardado)

                        # Crear una imagen con el tamaño del video original
                        frame_completo = frame_actual.copy()
                        cv2.imwrite(ruta_frame, frame_completo)
                        print(f"Frame guardado en {ruta_frame} (Índice del frame: {contador_frames})")
                        
                        # Registrar el índice del frame guardado
                        indices_frames_guardados.append(contador_frames)
                        frame_guardado = True

                        roi_frame = frame_actual[y:y+alto, x:x+ancho]
                        roi_redimensionado = cv2.resize(roi_frame, (int(ancho / 3), int(alto / 3)))
                        cv2.imshow("ROI - Dados quietos", roi_redimensionado)

                # Mostrar el progreso en tiempo real
                umbral_mostrar = cv2.resize(umbral, (int(ancho_video / 3), int(alto_video / 3)))
                cv2.imshow("Movimiento", umbral_mostrar)

        frame_anterior = frame_actual.copy()
        frame_mostrar = cv2.resize(frame_actual, (int(ancho_video / 3), int(alto_video / 3)))
        mascara_mostrar = cv2.resize(mascara, (int(ancho_video / 3), int(alto_video / 3)))
        cv2.imshow("Frame", frame_mostrar)
        cv2.imshow("Mascara", mascara_mostrar)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()


"""DETECTAR DADOS Y NÚMEROS DE LOS MISMOS"""
# Aplica filtros de limpieza
def procesar_dado(imagen, t1, t2, pk):
    imagen = imagen[y:y+alto, x:x+ancho]
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Desenfoque para reducir ruido
    img_desenfoque = cv2.GaussianBlur(img_gris, (3, 3), 0)
    imshow(img_desenfoque, title='Desenfoque Gaussiano')

    # Detectar bordes con Canny
    img_bordes = cv2.Canny(img_desenfoque, t1, t2)
    imshow(img_bordes, title='Bordes con Canny')

    # Operaciones morfologicas para limpiar bordes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pk, pk))
    img_dilatada = cv2.dilate(img_bordes, k, iterations=7)
    img_limpia = cv2.erode(img_dilatada, k, iterations=3)
    imshow(img_limpia, title='Operaciones Morfológicas')

    # Umbralización y detección de contornos
    _, thresh_otsu = cv2.threshold(img_limpia, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar solo contornos circulares
    contornos_circulares = []
    area_min = 100
    area_max = 500

    for cnt in contours_otsu:
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        if perimetro > 0:  # Evitar divisiones por cero
            circularidad = (4 * np.pi * area) / (perimetro ** 2)
            # Validar contornos
            if circularidad >= 0.7 and area_min <= area <= area_max:
                contornos_circulares.append(cnt)

    # Visualizar
    img_contornos_circulares = imagen.copy()
    cv2.drawContours(img_contornos_circulares, contornos_circulares, -1, (0, 255, 0), 2)
    imshow(img_contornos_circulares, title="Contornos Circulares Detectados")
    return contornos_circulares, imagen

# Detecta nro de cada dado
def agrupar_contornos_por_cercania(contornos, imagen, umbral_distancia):
    imagen = imagen[y:y+alto, x:x+ancho]
    img_proceso = imagen.copy()

    # Calcular centroides de cada contorno
    centroides = []
    for cnt in contornos:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroides.append((cx, cy))
            cv2.circle(img_proceso, (cx, cy), 5, (0, 255, 0), -1)  # Dibujar centroides
        else:
            centroides.append(None)
    imshow(img_proceso, title="Centroides calculados")

    # Agrupar centroides por cercanía
    grupos = []
    visitados = set()

    for i, centro1 in enumerate(centroides):
        if centro1 is None or i in visitados:
            continue

        grupo_actual = [contornos[i]]
        visitados.add(i)

        # Dibujar contorno inicial del grupo
        img_grupo = img_proceso.copy()
        cv2.drawContours(img_grupo, [contornos[i]], -1, (255, 0, 0), 2)

        for j, centro2 in enumerate(centroides):
            if j != i and centro2 is not None and j not in visitados:
                distancia = np.linalg.norm(np.array(centro1) - np.array(centro2))
                if distancia < umbral_distancia:
                    grupo_actual.append(contornos[j])
                    visitados.add(j)
                    # Dibujar linea entre los centroides conectados
                    cv2.line(img_grupo, centro1, centro2, (0, 0, 255), 1)
        
        grupos.append(grupo_actual)

    # Visualizar grupos finales
    img_final = imagen.copy()
    colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # Asignar colores
    for idx, grupo in enumerate(grupos):
        color = colores[idx % len(colores)]
        for cnt in grupo:
            cv2.drawContours(img_final, [cnt], -1, color, 2)
    
    imshow(img_final, title="Grupos finales")
    return grupos

# Dibuja las bounding box en los dados con su respectivo número
def marcar_dados_y_nros(imagen, grupos_contornos, tamano_bbox=100):
    img_marcada = imagen.copy()
    for idx, grupo in enumerate(grupos_contornos):
        if len(grupo) == 0:  # Ignorar grupos vacíos
            continue
        
        # Calcular el centro del grupo basado en sus centroides
        todos_puntos = np.vstack(grupo)  # Combina todos los puntos de los contornos en el grupo
        M = cv2.moments(todos_puntos)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            continue

        # Definir bounding box alrededor del centroide
        mitad_bbox = tamano_bbox // 2
        x1, y1 = cx - mitad_bbox, cy - mitad_bbox
        x2, y2 = cx + mitad_bbox, cy + mitad_bbox

        # Dibujar bounding box
        cv2.rectangle(img_marcada, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Poner el número del círculo dentro del dado
        texto = str(len(grupo))
        cv2.putText(img_marcada, texto, (cx, cy - mitad_bbox - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3, cv2.LINE_AA )

    imagen_recortada = img_marcada[y:y+alto, x:x+ancho]
    imshow(imagen_recortada, title="Bounding Boxes con el Número de Cada Dado")
    return img_marcada

archivos_dados = [f for f in os.listdir("capturas_dados") if f.lower().endswith('.png')]
for imagen in archivos_dados:
    ruta_imagen = os.path.join("capturas_dados", imagen)
    imagen_dado = cv2.imread(ruta_imagen)

    # Obtener números de los dados
    nro_dado, img_dados_nros = procesar_dado(imagen_dado, 280, 460, 2)
    grupos_contornos = agrupar_contornos_por_cercania(nro_dado, imagen_dado, 50)

    # Visualizar bounding box de los dados con sus números
    imagen_marcada = marcar_dados_y_nros(imagen_dado, grupos_contornos, tamano_bbox=120)
    nombre = f"{os.path.splitext(imagen)[0]}_dados_detectados.png"
    ruta = os.path.join("dados_detectados", nombre)
    cv2.imwrite(ruta, imagen_marcada)


"""GENERAR VIDEOS"""
archivos_videos = ["tirada_1.mp4", "tirada_2.mp4", "tirada_3.mp4", "tirada_4.mp4"]
carpeta_dados_detectados = "dados_detectados"
imagenes_reemplazo = sorted(os.listdir(carpeta_dados_detectados))

for i, archivo_video in enumerate(archivos_videos):
    ruta_video = os.path.join("videos-dados", archivo_video)
    cap = cv2.VideoCapture(ruta_video)
    
    if not cap.isOpened():
        print(f"No se pudo abrir el video {archivo_video}")
        continue

    # Dimensiones y FPS del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Crear video de salida
    nombre_salida = os.path.join(carpeta_salida, f"{os.path.splitext(archivo_video)[0]}_modificado.mp4")
    out = cv2.VideoWriter(nombre_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    ruta_imagen = os.path.join(carpeta_dados_detectados, imagenes_reemplazo[i])
    imagen_reemplazo = cv2.imread(ruta_imagen)
    if imagen_reemplazo is None:
        print(f"No se encontró la imagen de reemplazo {ruta_imagen}")
        cap.release()
        out.release()
        continue

    # Verificar si la imagen necesita ser rotada
    if imagen_reemplazo.shape[1] != width or imagen_reemplazo.shape[0] != height:
        imagen_reemplazo = cv2.rotate(imagen_reemplazo, cv2.ROTATE_90_CLOCKWISE)

    # Redimensionar la imagen de reemplazo al tamaño del frame
    imagen_reemplazo = cv2.resize(imagen_reemplazo, (width, height))

    # Reemplazar el frame correspondiente y los 15 siguientes
    indice_reemplazo = indices_frames_guardados[i]
    contador_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Si el frame está en el rango de reemplazo, usar la imagen de reemplazo
        if indice_reemplazo <= contador_frames < indice_reemplazo + 15:
            out.write(imagen_reemplazo)
        else:
            out.write(frame)
        
        contador_frames += 1

    cap.release()
    out.release()
    print(f"Video procesado: {nombre_salida}")

cv2.destroyAllWindows()


"""VER VIDEOS FINALES"""
videos = [f for f in os.listdir(carpeta_salida) if f.endswith('.mp4')]
for video_nombre in videos:
    ruta_video = os.path.join(carpeta_salida, video_nombre)
    cap = cv2.VideoCapture(ruta_video)

    print(f"Reproduciendo video: {video_nombre}")
    
    # Reproducir el video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin del video")
            break

        # Redimensionar el frame
        roi_frame = frame[y:y+alto, x:x+ancho]
        frame = cv2.resize(roi_frame, dsize=(int(ancho/3), int(alto/3)))

        # Visualizar
        cv2.imshow(f'Video {video_nombre}', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()

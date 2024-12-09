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
def eliminar_carpetas_contenido(carpeta1, carpeta2, carpeta3, carpeta4):
    carpetas = [carpeta1, carpeta2, carpeta3, carpeta4]
    for carpeta in carpetas:
        if os.path.exists(carpeta):
            shutil.rmtree(carpeta)
            print(f"Carpeta '{carpeta}' y su contenido han sido eliminados.")
        else:
            print(f"La carpeta '{carpeta}' no existe.")


"""
CREAR CARPETAS NECESARIAS
"""
eliminar_carpetas_contenido('capturas_dados', 'dados_detectados', 'dados', 'dados_con_nros')
os.makedirs("capturas_dados", exist_ok=True) #imagen de los dados quietos
os.makedirs("dados_detectados", exist_ok=True) #imagen con los dados detectados
os.makedirs("dados", exist_ok=True) #cada dado detectado (cambia en base a la ejecucion). capaz sirve para detectar el nro de cada dado
os.makedirs("dados_con_nros", exist_ok=True) #imagen con los circulos detectados de cada dado

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
indices_guardados = []  # Lista para guardar los índices de los frames donde se detectó quietud
for archivo_video in archivos_videos:
    # Leer video
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
   
    while cap.isOpened():
        ret, frame_actual = cap.read()
        if not ret:
            break

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

                        # Crear una imagen completa con el tamaño del video original
                        frame_completo = frame_actual.copy()
                        cv2.imwrite(ruta_frame, frame_completo)
                        print(f"Frame guardado en {ruta_frame}")
                        frame_guardado = True

                        roi_redimensionado = cv2.resize(frame_completo, (int(ancho / 3), int(alto / 3)))
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


"""DETECTAR DADOS Y NROS"""

# Filtros
def procesar_dado(imagen, t1, t2, pk):
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Desenfoque gaussiano para reducir ruido
    img_desenfoque = cv2.GaussianBlur(img_gris, (3, 3), 0)
    imshow(img_desenfoque, title='desenfoque')

    # Detectar bordes con Canny
    img_bordes = cv2.Canny(img_desenfoque, t1, t2)
    imshow(img_bordes, title='canny')

    # Operaciones morfologicas para limpiar bordes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pk, pk))
    img_dilatada = cv2.dilate(img_bordes, k, iterations=7)
    img_limpia = cv2.erode(img_dilatada, k, iterations=3)
    imshow(img_limpia, title='Operaciones morfolÃ³gicas')

    # Umbralizacion y deteccion de contornos
    _, thresh_otsu = cv2.threshold(img_limpia, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar solo contornos circulares con tamaño especi­fico
    contornos_circulares = []
    area_min = 100  # Ajustar segun tus necesidades
    area_max = 500  # Ajustar segun tus necesidades

    for cnt in contours_otsu:
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        if perimetro > 0:  # Evitar divisiones por cero
            circularidad = (4 * np.pi * area) / (perimetro ** 2)
            # Verificar que el contorno sea circular y este dentro del rango de area permitido
            if circularidad >= 0.7 and area_min <= area <= area_max:
                contornos_circulares.append(cnt)

    # Visualizar contornos circulares en la imagen
    img_contornos_circulares = imagen.copy()
    cv2.drawContours(img_contornos_circulares, contornos_circulares, -1, (0, 255, 0), 2)
    imshow(img_contornos_circulares, title="Contornos Circulares")

    return contornos_circulares, imagen

def agrupar_contornos_por_cercania_visual(contornos, imagen, umbral_distancia):

    # Crear copia de la imagen para dibujar
    img_proceso = imagen.copy()
    
    # Calcular los centroides de cada contorno
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

    # Agrupar centroides por cercani­a
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
                    # Dibujar li­nea entre los centroides conectados
                    cv2.line(img_grupo, centro1, centro2, (0, 0, 255), 1)
        
        grupos.append(grupo_actual)

    # Visualizar los grupos finales
    img_final = imagen.copy()
    colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for idx, grupo in enumerate(grupos):
        color = colores[idx % len(colores)]  # Asignar colores ci­clicamente
        for cnt in grupo:
            cv2.drawContours(img_final, [cnt], -1, color, 2)
    
    imshow(img_final, title="Grupos finales")
    return grupos

def marcar_dados_con_bounding_boxes_uniformes(imagen, grupos_contornos, tamano_bbox=100):
    """
    Dibuja bounding boxes uniformes centradas en los dados y muestra el numero de c­irculos detectados.
    Args:
        imagen: Imagen original en la que se dibujarÃ¡n los bounding boxes.
        grupos_contornos: Lista de grupos de contornos agrupados por cercani­a.
        tamano_bbox: TamaÃ±o fijo (en pi­xeles) para las bounding boxes.
    """
    img_marcada = imagen.copy()

    for idx, grupo in enumerate(grupos_contornos):
        if len(grupo) == 0:  # Ignorar grupos vacÃ­os
            continue
        
        # Calcular el centro del grupo basado en sus centroides
        todos_puntos = np.vstack(grupo)  # Combina todos los puntos de los contornos en el grupo
        M = cv2.moments(todos_puntos)  # Momentos del grupo
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # Coordenada X del centroide
            cy = int(M["m01"] / M["m00"])  # Coordenada Y del centroide
        else:
            continue  # Saltar si no se puede calcular el centroide

        # Definir la bounding box uniforme alrededor del centroide
        mitad_bbox = tamano_bbox // 2
        x1, y1 = cx - mitad_bbox, cy - mitad_bbox
        x2, y2 = cx + mitad_bbox, cy + mitad_bbox

        # Dibujar la bounding box
        cv2.rectangle(img_marcada, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Poner el numero de ci­rculos dentro del dado
        texto = str(len(grupo))  # Numero de ci­rculos detectados
        cv2.putText(img_marcada, texto, (cx, cy - mitad_bbox - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 3, cv2.LINE_AA )
    
    imshow(img_marcada, title="Bounding Boxes Uniformes con NÃºmeros")
    return img_marcada

archivos_dados = [f for f in os.listdir("capturas_dados") if f.lower().endswith('.png')]
# Para cada imagen procesada
# Procesar cada imagen
for imagen in archivos_dados:
    try:
        # Leer imagen
        ruta_imagen = os.path.join("capturas_dados", imagen)
        imagen_dado = cv2.imread(ruta_imagen)

        # Detectar NROS
        nro_dado, img_dados_nros = procesar_dado(imagen_dado, 280, 460, 2)

        if not nro_dado:
            print(f"No se detectaron contornos circulares en {imagen}")
            continue

        # Agrupar contornos por cercani­a
        umbral_distancia = 50  # Ajusta segun la escala de la imagen
        grupos_contornos = agrupar_contornos_por_cercania_visual(nro_dado, imagen_dado, umbral_distancia)

        # Marcar dados con bounding box uniforme y numeros
        imagen_marcada = marcar_dados_con_bounding_boxes_uniformes(imagen_dado, grupos_contornos, tamano_bbox=120)
        nombre = f"{os.path.splitext(imagen)[0]}_dados_detectados.png"
        ruta = os.path.join("dados_detectados", nombre)
        cv2.imwrite(ruta, imagen_marcada)

    except Exception as e:
        print(f"Error al procesar {imagen}: {e}")
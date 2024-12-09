# Detección y Reconocimiento de Dados en Secuencias de Video

Este trabajo tiene como objetivo desarrollar un algoritmo para detectar y leer automáticamente los números obtenidos en dados en reposo, a partir de secuencias de video.
A continuación, se detallan las tareas realizadas y cómo ejecutar el código.

## Descripción del Problema
En este ejercicio, se proporcionan cuatro secuencias de video (tirada_1.mp4, tirada_2.mp4, tirada_3.mp4, tirada_4.mp4), cada una mostrando una tirada de 5 dados. 
El objetivo es:

- Detección automática de los dados en reposo: el algoritmo debe identificar el momento en el que los dados se detienen y leer el número en cada uno.
- Generación de videos con los dados resaltados: una vez detectados los dados, se deben agregar cuadros de delimitación (bounding boxes) de color azul alrededor de los dados, y colocar el número correspondiente sobre ellos.
  
## Estructura del trabajo
El código está organizado de la siguiente manera:


- capturas_dados/        # Carpeta para guardar las imágenes con los dados quietos
- videos-dados/          # Carpeta con los videos de entrada
- dados_detectados/      # Carpeta para guardar las imágenes con los dados y números detectados
- videos_finales/        # Carpeta con los videos de salida

## Dependencias
Este proyecto requiere las siguientes librerías de Python:

- opencv-python (cv2)
- numpy
- matplotlib

### Puedes instalar todas las dependencias necesarias utilizando pip:

pip install opencv-python numpy matplotlib

## Descripción del Código
El código se divide en varias secciones clave:

1. Funciones de Utilidad: 
- imshow(): Muestra imágenes en ventanas con opciones de color, título y barra de color.
- eliminar_carpetas_contenido(): Elimina las carpetas de trabajo previas para reiniciar la ejecución.
  
2. Creación de Carpeta: 
- Se crea una carpeta capturas_dados para almacenar las imágenes procesadas y los videos resultantes. 

3. Detección de Dados en Reposo
- Función detectar_area(): Detecta el área de los dados en cada cuadro del video utilizando un rango de colores en el espacio HSV.
- Función detectar_movimiento(): Detecta la quietud de los dados al comparar los frames anteriores con los actuales. Se utiliza un umbral para determinar si los dados están quietos.
  
4. Procesamiento de Dados y Detección de Números
- Función procesar_dado(): Preprocesa la imagen de los dados (conversión a escala de grises, detección de bordes, operaciones morfológicas) y detecta contornos circulares que corresponden a los dados.
- Función agrupar_contornos_por_cercania(): Agrupa los contornos cercanos para identificar los dados individuales.
- Función marcar_dados_y_nros(): Dibuja las bounding boxes en los dados con sus respectivos números.

5. Generación de Videos con Bounding Boxes: 
- Los videos resultantes se guardan con los dados detectados y numerados, agregando cuadros de delimitación azul y los números sobre los dados.

## Instrucciones para Ejecutar el Código
- Coloca los videos en la carpeta videos-dados/.

- Ejecuta el script para procesar los videos

- Los resultados de las imágenes de los dados quietos se guardarán en capturas_dados/.

## Resultados Esperados
Al ejecutar el código, se procesarán los videos y se generarán los siguientes resultados:

- Detección de los dados en reposo: Cuando los dados se detienen, el sistema detectará su quietud y mostrará un mensaje indicando que los dados están quietos.
- Videos con bounding boxes y números: Se generarán videos donde los dados se destacan con un cuadro de delimitación azul, y encima de cada dado aparecerá el número reconocido.
  
## Conclusiones
Este proyecto permite detectar y reconocer automáticamente los números en dados a partir de secuencias de video. 
El algoritmo detecta los dados en reposo y resalta los números obtenidos.

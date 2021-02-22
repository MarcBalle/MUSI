## EJERCICIO 3 - EYE TOY ##

## Marc Balle Sánchez ##

## Se desarrolla un programa que, empleando la cámara integrada en el ordenador, permite la interacción
## con una pelota flotante en la pantalla a través del movimiento corporal.

## Cada vez que la pelota sea golpeada, esta cambiará de color y se desplazará hacia abajo.

## Para considerar que la pelota ha sido golpeada, el movimiento de golpeo debe poseer una magnitud mayor
## a un determinado umbral. De esta forma se evita que cualquier movimiento ínfimo interaccione con la
## pelota.

## Cuando la pelota golpee alguno de los límites de la pantalla, esta aparecerá en algún punto aleatorio
## y se moverá con una dirección y velocidad aleatoria.

## Se recomienda no hacer movimientos bruscos e interaccionar principalmente con las manos.

## Pulsar la tecla 'q' para finalizar

import time
import numpy as np
import cv2 as cv

## movimiento de la pelota con la ecuación de movimiento rectilíneo uniforme (MRU)
def movement (x,y,vx,vy,t,shape):
    if x >= shape[0] or x <= 0 or y >= shape[1] or y<=0: #si toca un límite
        return np.random.randint(0,shape[0]+1), np.random.randint(0,shape[1]+1), np.random.randint(-20,20), np.random.randint(-20,20) # x, y, vx, vy
    x_new = x + vx*t #MRU
    y_new = y + vy*t
    return int(x_new), int(y_new), vx, vy

#def event(color):
#    color = np.random.randint(0, 255, (1, 3))
#    return tuple(map(tuple, color))[0]

# función que maneja la interacción con la pelota
def interaction (diff, LK_p, cen, thresh = 20):

    #diff = vector movimiento del cuerpo respecto la malla estática
    # LK_p = puntos devueltos por calcOpticalFlowPyrLK
    # cen = centro del círculo
    # thresh = umbral mínimo para considerar un movimiento como significativo.

    diff = diff.reshape((diff.shape[0], 2))
    diff_norm = np.linalg.norm(diff, axis = 1) # norma del vector movimiento
    filt = np.where(diff_norm > thresh, True, False) # solo movimiento con norma o magnitud mayor a thresh
    LK_p_filt = LK_p.reshape((LK_p.shape[0],2))[filt] #filtramos los movimientos

    return (LK_p_filt.astype(int) == cen).any() # si algún movimiento coincide con el centro de la pelota, return True

# LK parameters
lk_params = dict( winSize  = (25,25), # tamaño adecuado al problema.
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv.VideoCapture(0) # se abre al cámara
ret, old_frame = cap.read() # se lee el primer frame
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY) # se pasa a escala de grises
height, width = old_gray.shape # se guardan las dimensiones del frame
grid = np.dstack(np.meshgrid(np.arange(width, step = 20), np.arange(height, step = 20))).reshape((-1,1,2)) #se crea la malla sobre la que se calcula el flujo
color = np.random.randint(0,255,(grid.shape[0] + 50,3)) #colores para la malla (en caso de querer visualizarla)

# parámetros iniciales de la pelota
x = int(width/2) #centro de la pantalla
y = int(height/2)
vx = 20
vy = 15
color_ball = (255,0,0)

while(True):
    start = time.time() # necesario para calcular un delta del tiempo, para el movimiento de la pelota según MRU
    ret, frame = cap.read() # se lee el siguiente frame
    mask = np.zeros_like(frame)  # se renueva la máscara cada vez (con fines de representar el flujo, si se desea)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # frame a escala de grises
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, grid.astype('float32'), None, **lk_params) # se calcula el flujo
    ##############################
    # Si se quiere dibujar las lineas de flujo sobre la malla, descomentar esta parte y cambiar frame por img en cv.circle y cv.imshow
    #for i, (new, old) in enumerate(zip(p1, grid)):
    #    a, b = new.ravel()
    #    c, d = old.ravel()
    #    mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)

    #img = cv.add(frame, mask)
    ##############################
    cv.circle(frame, center = (x,y), radius = 40, color = color_ball, thickness= -1) # se dibuja la pelota

    if interaction(p1-grid, p1, np.array([x,y]), 20): #se analiza si hay interacción
        # cambio de color de la pelota
        color_ball = np.random.randint(0, 255, (3,))
        color_ball = (int(color_ball [0]), int(color_ball [1]), int(color_ball [2]))
        color_ball = tuple(color_ball)
        #cambio de posición de la pelota
        x = x +50
        y = y +50

    # Se actualiza el frame anterior al actual
    old_gray = frame_gray.copy()

    # Se muestra por pantalla el frame
    cv.imshow('frame',frame)
    if cv.waitKey(30) & 0xFF == ord('q'): # finaliza
        break
    end = time.time() # necesario para el cálculo del delta del tiempo

    # se actualiza la posición y velocidad de la pelota
    x, y, vx, vy = movement (x,y,vx,vy,start-end, (width, height)) # start - end = delta del tiempo

cap.release()
cv.destroyAllWindows()
import cv2
import numpy as np
import pyautogui

# Imagem e método utilizados para identificação da pupila (parâmetros do cv2.matchTemplate)
template = cv2.imread('img/pupil.jpeg',0)       #Imagem base presente na pasta img
w1, h1 = template.shape[::-1]
method = eval('cv2.TM_CCORR_NORMED')            #


# Classificadores utilizados para a detecção da face e do olho
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# Inicialização de alguns parâmetros necessários
screenWidth, screenHeight = pyautogui.size()    # x e y max da tela
x_t,y_t,w_t,h_t = 0,0,0,0                       # Rastrado
x_f,y_f,w_f,h_f = 0,0,0,0                       # Rosto
x_e,y_e,w_e,h_e = 0,0,0,0                       # Olho
x_ds, x_es, x_di, x_ei = 0,0,0,0                # Mapeamento do x
y_ds, y_es, y_di, y_ei = 0,0,0,0                # Mapeamneto do y
count, estado, counti = 0,0,0
tempo = 20
# Habilita a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Inicializando os parâmetros de mapeamento
while count < tempo * 4.5:
    # Captura do frame
    ok, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # Definição da nova região de interesse em escala de cinza e colorida (região da face)
        roi_gray = gray[y:y+int(h/2), x:x+int(w/2)]
        roi_color = img[y:y+int(h/2), x:x+int(w/2)]
        x_f,y_f,w_f,h_f = x,y,w,h
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.35, 5)

        for (ex,ey,ew,eh) in eyes:
            # Definição da nova região de interesse em escala de cinza e colorida (região dos olhos)
            roi_gray2 = roi_gray[int(ey):int(ey+eh), int(ex):int(ex+ew)]
            roi_color2 = roi_color[ey:ey+eh, ex:ex+ew]
            x_e,y_e,w_e,h_e = ex,ey,ew,eh
        # Desenhando o retângulo ao redor do olho detectado por último
        cv2.rectangle(roi_color,(x_e,y_e),(x_e+w_e,y_e+h_e),(0,255,0),2)

        # Percorrendo a imagem roi_gray2 comparando com a imagem template (w1xh1)
        # O melhor resultado é encontrada através do máx global de res (cv2.minMaxLoc), encontrando a sua posição (x,y)
        res = cv2.matchTemplate(roi_gray2,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w1, top_left[1] + h1)

        # Desenhando o retângulo ao redor da pupila com um ponto vermelho no centro
        cv2.rectangle(roi_color2,top_left, bottom_right, (255,255,255), 2)
        cv2.circle(roi_color2,(int(top_left[0]+w1/2),int(top_left[1]+h1/2)), 4, (0,0,255), -1)
        x_t,y_t,w_t,h_t = top_left[0]+x_e+x_f,top_left[1]+y_e+y_f,w1,h1

    # Demora um pouco para detectar a pupila para o mapeamento
    if estado == 0:
        print("Preparando")
        count+=1
        if count == tempo/2:
            estado+=1
    # Mouse vai para o canto superior esquerdo da tela e a pessoa tem que olhar para ele
    elif estado == 1:
        if counti == 0:
            print("Fique olhando para o canto superior esquerdo da tela!")
        x_es = x_t
        y_es = y_t
        count+=1
        counti+=1
        if count == 1.5 * tempo:
            estado+=1
            counti = 0
    # Mouse vai para o canto inferior esquerdo da tela e a pessoa tem que olhar para ele
    elif estado == 2:
        if counti == 0:
            print("Fique olhando para o canto inferior esquerdo da tela!")
        x_ei = x_t
        y_ei = y_t
        count+=1
        counti+=1
        if count == 2.5 * tempo:
            estado+=1
            counti = 0
    # Mouse vai para o canto inferior direito da tela e a pessoa tem que olhar para ele
    elif estado == 3:
        if counti == 0:
            print("Fique olhando para o canto inferior direito da tela!")
        x_di = x_t
        y_di = y_t
        count+=1
        counti+=1
        if count == 3.5 * tempo:
            estado+=1
            counti = 0
    # Mouse vai para o canto superior direito da tela e a pessoa tem que olhar para ele
    else:
        if counti == 0:
            print("Fique olhando para o canto superior direito da tela!")
        x_ds = x_t
        y_ds = y_t
        count+=1
        counti+=1

    cv2.imshow("img", img)

x_to = x_es if x_ds>x_es else x_ds
x_tf = x_di if x_di>x_ei else x_ei
y_to = y_es if y_ds>y_es else y_ds
y_tf = y_di if y_di>y_ei else y_ei
x_t,y_t,w_t,h_t = 0,0,0,0
count = 0

# Tracking da pupila e movimentação do mouse
while True:
    # Captura do frame
    ok, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # Definição da nova região de interesse em escala de cinza e colorida (região da face)
        roi_gray = gray[y:y+int(h/2), x:x+int(w/2)]
        roi_color = img[y:y+int(h/2), x:x+int(w/2)]
        x_f,y_f,w_f,h_f = x,y,w,h
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.35, 5)

        for (ex,ey,ew,eh) in eyes:
            # Definição da nova região de interesse em escala de cinza e colorida (região dos olhos)
            roi_gray2 = roi_gray[int(ey):int(ey+eh), int(ex):int(ex+ew)]
            roi_color2 = roi_color[ey:ey+eh, ex:ex+ew]
            x_e,y_e,w_e,h_e = ex,ey,ew,eh
        # Desenhando o retângulo ao redor do olho detectado por último
        cv2.rectangle(roi_color,(x_e,y_e),(x_e+w_e,y_e+h_e),(0,255,0),2)

        # Percorrendo a imagem roi_gray2 comparando com a imagem template (w1xh1)
        # O melhor resultado é encontrada através do máx global de res (cv2.minMaxLoc), encontrando a sua posição (x,y)
        res = cv2.matchTemplate(roi_gray2,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w1, top_left[1] + h1)

        # Desenhando o retângulo ao redor da pupila com um ponto vermelho no centro
        cv2.rectangle(roi_color2,top_left, bottom_right, (255,255,255), 2)
        cv2.circle(roi_color2,(int(top_left[0]+w1/2),int(top_left[1]+h1/2)), 4, (0,0,255), -1)
        x_t,y_t,w_t,h_t = top_left[0]+x_e+x_f,top_left[1]+y_e+y_f,w1,h1

    count+=1

    if count == 10:
        # Se detectar a pupila - move o mouse
        if x_t!=0 and y_t!=0 and h_t!=0 and w_t!=0:
            x_p = int((screenWidth/(x_to - x_tf))*(x_t - x_tf))
            y_p = int((screenHeight/(y_tf - y_to))*(y_t - y_to))
            pyautogui.moveTo(x_p,y_p)
        count = 0

    # Mostrar a imagem
    cv2.imshow("Tracking", img)

    # Se apertar Esc sai
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

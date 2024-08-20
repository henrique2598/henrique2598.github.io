#-------------------
#-   Importações   -
#-------------------
import numpy as np
import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import json
import random

#--------------------------------------
#-   Definição dos paths e variáveis  -
#--------------------------------------
Camera_Index = 0
winName = 'Chamada Virtual - Grupo 6'
Path_CascadeClassifier = "Assets/haarcascade_frontalface_default.xml"
Path_TelaCadastro = "Assets/Background-Cadastro.png"
Path_TelaInicial = "Assets/Background-Inicial.png"
Path_TelaCaptura = "Assets/Background-Captura.png"
Path_TelaCondicional = "Assets/Background-Condicional.png"
Path_TelaEncerramento = "Assets/Background-Final.png"
Path_JsonClassDetails = "Assets/class_details.json"
Path_FacialModel = "Assets/facial_expression_model_structure.json"
Path_FacialWeights = "Assets/facial_expression_model_weights.h5"
Path_EncodingsFile = "Assets/encodings.pkl"

#----------------------
#-   Inicializações   -
#----------------------
# Carrega a detecção de rosto
face_cascade = cv2.CascadeClassifier(Path_CascadeClassifier)

# Carrega o modelo de expressões faciais
model = model_from_json(open(Path_FacialModel, "r").read())
model.load_weights(Path_FacialWeights)

# Emoções suportadas
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(Camera_Index)

# Cria o DataFrame para presenças
df_ListaDePresenca = pd.DataFrame(columns=['Nome do Aluno', 'Horário Entrada', 'Emoção Entrada', 'Horário Saída', 'Emoção Saída'])

# Carrega o background das telas
TelaCadastro = cv2.imread(Path_TelaCadastro)
TelaInicial = cv2.imread(Path_TelaInicial)
TelaCaptura = cv2.imread(Path_TelaCaptura)
TelaCondicional = cv2.imread(Path_TelaCondicional)
TelaEncerramento = cv2.imread(Path_TelaEncerramento)

# Carrega os detalhes da turma
with open(Path_JsonClassDetails, "r", encoding="utf-8") as j:
    class_details = json.loads(j.read())

# Carrega os encodings faciais
import pickle
with open(Path_EncodingsFile, "rb") as f:
    known_face_encodings = pickle.load(f)
    known_face_names = pickle.load(f)

#----------------------
#-   Reações de humor -
#----------------------
Reactions = {
    'angry': ["Respire fundo, transforme sua raiva em foco!", ...],
    'disgust': ["Encare o desconforto como um degrau para o sucesso!", ...],
    'fear': ["Coragem não é a ausência de medo, mas a decisão de seguir em frente!", ...],
    'happy': ["Sua alegria é o combustível para um dia incrível!", ...],
    'sad': ["Cada novo dia é uma nova chance de aprender algo incrível!", ...],
    'surprise': ["A surpresa é a chance de explorar novas possibilidades!", ...],
    'neutral': ["Cada aula é uma nova chance de descobrir algo incrível!", ...]
}

#--------------------------------------------
#-   Função para detectar e cadastrar alunos   -
#--------------------------------------------
def DetectaECadastraAluno(face_image):
    global df_ListaDePresenca
    aluno, aluno_nome = DetectaAluno(face_image)
    
    if aluno_nome:
        emocao = DetectaEmocao(face_image)
        
        if aluno_nome not in df_ListaDePresenca['Nome do Aluno'].values:
            new_row = {'Nome do Aluno': aluno_nome, 'Horário Entrada': '', 'Emoção Entrada': '', 'Horário Saída': '', 'Emoção Saída': ''}
            df_ListaDePresenca.loc[len(df_ListaDePresenca)] = new_row
            
        return aluno_nome, emocao
    return None, None

#------------------------------------------
#-   Função para identificação do aluno   -
#------------------------------------------
def DetectaAluno(frame):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    aluno_nome = None
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        aluno_nome = name
        break  # Supondo que haja apenas um rosto por vez
    
    return aluno_nome, aluno_nome

#----------------------------------------------------
#-   Função para identificação da emoção do aluno   -
#----------------------------------------------------
def DetectaEmocao(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (48, 48))
    img_pixels = np.expand_dims(face_image, axis=0)
    img_pixels = np.expand_dims(img_pixels, axis=3)
    img_pixels /= 255

    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    emotion = emotions[max_index]
    return emotion

#-------------------------------------------------------
#-   Função para gerar uma resposta para cada emoção   -
#-------------------------------------------------------
def GerarCondicionalDeEmocao(emotion, aluno):    
    ReactionList = Reactions.get(emotion, Reactions['neutral'])
    Condicional = f"{aluno}, {ReactionList[random.randint(0, len(ReactionList)-1)]}"
    
    TelaCondicional = cv2.imread(Path_TelaCondicional)
    cv2.putText(TelaCondicional, Condicional, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imshow(winName, TelaCondicional)
    Reagindo = True
    while Reagindo:
        if cv2.waitKey(1) & 0xFF == ord('p'):
            Reagindo = False

#-------------------------------------------------------------------------------
#-   Função para gravar as informações de presença e humor no banco de dados   -
#-------------------------------------------------------------------------------
def GravarNoBanco(df_ListaDePresenca, aluno, emocao, periodo, horario, Total_Students):
    if periodo == 1:
        if (df_ListaDePresenca['Nome do Aluno'] == aluno).any():
            pass
        else:
            new_row = {'Nome do Aluno': aluno, 'Horário Entrada': horario, 'Emoção Entrada': emocao, 'Horário Saída': '', 'Emoção Saída': ''}
            df_ListaDePresenca.loc[len(df_ListaDePresenca)] = new_row
            Total_Students += 1
    else:
        if (df_ListaDePresenca['Nome do Aluno'] == aluno).any():
            index = df_ListaDePresenca.index[df_ListaDePresenca['Nome do Aluno'] == aluno].tolist()
            if df_ListaDePresenca.iloc[index[0]]['Emoção Saída'] == '':
                df_ListaDePresenca.at[index[0], 'Horário Saída'] = horario
                df_ListaDePresenca.at[index[0], 'Emoção Saída'] = emocao
                Total_Students += 1
    return df_ListaDePresenca, Total_Students

#--------------------------------------------------
#-   Função para gerar o relatório do professor   -
#--------------------------------------------------
def GerarRelatorio(df_ListaDePresenca):
    nome_arquivo = f"Relatorio_Turma_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    df_ListaDePresenca.to_excel(nome_arquivo, index=False)
    print(f"Relatório gerado com sucesso: {nome_arquivo}")

#----------------------------
#-      Loop Principal      -
#----------------------------
cv2.putText(TelaInicial, class_details['Professor'], (120, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
cv2.putText(TelaInicial, class_details['Disciplina'], (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
cv2.putText(TelaInicial, class_details['Turma'], (90, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

Periodo = 0
DandoAula = True
Capturando = False

while DandoAula:
    cv2.imshow(winName, TelaInicial)
    
    while Periodo == 0:
        if cv2.waitKey(1) & 0xFF == ord('c'):
            Periodo = 1
            TelaCaptura = cv2.imread(Path_TelaCaptura)
            cv2.putText(TelaCadastro, "Entrada", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        elif cv2.waitKey(1) & 0xFF == ord('e'):
            Periodo = 1
            TelaCaptura = cv2.imread(Path_TelaCaptura)
            cv2.putText(TelaCadastro, "Entrada", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            Periodo = 2
            TelaCaptura = cv2.imread(Path_TelaCaptura)
            cv2.putText(TelaCadastro, "Saida", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    Capturando = True
    Total_Students = 0

    while Capturando:
        ret, cam = cap.read()
        gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(cam, (x, y), (x+w, y+h), (255, 0, 0), 2)
            detected_face = cam[int(y):int(y+h), int(x):int(x+w)]
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            detected_face = cv2.resize(detected_face, (48, 48))

            Student_Name, Student_emotion = DetectaECadastraAluno(detected_face)

            if Student_Name:
                df_ListaDePresenca, Total_Students = GravarNoBanco(df_ListaDePresenca, Student_Name, Student_emotion, Periodo, datetime.now(), Total_Students)
                GerarCondicionalDeEmocao(Student_emotion, Student_Name)

        img = cv2.hconcat([cam, TelaCaptura])
        cv2.putText(img, str(Total_Students), (920, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(img, str(Student_Name), (760, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(img, str(Student_emotion), (760, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.imshow(winName, img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            Capturando = False
    
    if Periodo == 1:
        Periodo = 0
    elif Periodo == 2:
        GerarRelatorio(df_ListaDePresenca)
        cv2.imshow(winName, TelaEncerramento)
        while DandoAula:
            if cv2.waitKey(1) & 0xFF == ord('x'):
                DandoAula = False

cap.release()
cv2.destroyAllWindows()

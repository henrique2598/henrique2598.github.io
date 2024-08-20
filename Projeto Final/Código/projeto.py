#-------------------
#-   Importações   -
#-------------------
import numpy as np
import cv2
from keras.utils import img_to_array
from keras.models import model_from_json
import json
from datetime import datetime
import pandas as pd
import random
import face_recognition  # Precisamos adicionar essa parte

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
Path_ImagemDosAlunos = "Assets/ImagemDosAlunos/"  # Diretório com imagens dos alunos

#----------------------
#-   Inicializações   -
#----------------------
# Definição do classificador para detecção dos rostos
face_cascade = cv2.CascadeClassifier(Path_CascadeClassifier)

# Definição do modelo utilizado para reconhecimento da expressão facial
model = model_from_json(open(Path_FacialModel, "r").read())

# Definição dos pesos do modelo utilizado para reconhecimento da expressão facial
model.load_weights(Path_FacialWeights)

# Definição das imagens suportadas
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Definição da entrada de vídeo
cap = cv2.VideoCapture(Camera_Index)

# Criação do DataFrame Pandas
df_ListaDePresenca = pd.DataFrame(columns=['Nome do Aluno', 'Horário Entrada', 'Emoção Entrada', 'Horário Saída', 'Emoção Saída'])

# Carrega o backgroud de cada etapa
TelaCadastro = cv2.imread(Path_TelaCadastro)
TelaInicial = cv2.imread(Path_TelaInicial)
TelaCaptura = cv2.imread(Path_TelaCaptura)
TelaCondicional = cv2.imread(Path_TelaCondicional)
TelaEncerramento = cv2.imread(Path_TelaEncerramento)

# Define a janela de exibição das imagens
cv2.namedWindow(winName, cv2.WINDOW_FULLSCREEN)

# Posiciona a janela na posição (100, 100)
cv2.moveWindow(winName, 100, 100)

# Leitura do json com as informações da turma
with open(Path_JsonClassDetails, "r", encoding="utf-8") as j:
     class_details = json.loads(j.read())


# Função para carregar imagens de alunos e criar codificações
def load_student_encodings():
    known_face_encodings = []
    known_face_names = []
    
    for student_name in student_names:
        image_path = f"{Path_StudentImages}/{student_name}.jpg"
        student_image = face_recognition.load_image_file(image_path)
        student_face_encoding = face_recognition.face_encodings(student_image)
        
        if student_face_encoding:
            known_face_encodings.append(student_face_encoding[0])
            known_face_names.append(student_name)
    
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_student_encodings()

#------------------------------------------
#-   Função para identificação do aluno   -
#------------------------------------------
def DetectaAluno(detected_face):
    """
    Identifica o aluno com base na face detectada.
    
    :param detected_face: A imagem da face detectada
    :return: O nome do aluno identificado, ou uma string vazia se não for identificado
    """
    # Codificar a face detectada
    detected_face_encoding = face_recognition.face_encodings(detected_face)
    
    if not detected_face_encoding:
        return ""
    
    detected_face_encoding = detected_face_encoding[0]
    
    # Comparar com os rostos conhecidos
    matches = face_recognition.compare_faces(known_face_encodings, detected_face_encoding)
    name = ""
    
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    
    return name

#------------------------
#-   Reações de humor   -
#------------------------
Reactions_angry = [
	"Respire fundo, transforme sua raiva em foco!",
	"Cada desafio é uma oportunidade para crescer, até a frustração!",
	"Controle suas emoções e domine seu dia!",
	"Use essa energia para conquistar seus objetivos!",
	"O que importa é como você escolhe reagir, não o que aconteceu!"
	]
Reactions_disgust = [
	"Encare o desconforto como um degrau para o sucesso!",
	"Desafios momentâneos não definem seu dia, siga em frente!",
	"Transforme o que te incomoda em motivação para vencer!",
	"Você é mais forte do que qualquer situação desconfortável!",
	"Foque no que importa: seu crescimento e aprendizado!"
	]
Reactions_fear = [
	"Coragem não é a ausência de medo, mas a decisão de seguir em frente!",
	"Transforme o medo em combustível para sua vitória!",
	"Cada passo dado com medo é um passo mais perto do sucesso!",
	"Você é mais corajoso do que imagina; vá em frente!",
	"Enfrente seus medos, eles são apenas oportunidades disfarçadas!",
	]
Reactions_happy = [
	"Sua alegria é o combustível para um dia incrível!",
	"Use sua energia positiva para brilhar ainda mais!",
	"A felicidade de hoje é o começo de grandes conquistas!",
	"Aproveite seu entusiasmo e transforme-o em aprendizado!",
	"Seu sorriso é a chave para um dia produtivo e gratificante!"
	]
Reactions_sad = [
	"Cada novo dia é uma nova chance de aprender algo incrível!",
	"Você é capaz de coisas incríveis, acredite em si mesmo!",
	"Mesmo os dias difíceis te ajudam a crescer!",
	"Seu esforço hoje será sua vitória amanhã!",
	"Nunca subestime o poder de um pequeno passo em frente!"
	]
Reactions_surprise = [
	"A surpresa é a chance de explorar novas possibilidades!",
	"Aceite o inesperado como uma oportunidade para aprender!",
	"Deixe a surpresa inspirar sua curiosidade e entusiasmo!",
	"O inesperado pode abrir portas para grandes descobertas!",
	"Abrace a surpresa, ela pode levar você a novas aventuras!"
	]
Reactions_neutral = [
	"Cada aula é uma nova chance de descobrir algo incrível!",
	"Às vezes, o simples ato de começar é o primeiro passo para grandes conquistas.",
	"Sua atitude positiva pode transformar um dia comum em algo extraordinário!",
	"Dê o seu melhor hoje; cada pequeno esforço conta!",
	"Aproveite o dia e faça dele uma oportunidade de crescimento!"
	]

#--------------------------------------------
#-   Função para cadastro de novos alunos   -
#--------------------------------------------
def CadastraAluno():
    # Exibe o Loop para cadastro dos alunos
    Capturando = True
    while Capturando:
        # Lê a webcam
        ret, cam = cap.read()

        # Atualiza a tela de cadastro
        img = cv2.hconcat([cam, TelaCadastro])

        # Exibe a tela
        cv2.imshow(winName, img)

        # Verifica se o cadastro foi concluído
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            Capturando = False

#----------------------------------------------------
#-   Função para identificação da emoção do aluno   -
#----------------------------------------------------
def DetectaEmocao(detected_face):
    # Detecta a emoção
    img_pixels = img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis=0)

    img_pixels /= 255  # Pixels are in scale of [0, 255]. Normalize all pixels in scale of [0, 1]

    predictions = model.predict(img_pixels)  # Store probabilities of 7 expressions

    # Find max indexed array 0: angry, 1: disgust, 2: fear, 3: happy, 4: sad, 5: surprise, 6: neutral
    max_index = np.argmax(predictions[0])

    emotion = emotions[max_index]

    return emotion

#-------------------------------------------------------
#-   Função para identificação da emoção do aluno  -
#-------------------------------------------------------
def GerarCondicionalDeEmocao(emotion, aluno):	
	if (emotion=='angry'):
		ReactionList = Reactions_angry
	elif (emotion=='disgust'):
		ReactionList = Reactions_disgust
	elif (emotion=='fear'):
		ReactionList = Reactions_fear
	elif (emotion=='happy'):
		ReactionList = Reactions_happy
	elif (emotion=='sad'):
		ReactionList = Reactions_sad
	elif (emotion=='surprise'):
		ReactionList = Reactions_surprise
	else:
		ReactionList = Reactions_neutral

	Condicional = aluno + ", " + ReactionList[random.randint(0, len(ReactionList)-1)]
	
	TelaCondicional = cv2.imread(Path_TelaCondicional)
	cv2.putText(TelaCondicional, Condicional, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

	# Atualiza a tela com a condicional
	cv2.imshow(winName,TelaCondicional)
	Reagindo=True
	#  Encerrar o programa
	while (Reagindo==True):
		if cv2.waitKey(1) & 0xFF == ord('p'):
			Reagindo = False
#-------------------------------------------------------------------------------
#-   Função para gravar as informações de presença e humor no banco de dados   -
#-------------------------------------------------------------------------------
def GravarNoBanco(df_ListaDePresenca, aluno, emocao, periodo, horario, Total_Students):
	# Grava a entrada do aluno
	if (periodo == 1):
		# Verifica se o aluno já foi ou não registrado
		if (df_ListaDePresenca == aluno).any().any():
			# O aluno já marcou a entrada
			pass
		else:
		# O aluno ainda não marcou a entrada
			new_row = {'Nome do Aluno': aluno, 'Horário Entrada': horario, 'Emoção Entrada': emocao, 'Horário Saída': '', 'Emoção Saída': ''}
			df_ListaDePresenca.loc[len(df_ListaDePresenca)] = new_row
			# Atualiza os alunos presentes
			Total_Students+=1
	# Grava a saída do aluno
	else:
		# Verifica se o aluno marcou a entrada
		if (df_ListaDePresenca == aluno).any().any():
			# O aluno marcou a entrada - identifica o index
			index = df_ListaDePresenca.index[df_ListaDePresenca['Nome do Aluno'] == aluno].tolist()
			# Verifica se o aluno marcou a saída
			if (df_ListaDePresenca.iloc[index[0]]['Emoção Saída'] == ''):
				# O aluno não marcou a saída			
				df_ListaDePresenca.at[index[0], 'Horário Saída'] = horario
				df_ListaDePresenca.at[index[0], 'Emoção Saída'] = emocao
				# Atualiza os alunos presentes
				Total_Students+=1
		# O aluno não marcou a entrada
		else:
			pass
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
# Adiciona as informações da turma na tela inicial
cv2.putText(TelaInicial, class_details['Professor'], (120, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
cv2.putText(TelaInicial, class_details['Disciplina'], (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
cv2.putText(TelaInicial, class_details['Turma'], (90, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Variáveis de estado do programa
Periodo = 0
DandoAula=True
Capturando=False

# Loop principal
while(DandoAula==True):
	#Exibe tela inicial com informações da turma
	cv2.imshow(winName, TelaInicial)
	
	# Decide se é cadastro, entrada ou saída dos alunos
	while (Periodo==0):
		# Cadastro de alunos
		if cv2.waitKey(1) & 0xFF == ord('c'):
			CadastraAluno()
			cv2.imshow(winName, TelaInicial)
		# Entrada de alunos
		elif cv2.waitKey(1) & 0xFF == ord('e'):
			Periodo = 1
			TelaCaptura = cv2.imread(Path_TelaCaptura)
			cv2.putText(TelaCadastro, "Entrada", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
		# Saída de alunos
		elif cv2.waitKey(1) & 0xFF == ord('s'):
			Periodo = 2
			TelaCaptura = cv2.imread(Path_TelaCaptura)
			cv2.putText(TelaCadastro, "Saida", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

	# Habilita a etapa de cadastro e inicializa a variável contadora de alunos registrados
	Capturando = True
	Total_Students = 0

	# Exibe o Loop para registro dos alunos
	while(Capturando==True):
		# Lê a webcam
		ret, cam = cap.read()
		gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)

		# Detecta os rostos
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		
		# Limpa as variáveis
		Student_Name = ""
		Student_emotion = ""

		# Percorre os rostos detectados
		for (x,y,w,h) in faces:
			cv2.rectangle(cam,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
			
			# Recorta o rosto detectado e pré-processa a imagem
			detected_face = cam[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			
			#Validar se a turma está correta
			Student_Name = DetectaAluno

			if (Student_Name != ""):
				# Função que detecta a emoção
				Student_emotion = DetectaEmocao(detected_face)
				# Grava as informações da turma
				Student_Name = 'Aluno1'
				df_ListaDePresenca, Total_Students = GravarNoBanco(df_ListaDePresenca, Student_Name, Student_emotion, Periodo, datetime.now(), Total_Students)
				# Reação personalizada de acordo com a emoção
				GerarCondicionalDeEmocao(Student_emotion, Student_Name)

		# Atualiza a tela de cadastro
		img = cv2.hconcat([cam, TelaCaptura])

		# Adiciona as informações atualizadas na tela
		cv2.putText(img, str(Total_Students), (920, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
		cv2.putText(img, str(Student_Name), (760, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
		cv2.putText(img, str(Student_emotion), (760, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

		# Exibe a tela
		cv2.imshow(winName,img)

		# Verifica se o cadastro foi concluído
		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			Capturando = False
	

	# Decide o fluxo com base no período
	if Periodo == 1:
		# Finalizou a entrada - Volta para tela inicial   
		Periodo = 0
	elif Periodo == 2:
		# Finalizou a saída - gerar relatório
		GerarRelatorio(df_ListaDePresenca)
		# Atualiza a tela de encerramento
		cv2.imshow(winName,TelaEncerramento)
		#  Encerrar o programa
		while (DandoAula==True):
			if cv2.waitKey(1) & 0xFF == ord('x'):
				DandoAula = False


# Fechas a câmera e janela aberta		
cap.release()
cv2.destroyAllWindows()

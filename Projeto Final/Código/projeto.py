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
Path_Relatorio = "Relatórios/"
Path_ImagensDosAlunos = "Alunos Cadastrados"  # Diretório com imagens dos alunos
Path_CascadeClassifier = "Assets/haarcascade_frontalface_default.xml"
Path_TelaCadastro = "Assets/Background-Cadastro.png"
Path_TelaInicial = "Assets/Background-Inicial.png"
Path_TelaCaptura = "Assets/Background-Captura.png"
Path_TelaCondicional = "Assets/Background-Condicional.png"
Path_TelaEncerramento = "Assets/Background-Final.png"
Path_JsonClassDetails = "Assets/class_details.json"
Path_FacialModel = "Assets/facial_expression_model_structure.json"
Path_FacialWeights = "Assets/facial_expression_model_weights.h5"


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


#Definição da variável de nome dos alunos
student_names = ['henrique']  # Ideal fazermos simulando a lista de chamada

# Função para carregar imagens de alunos e criar codificações
def get_rostos():
	rostos_conhecidos = []
	nome_dos_rostos = []
	for student_name in student_names:
		image_path = f"{Path_ImagensDosAlunos}/{student_name}.png"
		student_image = face_recognition.load_image_file(image_path)
		student_face_encoding = face_recognition.face_encodings(student_image)
		if (len(student_face_encoding) > 0):
			rostos_conhecidos.append(student_face_encoding[0])
			nome_dos_rostos.append(student_name)
	return rostos_conhecidos, nome_dos_rostos

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
    #detected_face_encoding = face_recognition.face_encodings(detected_face)  
    #detected_face_encoding = detected_face_encoding[0]
    

    rostos_cadastrados, nomes_cadastrados = get_rostos()


    # Comparar com os rostos conhecidos
    matches = face_recognition.compare_faces(rostos_cadastrados, detected_face)
    face_distances = face_recognition.face_distance (rostos_cadastrados, detected_face)
    melhor_id = np.argmin(face_distances)
    if matches[melhor_id]:
        nome = nomes_cadastrados[melhor_id]
    else:
        nome = "Desconhecido"

    return nome


#------------------------
#-   Reações de humor   -
#------------------------
Reactions_angry = [
	"Respire fundo, transforme\nsua raiva em foco!",
	"Cada desafio e uma\noportunidade para crescer,\nate a frustracao!",
	"Controle suas emocoes e\ndomine seu dia!",
	"Use essa energia para\nconquistar seus\nobjetivos!",
	"O que importa e como voce\nescolhe reagir, nao o que\naconteceu!"
	]
Reactions_disgust = [
	"Encare o desconforto como\num degrau para o sucesso!",
	"Desafios momentaneos nao\ndefinem seu dia, siga em\nfrente!",
	"Transforme o que te\nincomoda em motivacao para\nvencer!",
	"Voce e mais forte do que\nqualquer situacao\ndesconfortavel!",
	"Foque no que importa: seu\ncrescimento e aprendizado!"
	]
Reactions_fear = [
	"Coragem nao e a ausencia\nde medo, mas a decisao\nde seguir em frente!",
	"Transforme o medo em\ncombustivel para sua\nvitoria!",
	"Cada passo dado com medo\ne um passo mais perto\ndo sucesso!",
	"Voce e mais corajoso do\nque imagina; va em frente!",
	"Enfrente seus medos, eles\nsao apenas oportunidades\ndisfarcadas!",
	]
Reactions_happy = [
	"Sua alegria e o combustivel\npara um dia incrivel!",
	"Use sua energia positiva\npara brilhar ainda mais!",
	"A felicidade de hoje e o\ncomeco de grandes\nconquistas!",
	"Aproveite seu entusiasmo e\ntransforme-o em\naprendizado!",
	"Seu sorriso e a chave para\num dia produtivo e\ngratificante!"
	]
Reactions_sad = [
	"Cada novo dia e uma nova\nchance de aprender\nalgo incrivel!",
	"Voce e capaz de coisas\nincriveis, acredite em\nsi mesmo!",
	"Mesmo os dias dificeis\nte ajudam a crescer!",
	"Seu esforco hoje sera\nsua vitoria amanha!",
	"Nunca subestime o poder de\num pequeno passo em frente!"
	]
Reactions_surprise = [
	"A surpresa e a chance de\nexplorar novas\npossibilidades!",
	"Aceite o inesperado como\numa oportunidade para\naprender!",
	"Deixe a surpresa inspirar\nsua curiosidade e entusiasmo!",
	"O inesperado pode abrir\nportas para grandes\ndescobertas!",
	"Abrace a surpresa, ela\npode levar voce a novas\naventuras!"
	]
Reactions_neutral = [
	"Cada aula e uma nova\nchance de descobrir algo\nincrivel!",
	"As vezes, o simples ato\nde começar e o primeiro\npasso para grandes\nconquistas.",
	"Sua atitude positiva pode\ntransformar um dia comum\nem algo extraordinario!",
	"De o seu melhor hoje; cada\npequeno esforco conta!",
	"Aproveite o dia e faca dele\numa oportunidade de\ncrescimento!"
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
def GerarCondicionalDeEmocao(cam, emotion, aluno):	
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

	Condicional = aluno + ",\n" + ReactionList[random.randint(0, len(ReactionList)-1)]

	TelaCondicional = cv2.imread(Path_TelaCondicional)
	cv2.putText(TelaCondicional, str(aluno), (135, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	cv2.putText(TelaCondicional, str(emotion), (150, 145), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	
	y0, dy = 200, 35
	for i, line in enumerate(Condicional.split('\n')):
		y = y0 + i*dy
		cv2.putText(TelaCondicional, line, (45, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    
	# Atualiza a tela de cadastro
	img = cv2.hconcat([cam, TelaCondicional])
	
	# Atualiza a tela com a condicional
	cv2.imshow(winName,img)
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
	df_ListaDePresenca.to_excel(Path_Relatorio+nome_arquivo, index=False)


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
			cv2.putText(TelaCaptura, "Entrada", (135, 215), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		# Saída de alunos
		elif cv2.waitKey(1) & 0xFF == ord('s'):
			Periodo = 2
			TelaCaptura = cv2.imread(Path_TelaCaptura)
			cv2.putText(TelaCaptura, "Saida", (135, 215), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

	# Habilita a etapa de cadastro e inicializa a variável contadora de alunos registrados
	Capturando = True
	Total_Students = 0

	# Exibe o Loop para registro dos alunos
	while(Capturando==True):
		# Lê a webcam
		ret, cam = cap.read()
		frame = cam
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
			rgb_frame = frame[:, :, ::-1]
			localizacao_dos_rostos = face_recognition.face_locations(rgb_frame)
			rosto_desconhecidos = face_recognition.face_encodings(rgb_frame, localizacao_dos_rostos)
			for (top, right, bottom, left), rosto_desconhecido in zip(localizacao_dos_rostos, rosto_desconhecidos):
				Student_Name = DetectaAluno(rosto_desconhecido)

			if (Student_Name != ""):
				#Validar se o aluno já registrou a presença na entrada
				if (Student_Name not in df_ListaDePresenca['Nome do Aluno'].values):
					Student_emotion = DetectaEmocao(detected_face)
					df_ListaDePresenca, Total_Students = GravarNoBanco(df_ListaDePresenca, Student_Name, Student_emotion, Periodo, datetime.now(), Total_Students)
					GerarCondicionalDeEmocao(cam, Student_emotion, Student_Name)
				#Validar se o aluno já registrou a presença na saída
				elif (Periodo == 2) and (df_ListaDePresenca['Horário Saída'][df_ListaDePresenca.index[df_ListaDePresenca['Nome do Aluno'] == Student_Name].tolist()[0]]==''):
					Student_emotion = DetectaEmocao(detected_face)
					df_ListaDePresenca, Total_Students = GravarNoBanco(df_ListaDePresenca, Student_Name, Student_emotion, Periodo, datetime.now(), Total_Students)
					GerarCondicionalDeEmocao(cam, Student_emotion, Student_Name)

		# Atualiza a tela de cadastro
		img = cv2.hconcat([cam, TelaCaptura])

		# Adiciona as informações atualizadas na tela
		cv2.putText(img, str(Total_Students), (925, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		
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

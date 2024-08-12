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

#---------------------------
#-   Definição dos Paths   -
#---------------------------
Camera_Index = 0
Path_CascadeClassifier = "Assets/haarcascade_frontalface_default.xml"
Path_TelaInicial = "Assets/Background-Inicial.png"
Path_TelaCadastro = "Assets/Background-Cadastro.png"
Path_TelaEncerramento = "Assets/Background-Final.png"
Path_JsonClassDetails = "Assets/class_details.json"
Path_FacialModel = "Assets/facial_expression_model_structure.json"
Path_FacialWeights = "Assets/facial_expression_model_weights.h5"


#-----------------------------
#-   opencv initialization   -
#-----------------------------
face_cascade = cv2.CascadeClassifier(Path_CascadeClassifier)
cap = cv2.VideoCapture(Camera_Index)



# Create the pandas DataFrame
df_ListaDePresenca = pd.DataFrame(columns=['Nome do Aluno', 'Horário Entrada', 'Emoção Entrada', 'Horário Saída', 'Emoção Saída'])


#--------------------------------
#-   interface initialization   -
#--------------------------------
#Carrega o backgroud de cada etapa
TelaInicial = cv2.imread(Path_TelaInicial)
TelaCadastro = cv2.imread(Path_TelaCadastro)
TelaEncerramento = cv2.imread(Path_TelaEncerramento)

# Convert JSON String to Python
with open(Path_JsonClassDetails, "r", encoding="utf-8") as j:
     class_details = json.loads(j.read())

# Print values using keys
cv2.putText(TelaInicial, class_details['Professor'], (120, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
cv2.putText(TelaInicial, class_details['Disciplina'], (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
cv2.putText(TelaInicial, class_details['Turma'], (90, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Define a janela de exibição das imagens, com tamanho automático
winName = 'Chamada Virtual - Grupo 6'
cv2.namedWindow(winName, cv2.WINDOW_FULLSCREEN)

# Posiciona a janela na metade direita do (meu) monitor
cv2.moveWindow(winName, 100, 100)





#-----------------------------------------------
#-  face expression recognizer initialization  -
#-----------------------------------------------
model = model_from_json(open(Path_FacialModel, "r").read())
model.load_weights(Path_FacialWeights)
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')



def DetectaAluno(detected_face):
	pass



def DetectaEmocao(detected_face):
	#detecta a emoção
	img_pixels = img_to_array(detected_face)
	img_pixels = np.expand_dims(img_pixels, axis = 0)

	img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
	
	predictions = model.predict(img_pixels) #store probabilities of 7 expressions
	
	#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
	max_index = np.argmax(predictions[0])
	
	emotion = emotions[max_index]

	return emotion


def GerarCondicionalDeEmocao(emotion):
	pass


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

def GerarRelatorio(df_ListaDePresenca):
	# print dataframe.
	print(df_ListaDePresenca)
	
#----------------------------
#-      Loop Principal      -
#----------------------------


# Repetição - Dia de Aula
Periodo = 0
DandoAula=True
Cadastrando=False


while(DandoAula==True):
	#Exibe tela inicial com informações da turma
	cv2.imshow(winName, TelaInicial)
	
	# Decide se é entrada ou saída dos alunos
	while (Periodo==0):
		if cv2.waitKey(1) & 0xFF == ord('e'):
			Periodo = 1
			TelaCadastro = cv2.imread(Path_TelaCadastro)
			cv2.putText(TelaCadastro, "Entrada", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
		elif cv2.waitKey(1) & 0xFF == ord('s'):
			Periodo = 2
			TelaCadastro = cv2.imread(Path_TelaCadastro)
			cv2.putText(TelaCadastro, "Saida", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

	# Habilita a etapa de cadastro e inicializa as variáveis
	Cadastrando = True
	Total_Students = 0

	# Exibe o Loop para registro dos alunos
	while(Cadastrando==True):
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
				# Reação personalizada de acordo com a emoção
				GerarCondicionalDeEmocao(Student_emotion)
				# Grava as informações da turma
				Student_Name = 'Aluno1'
				df_ListaDePresenca, Total_Students = GravarNoBanco(df_ListaDePresenca, Student_Name, Student_emotion, Periodo, datetime.now(), Total_Students)


		# Atualiza a tela de cadastro
		img = cv2.hconcat([cam, TelaCadastro])

		#write total students text above rectangle
		cv2.putText(img, str(Total_Students), (920, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
		#write student text above rectangle
		cv2.putText(img, str(Student_Name), (760, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
		#write emotion text above rectangle
		cv2.putText(img, str(Student_emotion), (760, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
		cv2.imshow(winName,img)

		# Verifica se o cadastro foi concluído
		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			Cadastrando = False
	
	# Decide o fluxo com base no período
	if Periodo == 1:
		# Finalizou a entrada - Volta para tela inicial   
		Periodo = 0
	elif Periodo == 2:
		# Finalizou a saída - gerar relatório e encerrar o programa
		GerarRelatorio(df_ListaDePresenca)
		# Atualiza a tela de encerramento
		cv2.imshow(winName,TelaEncerramento)
		while (DandoAula==True):
			if cv2.waitKey(1) & 0xFF == ord('x'):
				DandoAula = False


#kill open cv things		
cap.release()
cv2.destroyAllWindows()
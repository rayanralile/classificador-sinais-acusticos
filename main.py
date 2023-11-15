from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QLineEdit, QPushButton, QFileDialog, QComboBox, QWidget, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QMessageBox
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap

from TCC_tools import ID3_classifier

# Agora os imports da Rede Neural
import numpy as np
import scipy.signal
import librosa
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle

def carrega_objeto(file_path = "data.pickle"):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj

def load_data(filename, start_seconds, end_seconds):
    # Carrega a rede neural selecionada. Como só tem uma, está direta
    #model = load_model("modelKeras40epocas.h5")
    
    #self.audio_path.text()
    path = filename
    # Reescrita de código da área jupyter
    
    scaler = MinMaxScaler()
    #audio_data, original_sample_rate = librosa.load("6__10_07_13_marDeCangas_Entra.wav", sr=None)

    audio_data, original_sample_rate = librosa.load(path, sr=None)
    audio_data = librosa.resample(audio_data, orig_sr=original_sample_rate, target_sr=4800)
    sample_rate = 4800

    # Computa "start e end" para 5 minutos em torno do centro - inserido manualmente
    start_sample = max(0, start_seconds * sample_rate)
    #end_sample = min(len(audio_data), int((2.5*60 + 2.5 * 60) * sample_rate))
    #seconds = math.floor(len(audio_data) / sample_rate)
    #end_sample = min(len(audio_data), seconds * sample_rate)
    end_sample = min(len(audio_data), int((end_seconds) * sample_rate))#43

    seconds = end_seconds - start_seconds

    # Pega 5 minutos
    audio_data = audio_data[start_sample:end_sample]

    # Reshape para 300 amostras de 1 segundo cada
    #chunks = audio_data.reshape(300, sample_rate)
    chunks = audio_data.reshape(seconds, sample_rate)

    psd_list = []
    for chunk in chunks:
        freqs, psd = scipy.signal.welch(chunk, sample_rate, nperseg=480, noverlap=240)

        psd = psd[(freqs >= 10) & (freqs <= 2000)]
        psd_list.append(psd)

    result = np.array(psd_list)


    samples = []


    samples.extend(result)


    samples = np.array(samples)


    samples = scaler.fit_transform(samples)

    return samples

def create_plot(classificados, plot_name = "default.svg"):
    # Plotagem dos resultados
    classifications = classificados
    grande = 0
    pequeno = 0
    for classif in classifications:
        if classif == 'Navio Grande Porte' or classif == 'Grande Porte' or classif == 0:
            grande += 1
        if classif == 'Navio Pequeno Porte' or classif == 'Pequeno Porte' or classif == 1:
            pequeno += 1
    # plotagem da pizza
    labels = ['Navio Grande Porte', 'Navio Pequeno Porte']
    sizes = [grande, pequeno]

    colors = ['#ff9999','#66b3ff']


    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

    # Círculo branco central pra fazer a cara da pizza
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)


    ax1.axis('equal')  
    plt.tight_layout()


    plt.savefig("output.png", format='png')
    plt.savefig(plot_name, format='svg')

    # Agora o veredito:
    total = grande + pequeno
    if grande > pequeno and (grande/total) >= 0.6:
        return "Navio de Grande Porte"
    elif pequeno > grande and (pequeno/total) >= 0.6:
        return "Navio de Pequeno Porte"
    else:
        return "Navio de Porte Desconhecido"

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Clasificador de Sinais Acústicos - Navios")
        self.setGeometry(0, 0, 1024, 768)

        # Parâmetro global
        self.start = 0
        self.end = 43

        widget = QWidget()
        layout = QGridLayout()

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Linha 1
        label1 = QLabel("Escolha seu áudio")
        layout.addWidget(label1, 0, 0, 1, 2) # Span entre as colunas

        # Linha 2
        self.audio_path = QLineEdit()
        layout.addWidget(self.audio_path, 1, 0)
        self.open_button = QPushButton("Abrir")
        layout.addWidget(self.open_button, 1, 1)
        self.open_button.clicked.connect(self.open_dialog)

        # Linha 3
        label2 = QLabel("Escolha sua Rede Neural")
        layout.addWidget(label2, 2, 0, 1, 2)

        # Linha 4
        self.combobox = QComboBox()
        self.combobox.addItem("Multi-layer Perceptron - Keras")
        self.combobox.addItem("Multi-layer Perceptron - Scikit-learn")
        self.combobox.addItem("Indutivo - ID3")
        self.combobox.setCurrentIndex(0)
        layout.addWidget(self.combobox, 3, 0)
        self.execute_button = QPushButton("Executar")
        layout.addWidget(self.execute_button, 3, 1)
        self.execute_button.clicked.connect(self.execute_function)

        # Linha 5
        self.result_label = QLabel("Resultado: ")
        layout.addWidget(self.result_label, 4, 0, 1, 2)

        # Linha 6
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view, 5, 0, 1, 2)

        # Linha 7
        self.classification_label = QLabel("Classificação: ")
        layout.addWidget(self.classification_label, 6, 0)
        self.classification_text = QLineEdit()
        self.classification_text.setReadOnly(True)
        layout.addWidget(self.classification_text, 6, 1)

    def open_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Abrir áudio mono', '', "WAV Files (*.wav)")
        if file_name:
            self.audio_path.setText(file_name)

    def plotQt(self):
        # Aqui ocorre o carregamento dessa imagem
        pixmap = QPixmap('output.png')
        self.scene.clear()
        self.scene.setSceneRect(QRectF(pixmap.rect()))  # Redimensiona para o tamanho da imagem
        self.scene.addPixmap(pixmap)  # Adiciona a imagem na cena

    def keras_run(self):
        samples = load_data(self.audio_path.text(), start_seconds=self.start, end_seconds=self.end)

        self.audio_path.text()
        model = load_model("modelKeras.h5")
        # Predição com Keras
        classifications = model.predict(samples) # executa a predição
        predicted_labels = np.argmax(classifications, axis=1) # Pegar a classificação com maior probabilidade
        
        
        resultadoFinal = create_plot(predicted_labels,"kerasMLP.svg") # Cria o plot, salva em imagem svg e no output.png
        # e decide qual classificação final irá adotar.
        self.classification_text.setText(resultadoFinal)
        
        self.plotQt() # Chama instruções para carregar output.png dentro do form do Qt como imagem
        
        
    
    def scikit_run(self):
        #self.show_alert('A rede scikit ainda não foi implementada','Rede não implementada')
        # Carrega a rede neural selecionada. Como só tem uma, está direta
        mlp = carrega_objeto("scikitmlp.pkl")
        
        samples = load_data(self.audio_path.text(), start_seconds=self.start, end_seconds=self.end)

        y_pred_train_cangas = mlp.predict(samples)

        resultadoFinal = create_plot(y_pred_train_cangas,"scikit-learnMLP.svg")

        self.classification_text.setText(resultadoFinal)

        self.plotQt()
    
    def ID3_run(self):
        amostras = load_data(self.audio_path.text(), start_seconds=self.start, end_seconds=self.end)

        amostras *= 100000000 # usando a escala arbitrada no ID3 nosso

        classificacoes = ID3_classifier(samples=amostras,file_path='indutivo.rbf',safe=True,key_path='tcc.key')
        
        resultadoFinal = create_plot(classificacoes,"ID3.svg")

        self.classification_text.setText(resultadoFinal)

        self.plotQt()

    def MLP_navios_baleias(self):
        samples = load_data(self.audio_path.text(), start_seconds=self.start, end_seconds=self.end)

        self.audio_path.text()
        model = load_model('modelBaleiasNavios2.h5')

        classifications = model.predict(samples) # executa a predição
        predicted_labels = np.argmax(classifications, axis=1) # Pegar a classificação com maior probabilidade

        # Agora temos a variável final_decision que vai guardar a classificação final e será usado
        # para mostrar o veredito de classificação ao usuário

        final_decision = ''

        # Aqui os pontos classificados serão contabilizados para a decisão entre navios ou baleias

        qtd_navios = 0
        qtd_baleias = 0
        for classif in predicted_labels:
            if classif == 'Navio' or classif == 0:
                qtd_navios += 1
            if classif == 'Baleia' or classif == 1:
                qtd_baleias += 1
                
        print(f'Navio = {qtd_navios}')
        print(f'Baleia = {qtd_baleias}')

        if (qtd_baleias/(qtd_navios+qtd_baleias)) >= 0.6:
            final_decision = "Baleia"
        elif (qtd_baleias/(qtd_navios+qtd_baleias)) > 0.4:
            final_decision = "Não identificado"
        else:
            final_decision = "Navio"
            
        # plotagem da pizza
        labels = ['Navio', 'Baleia']
        sizes = [qtd_navios, qtd_baleias]

        colors = ['#ff9999','#66b3ff']


        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

        # Círculo branco central pra fazer a cara da pizza
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)


        ax1.axis('equal')  
        plt.tight_layout()


        plt.savefig("output.png", format='png')
        plt.savefig("default.svg", format='svg')

        return final_decision
        


    def execute_function(self):
        # Primeiro verifica se o sinal é uma baleia.
        # Casos seja, ele já faz o plot na ferramenta
        # e encerra a execução da função
        firstClassif = self.MLP_navios_baleias()
        if firstClassif == "Baleia" or firstClassif == "Não identificado":
            self.classification_text.setText(firstClassif)
            self.plotQt()
            # Como não há mais execução, sai da função
            return
        
        # Caso contrário, sendo navio, vamos determinar o seu porte
        else:
            selected = self.combobox.currentText()
            if selected == "Multi-layer Perceptron - Keras":
                self.keras_run()
            elif selected == "Multi-layer Perceptron - Scikit-learn":
                self.scikit_run()
            elif selected == "Indutivo - ID3":
                self.ID3_run()
    
    # ABAIXO AS FUNÇÕES MINHAS DE SISTEMA
    def show_alert(self, msg, title = "Mensagem"):
        alert = QMessageBox()
        alert.setWindowTitle(title)
        alert.setText(msg)
        alert.exec_()

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

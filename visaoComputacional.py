import cv2
import numpy as np

# Codigo de limiarizacao global utilizando limiar L e sem idexacao logica
def limiarizacao_global_1(I , L):
    n_linhas, n_colunas = I.shape
    I_bin = np.zeros((n_linhas, n_colunas), np.uint8)

    for x in np.arange(0, n_colunas):
        for y in np.arange(0, n_linhas):

            if I[y,x] >= L:
                I_bin[y,x] = 255

    return I_bin 


# Codigo de limiarizacao global utilizando limiar L e com idexacao logica
def limiarizacao_global_2(I, limiar):
    n_linhas, n_colunas = I.shape

    I_bin = np.zeros((n_linhas,n_colunas), np.uint8)

    # Acessa os elementos de I_bin por idexacao logica
    indices = (I > limiar)
    I_bin[indices] = 255

    return I_bin


# Calcula o histograma
def imhist(I):

    hist = np.zeros(256)
    n_linhas,n_colunas = I.shape

    for y in np.arange(n_linhas):
        for x in np.arange(n_colunas):
            pixel_value = I[y,x]
            hist[pixel_value] = hist[pixel_value] + 1

    return hist


# Escalona a imagem com base no pixels vizinhos
def escalonador_img(I, M_l, N_l):

    M, N = I.shape

    A = np.array([[N_l/N, 0], [0, M_l/M]]) #x y

    A_inv = np.linalg.inv(A)

    I2 = np.zeros((M_l,N_l), np.uint8)

    vetor_colunas_d = np.zeros((2, 1))

    for y_l in np.arange(M_l):
        for x_l in np.arange(N_l):
            vetor_coluna = np.array([[x_l], [y_l]])
            for x in np.arange(len(A)): 
                v = 0
                for y in np.arange(len(A)): 
                    v += A_inv[y][x] * vetor_coluna[y]
                vetor_colunas_d[x] = v

            x_t = vetor_colunas_d[0][0]
            y_t = vetor_colunas_d[1][0]

            x1 =int(x_t)
            y1= int(y_t)
            x2, y2 = x1+1, y1+1
    
            # Pega os quatro pixels vizinhos
            I11 = I[y1, x1] # A
            I12 = I[y2, x1] # C
            I21 = I[y1, x2] # B
            I22 = I[y2, x2] # D

            # Faz a interpolação
            E = (I11*((x2-x_t)/(x2-x1))) +  (((x_t-x1)/(x2-x1))*I21)
            F = (I12*((x2-x_t)/(x2-x1))) +  (((x_t-x1)/(x2-x1))*I22)
            G = (E*((y2-y_t)/(y2-y1))) +  (((y_t-y1)/(y2-y1))*F)

            I2[y_l, x_l] = G

    return I2


# Realiza a homografia calculando os valores dos parametros h
def homografia(pts_org, pts_dst):
    A = []
    b = []
    for (x, y), (x_l,y_l) in zip(pts_org, pts_dst):
        A.append([x, y, 1, 0, 0, 0, -x_l*x, -x_l*y])
        A.append([0, 0, 0, x, y, 1, -y_l*x, -y_l*y])
        b.append([x_l])
        b.append([y_l])

    A = np.array(A)
    b = np.array(b)
    u = 0
    A_inv = np.linalg.inv(A)

    h = A_inv @ b
    H = np.ones((3,3))

    for x in np.arange(0,3):
        for y in np.arange(0,3):
            if u < 8:
                H[x,y] = h[u]
                u +=1
    return H


# Buffer simplificado para armazenar frames de vídeo
class videoBuffer:

    def __init__(self, image_shape, tamanho):
        self.tamanho = tamanho
        self.inicio = self.tamanho-1
        self.final =  0
        self.buffer = np.zeros((image_shape[0], image_shape[1], tamanho))

    def insereFrame(self, frame):
        self.inicio += 1
        if self.inicio == self.tamanho:
            self.inicio = 0
        
        self.final += 1
        if self.final == self.tamanho:
            self.final = 0

        self.buffer[:,:,self.inicio] = frame

    def primeiroFrame(self):
        return self.buffer[:,:,self.inicio]
    
    def ultimoFrame(self):
        return self.buffer[:,:,self.final]


# Filtro gaussiano 
def gaussianKernel(ksize, sigma):
    
    kernel = np.zeros((ksize,ksize), np.float32)
    v1 = 1/(2 * np.pi * (sigma**2))
    v2 = -1/(2 * (sigma**2))

    for y in np.arange(0,ksize):
        for x in np.arange(0,ksize):
            
            # corrige coordenadas para a função gaussiana
            j = y - (ksize-1)/2
            i = x - (ksize-1)/2

            kernel[y,x] = v1 * np.exp(v2*(j**2 + i**2))

    kernel = kernel/np.sum(kernel)
    return kernel

# Filtro gaussiano 
def escalaImagem(I, tipo_dst):

    I = np.float32(I)
    I_scaled = (I - np.min(I)) / (np.max(I)- np.min(I))

    if(tipo_dst == np.uint8):
        I_scaled = np.uint8(255*I_scaled)

    return I_scaled

def color_segmentation(I, ref_color, limiar):

    n_linhas, n_colunas, n_camadas = I.shape

    Ref = np.zeros((n_linhas, n_colunas, n_camadas), np.float32)
    Ref[:,:,0] = ref_color[0]
    Ref[:,:,1] = ref_color[1]
    Ref[:,:,2] = ref_color[2]

    D = np.sqrt(np.sum((np.float32(I) - Ref)**2, axis=2))

    I_bin = np.zeros((n_linhas, n_colunas), np.uint8)
    I_bin[D <= limiar] = 255

    return I_bin

# Função que realiza a operação de reconstrução morfológica
def imreconstruction(Mask, Marker):

    # Operação morfológica de reconstrução
    num_pixels_brancos = 0
    kernel = np.ones((3,3), np.float32)

    # critério de parada: quando não houver mais alteração na imagem Marker
    while num_pixels_brancos != np.sum(Marker):
        num_pixels_brancos = np.sum(Marker)
        Marker = cv2.dilate(Marker, kernel, iterations=1)
        Marker = cv2.bitwise_and(Mask, Marker)

    return Marker

# Função para remover borda
def imclearboard(I):

    Marker = I.copy()
    Marker[1:-1,1:-1] = 0

    I2 = imreconstruction(I, Marker)
    I3 = I - I2
    
    return I3

# Função para obter componentes conectados e analisar as regioes
def analisaRegioes(Ibin):

    infoRegioes = []

    #analise de componentes conectados
    num_labels, I_labels = cv2.connectedComponents(Ibin)

    # acessar cada componente conectado
    for n in np.arange(1, num_labels): # tem que considerar de 1 pois exclui-se o fundo
        
        # cria dicinário para armazenas as info do componente n
        dados_componente = dict()

        #imagem do componente com rótulo n
        Icomponente = np.uint8(I_labels == n) * 255
        dados_componente['imagem'] = Icomponente.copy()

        # definindo bound box
        y, x = np.where(Icomponente)
        ymin = np.min(y)
        xmin = np.min(x)
        xmax = np.max(x)
        ymax= np.max(y)
        
        p1 = np.array([xmin, ymin])
        p2 = np.array([xmax, ymax])

        dados_componente['bb_point1'] = p1
        dados_componente['bb_point2'] = p2

        #area
        m00 = mpq(Icomponente, 0, 0)
        dados_componente['area'] = m00

        # centroide
        m10 = mpq(Icomponente, 1,0)
        m01 = mpq(Icomponente, 0 ,1)
        xc = m10/m00
        yc = m01/m00
        dados_componente['centroide'] = np.array([xc, yc])

        # momentos centrai
        u11 = upq(Icomponente, 1, 1)
        u20 = upq(Icomponente, 2, 0)
        u02 = upq(Icomponente, 0, 2)
        dados_componente['u11'] = u11
        dados_componente['u20'] = u20
        dados_componente['u02'] = u02

        J = np.array([[u20, u11],[u11, u02]])
        Je = 4/m00 * J
        
        # determinacao dos autovalores e autovetores de matriz Je
        autovalores, autovetores = np.linalg.eig(Je)
        raios = -np.sort(-np.sqrt(autovalores))
        dados_componente['raios'] = raios
        dados_componente['excentricidade'] = raios[1]/raios[0]

        # pegando o autovetor associado ao maior autovalor
        pos = np.argmax(autovalores)
        vx = autovetores[0, pos]
        vy = autovetores[1, pos]
        orientacao = np.rad2deg(np.arctan2(vy, vx))
        dados_componente['orientacao'] = orientacao
        
        #extrai coordenadas dos pixels de borda
        countours, _ = cv2.findContours(Icomponente, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        x = countours[0][:, 0, 0]
        y = countours[0][:,0, 1]

        # perímetro
        N = len(x)
        perimetro = np.sqrt( ((y[-1] - y[0])**2) + ((x[-1] - x[0])**2))

        for n in np.arange(0, N-1):
            distancia= np.sqrt( ((y[n] - y[n+1])**2) + ((x[n] - x[n+1])**2))
            perimetro = perimetro + distancia

        dados_componente['perimetro'] = perimetro
        
        #circularidade
        m00 = mpq(Icomponente, 0, 0)
        circularidade = 4*np.pi*m00/(perimetro**2)
        dados_componente['circularidade'] = circularidade

        infoRegioes.append(dados_componente)
    return infoRegioes


# Função para obter os momentos
def mpq(Icomponente, p, q):

    y,x = np.where(Icomponente)
    momento = np.sum((x**p)*(y**q)) # multiplicação elemento a aelemtno

    return momento

# Função que obtem o centroide 
def upq(Icomponente, p, q):

    #centroide
    m00 = mpq(Icomponente, 0, 0)
    m10 = mpq(Icomponente, 1, 0)
    m01 = mpq(Icomponente, 0, 1)

    xc = m10/m00
    yc = m01/m00

    y,x = np.where(Icomponente)
    momento_central = np.sum(((x-xc)**p)*((y-yc)**q))
    return momento_central


# Função que obtem a similaridade
def similaridade(I1,I2, metrica):

    # Transformar para float
    I1 = np.float32(I1)/255
    I2 = np.float32(I2)/255

    if metrica == 'SAD':
        sim = np.sum(np.abs(I1-I2))
    if metrica == 'ZSSD':
        media_I1 = np.mean(I1)
        media_I2 = np.mean(I2)

        # Calcular a soma das diferenças ao quadrado
        sim = np.sum(((I1 - media_I1) - (I2 - media_I2)) ** 2)

    return sim


# funcao de supressao minima
def supressao_min(S, M, tamanho = 10):
    #y0 e x0 retornam onde esta o ponto
    y0,x0 = np.where(M)

    tamanho_janela = 10  # tamanho fixo da janela

    for k in range(0,x0.size):
        # Definir limites da janela usando np.clip para evitar sair dos limites
        y_start = (y0[k] - tamanho_janela)
        y_end = (y0[k] + tamanho_janela)
        x_start = (x0[k] - tamanho_janela)
        x_end = (x0[k] + tamanho_janela)
        
        teste = np.min(S[y_start:y_end, x_start:x_end])
        if S[y0[k], x0[k]] != teste:
            M[y0[k], x0[k]] = 0

    cv2.imshow('Imagem binaria2', M)
    y1,x1 = np.where(M)
    return y1,x1,M
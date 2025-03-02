import numpy as np
import cv2
import visaoComputacional as visco

# --------------- Fazendo homografia -----------------------
I1 = cv2.imread('./imagens/im5.png')
cv2.imshow('Imagem Original', I1)

# Definir cor de referência
cor_referencia = np.array([120, 180, 106])

# Separa os componentes RGB da cor de referência
br = cor_referencia[0]
gr = cor_referencia[1]
rr = cor_referencia[2]

# Separação dos canais de cor
b = I1[:, :, 0]
g = I1[:, :, 1]
r = I1[:, :, 2]
#n_linhas, n_colunas, _ = I1.shape
delta = 50

# Criação da matriz binária
dist = np.sqrt((b - br)**2 + (g - gr)**2 + (r - rr)**2)
M = np.where(dist <= delta, 255, 0).astype(np.uint8)
cv2.imshow('Componentes com Bounding Boxes', M)

# Extração de regiões e contorno
infoRegioes = visco.analisaRegioes(M)

# obtendo o centroide e dando sorted()
dic = {i: regiao['centroide'] for i, regiao in enumerate(infoRegioes)}
dic_sorted_by_keys = dict(sorted(dic.items()))


# Separação em listas as coordenadas dos centróides
lista_pixel_menor, lista_pixel_maior = [], []
for item in dic.items():
    if item[1][0] < 380:
        lista_pixel_menor.append(item[1])  # Adiciona só as coordenadas do centróide
    else:
        lista_pixel_maior.append(item[1])  # Adiciona só as coordenadas do centróide

# definindo tamanho final da imagem
n_linhas, n_colunas = 331, 666

# Verifica se há elementos suficientes
if len(lista_pixel_menor) > 1 and len(lista_pixel_maior) > 1:
    # Definir pontos para homografia
    pts_org = np.array([
        lista_pixel_menor[0],  # Primeiro ponto menor
        lista_pixel_menor[1],  # Segundo ponto menor
        lista_pixel_maior[1],  # Segundo ponto maior
        lista_pixel_maior[0]   # Primeiro ponto maior
    ])

    #Novos pontos
    pts_dst = np.array([[0, 0], [0, n_linhas-1], [n_colunas-1, n_linhas-1], [n_colunas-1, 0]])

    # Cálculo  e Aplicação da homografia
    H = visco.homografia(pts_org, pts_dst)
    I3 = cv2.warpPerspective(I1, H, (n_colunas, n_linhas))
    cv2.imshow('Resultado da Homografia', I3)

else:
   pass



# -------------------- retirando a borda --------------
# Aplicação do limiar binário com Otsu
I3 = cv2.cvtColor(I3, cv2.COLOR_BGR2GRAY)
_, I4 = cv2.threshold(I3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

I5 = 255 - I4

Mask = I5.copy()
Marker = Mask.copy()
Marker[1:-1, 1:-1] = 0

# chamando função para realizar a operação de dilatação e bitwise_and
I_reco = visco.imreconstruction(Mask, Marker)

I5 = I5 - I_reco
cv2.imshow('Imagem com elementos da borda retirados', I5)

# obtendo as regiões da imagem I5
infoRegioes2 = visco.analisaRegioes(I5)


#--------------- Separando tempaltes-----------------------

# Carregando imagem dos templates
template = cv2.imread('./imagens/template_letras.png')
cv2.imshow('template', template)

# Convertendo de BGR para escala de cinza
temp1 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#Realizando limiar de otsu e obtendo o negativo
_, temp2 = cv2.threshold(temp1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
temp3 = 255 - temp2
cv2.imshow('template3', temp3)

#Obtendo os componentes coenctados e organizando os mesmos de acordo com coordenada x
infoRegioes  = visco.analisaRegioes(temp3)
infoRegioes = sorted(infoRegioes, key=lambda regiao: regiao['centroide'][0])


# Realizando recorte dos componentes
dic = {}
lista = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
    'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', 'W']

for i, letra in enumerate(lista):

    # Obter os pontos da bounding box
    p1 = infoRegioes[i]['bb_point1']
    p2 = infoRegioes[i]['bb_point2']
    

    imagem_letra = template[p1[1]:p2[1], p1[0]:p2[0]]  # Recorte da imagem da letra
    dic[letra] = imagem_letra  # Armazenar no dicionário com a letra como chave

cv2.imshow('Letra I', dic['O'])


# #----------------- Comparando tamanhos ------------------
lista_seq = []
lista_centro_x = []
lista_centro_y = []

#print(len(infoRegioes2))

for letra, imagem in dic.items():    
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # realizando resize no template
    imagem_resized = cv2.resize(imagem, (50, 50))

    for idx, regiao in enumerate(infoRegioes2):
        bb_point1 = regiao['bb_point1']
        bb_point2 = regiao['bb_point2']

        # Obtendo area para realizar o threshold e calcular a similaridade
        area = regiao['area']

        if area > 150:
            imagem_letra = I5[bb_point1[1]:bb_point2[1], bb_point1[0]:bb_point2[0]]   

            # realizando resize no componente da imagem       
            imagem_letra_resized = cv2.resize(imagem_letra, (50, 50))

            # calculo da similaridade utilizando o método ZSSD
            I1 = 255 - imagem_letra_resized 
            I1 = np.float32(I1) / 255
            I2 = np.float32(imagem_resized) / 255

            media_I1 = np.mean(I1)
            media_I2 = np.mean(I2)

            sim = np.sum(((I1 - media_I1) - (I2 - media_I2)) ** 2)
            sim = sim*1.5

            # Classificar a letra cmom base no valor de similaridade
            if sim < 365:
                lista_seq.append(letra)
                lista_centro_x.append(infoRegioes2[idx]['centroide'][0])
                lista_centro_y.append(infoRegioes2[idx]['centroide'][1])
            else:
                pass
    
 

 #----------------------------- Apresentando a mensagem -----------------

# Tolerâncias para agrupamento
tolerancia_y = 65 # Intervalo de pixels para agrupar em "níveis" verticais
tolerancia_x = 50 # Intervalo de pixels para adicionar espaços entre letras

# Combinando as letras e coordenadas em uma lista de tuplas
letras_com_coords = list(zip(lista_seq, lista_centro_x, lista_centro_y))

# Ordenando primeiro pela coordenada Y em blocos e depois pela coordenada X dentro de cada nível
letras_com_coords.sort(key=lambda item: (item[2] // tolerancia_y, item[1]))

# Inicializa variáveis para armazenar a saída formatada
texto_formatado = []
nivel_atual = letras_com_coords[0][2] // tolerancia_y
linha_texto = []
ultimo_x = int(letras_com_coords[0][1])  # Converte para int

# Itera pelas letras ordenadas para agrupar
for letra, x, y in letras_com_coords:
    x = int(x) 
    y = int(y) 
    
    # Verifica se estamos no mesmo nível (usando tolerância em Y)
    if y // tolerancia_y == nivel_atual:
        # Adiciona letras com espaços se necessário
        if linha_texto:
            # Adiciona espaço proporcional entre as letras
            if x - ultimo_x > tolerancia_x:
                espacos = " " * ((x - ultimo_x) // tolerancia_x)
                linha_texto.append(espacos)
        linha_texto.append(letra)
        ultimo_x = x 
    else:
        # Finaliza a linha atual e passa para o próximo nível
        texto_formatado.append("".join(linha_texto).strip())
        linha_texto = [letra]
        nivel_atual = y // tolerancia_y
        ultimo_x = x 

# Adiciona a última linha ao texto formatado
if linha_texto: 
    texto_formatado.append("".join(linha_texto).strip())

# Imprime o resultado final de forma formatada
for linha in texto_formatado:
    print(linha)

cv2.waitKey(0)
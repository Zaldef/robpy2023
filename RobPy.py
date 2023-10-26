import numpy as np
import matplotlib.pyplot as plot


# Parte 1


def cria_vetor3(vlist: list[float]) -> np.ndarray:
    """
    Função que recebe uma lista e cria um vetor (np.ndarray) coluna de 3 elementos
    :param vlist: Lista com as componentes [vx, vy, vz] do vetor desejado
    :return: np.ndarray: vetor (3, 1) com os valores desejados
    """
    if len(vlist) != 3:
        raise ValueError('A lista deve possuir 3 posições')
    return np.asarray([vlist]).T


def checa_vetor3(v: np.ndarray) -> None:
    """
    Função sem retorno, mas que deve gerar uma exceção caso o tamanho do vetor não seja (3, 1)
    :param v:
    :return:
    """
    if v.shape != (3, 1):
        raise ValueError('O vetor deveria ser 3X1')


def produto_escalar(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calcula o produto escalar entre dois vetores.
    :param v1: vetor (np.ndarray) coluna de 3 elementos
    :param v2: vetor (np.ndarray) coluna de 3 elementos
    :return: escalar: resultado de v1.v2
    """
    checa_vetor3(v1)
    checa_vetor3(v2)
    aux = v1.T @ v2
    return float(aux[0][0])


def norma_vetor(v: np.ndarray) -> float:
    """
    Calcula a norma de um vetor
    :param v: vetor (np.ndarray) coluna de 3 elementos
    :return: escalar: norma do vetor
    """
    return np.sqrt(produto_escalar(v, v))


def tamanho_proj_vetores(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calcula o tamanho da projeção de v1 sobre v2 (escalar)
    :param v1: vetor (np.ndarray) coluna de 3 elementos
    :param v2: vetor (np.ndarray) coluna de 3 elementos
    :return: escalar: tamanho da projeção de v1 sobre v2
    """
    return norma_vetor(proj_vetores(v1, v2))


def proj_vetores(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calcula o vetor projeção de v1 sobre v2
    :param v1: vetor (np.ndarray) coluna de 3 elementos
    :param v2: vetor (np.ndarray) coluna de 3 elementos
    :return: vetor (np.ndarray) coluna de 3 elementos com o resultado da projeção
    """
    checa_vetor3(v1)
    checa_vetor3(v2)
    aux = produto_escalar(v1, v2) / produto_escalar(v2, v2)
    return aux * v2


def ang_vetores(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calcula o ângulo entre dois vetores em radianos.
    :param v1: vetor (np.ndarray) coluna de 3 elementos
    :param v2: vetor (np.ndarray) coluna de 3 elementos
    :return: escalar: ângulo em radianos
    """
    return np.arccos(produto_escalar(v1, v2) / (norma_vetor(v1) * norma_vetor(v2)))


def produto_vetorial(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calcula o produto vetorial v1 x v2.
    :param v1: vetor (np.ndarray) coluna de 3 elementos
    :param v2: vetor (np.ndarray) coluna de 3 elementos
    :return: vetor (np.ndarray) coluna de 3 elementos com o resultado de v1 x v2
    """
    checa_vetor3(v1)
    checa_vetor3(v2)
    
    return np.asarray([
        [v1[1]*v2[2] - v1[2]*v2[1]],
        [v1[2]*v2[0] - v1[0]*v2[2]],
        [v1[0]*v2[1] - v1[1]*v2[0]]
    ])

# Parte 2


def plota_vetor3(v: np.ndarray,
                 *args,
                 vo: np.ndarray = np.zeros([3, 1]),
                 color='b',
                 **kwargs) -> list:
    """
    Utiliza o pacote matplotlib.plotpy para plotar um vetor em um diagrama 3D. É necessário utilizar eixos criados com o
    comando matplotlib.plotly.axis(projection='3d').
    Cuidado: os eixos 3d no matplotlib não possuem escala fixa, portanto os gráficos podem parecer distorcidos.
    :param color:
    :param v: vetor a ser plotado.
    :param args: parâmetros padrão do plot
    :param vo: vetor que vai da origem do sistema de coordenadas até a base do vetor a ser plotado. É [0, 0, 0].T por
    padrão.
    :param kwargs: parâmetros padrão do plot.
    :return: lista de elementos de linha do vetor plotado.
    """

    a = plot.plot([vo[0][0], v[0][0] + vo[0][0]],
                  [vo[1][0], v[1][0] + vo[1][0]],
                  [vo[2][0], v[2][0] + vo[2][0]],
                  linewidth=4,
                  color=color)
    b = plot.plot(v[0][0] + vo[0][0],
                  v[1][0] + vo[1][0],
                  v[2][0] + vo[2][0],
                  marker='>',
                  markersize=15,
                  color=color)

    return [a, b]


def matriz_rotacao_x(theta: float) -> np.ndarray:
    """
    Função que retorna a matriz de rotação que leva um vetor de uma base 'a' para uma base 'b' gerada a partir da
    rotação da base 'a' em torno do eixo x por um ângulo 'theta' positivo em radianos.
    :param theta: ângulo de rotação
    :return: matriz de rotação
    """
    s = np.sin(theta)
    c = np.cos(theta)

    return np.asarray([ [1, 0, 0],
                        [0, c, s],
                        [0, -s, c]])


def matriz_rotacao_y(theta: float) -> np.ndarray:
    """
    Função que retorna a matriz de rotação que leva um vetor de uma base 'a' para uma base 'b' gerada a partir da
    rotação da base 'a' em torno do eixo y por um ângulo 'theta' positivo em radianos.
    :param theta: ângulo de rotação
    :return: matriz de rotação
    """
    s = np.sin(theta)
    c = np.cos(theta)

    return np.asarray([ [c, 0, -s],
                        [0, 1, 0],
                        [s, 0, c]])


def matriz_rotacao_z(theta: float) -> np.ndarray:
    """
    Função que retorna a matriz de rotação que leva um vetor de uma base 'a' para uma base 'b' gerada a partir da
    rotação da base 'a' em torno do eixo z por um ângulo 'theta' positivo em radianos.
    :param theta: ângulo de rotação
    :return: matriz de rotação
    """
    s = np.sin(theta)
    c = np.cos(theta)

    return np.asarray([ [c, s, 0],
                        [-s, c, 0],
                        [0, 0, 1]])


# Parte 3


def checa_vetor4(v: np.ndarray) -> None:
    """
    Verifica se um vetor é um vetor de 4 linhas e uma coluna. Caso não seja, levanta uma exceção.
    :param v: vetor a verificar
    :return: nenhum.
    """
    if v.shape != (4, 1):
        raise ValueError('O vetor deveria ser 4X1')



def checa_matriz33(m: np.ndarray) -> None:
    """
    Verifica se uma matriz possui 3 linhas e 3 colunas. Caso não seja, levanta uma exceção.
    :param m: matriz a verificar
    :return: nenhum.
    """
    if m.shape != (3, 3):
        raise ValueError('A matriz deveria ser 3X3')


def checa_matriz44(m: np.ndarray) -> None:
    """
    Verifica se uma matriz possui 4 linhas e 4 colunas. Caso não seja, levanta uma exceção.
    :param m: matriz a verificar
    :return: nenhum.
    """
    if m.shape != (4, 4):
        raise ValueError('A matriz deveria ser 4X4')


def cria_vetor4(v3: np.ndarray) -> np.ndarray:
    """
    Recebe um vetor (3, 1) e cria um vetor (4, 1) após concatenar o valor 1 ao final deste vetor.
    :param v3:
    :return:
    """
#    return np.vstack((v3, 1))
    return np.append(v3, np.asarray([[1]]), axis=0)

def checa_matriz_rotacao(m3: np.ndarray, det_tol: float = 0.01) -> None:
    """
    Recebe uma matriz (3, 3), verifica suas dimensões e verifica se seu determinante é 1, pois matrizes de rotação devem
    possuir determinante unitário independente do número de rotações realizadas.
    :param m3: matriz a verificar
    :param det_tol: tolerância do valor do determinante
    :return: não há
    """
    
    if det_tol < 0:
        raise ValueError('O valor da tolerância do determinante deve ser positivo.')

    checa_matriz33(m3)
    erro = np.abs(1-np.linalg.det(m3))
    if erro > det_tol:
        raise ValueError('Pelo valor do determinante, esta matriz não é de rotação')



def cria_operador4(m_rot_b_a: np.ndarray = np.eye(3), v_o_a: np.ndarray = np.zeros([3, 1]), det_tol: float = 0.01) \
        -> np.ndarray:
    """
    Cria um operador de construção de vetores por transformação homogênea (4, 4) que recebe um vetor origem escrito na
    base 'a' e uma matriz de rotação que leva da base 'b' para a base 'a'.
    :param m_rot_b_a: matriz de rotação associada
    :param v_o_a: vetor origem associado
    :param det_tol:
    :return:
    """
    checa_matriz_rotacao(m_rot_b_a, det_tol)
    checa_vetor3(v_o_a)

    T = np.append(m_rot_b_a, v_o_a, axis=1) # adiciona no eixo das colunas
    T = np.append(T, np.asarray([[0,0,0,1]]), axis=0) # eixo das linhas

    return T


def constroi_vetor(v_b: np.ndarray,
                   m_rot_b_a: np.ndarray = np.eye(3),
                   v_o_a: np.ndarray = np.zeros([3, 1]),
                   det_tol: float = 0.01) -> np.ndarray:
    """
    Recebe um vetor v_b escrito na base 'b'. A partir da matriz de rotação m_rot_b_a e do vetor origem v_o_a, constroi o
    operador de transformação homogênea que constrói um vetor na base 'a' que aponta para o mesmo ponto que o vetor v_b.
    :param v_b: vetor referência na base 'b'
    :param m_rot_b_a: matriz de rotação que leva de 'b' a 'a'
    :param v_o_a: vetor origem da base 'b' escrito na base 'a'
    :param det_tol: tolerância do determinante
    :return: vetor (3, 1) na base a
    """
    checa_vetor3(v_b)
    T = cria_operador4(m_rot_b_a=m_rot_b_a, v_o_a=v_o_a, det_tol=det_tol)

    v_b4 = cria_vetor4(v_b)

    v_a4 = T @ v_b4

    return v_a4[0:3][:]


# Parte 4


def __distancia_entre_retas_np(po1: np.ndarray, vs1: np.ndarray, po2: np.ndarray, vs2: np.ndarray) -> float:
    """
    *** FUNÇÃO INTERNA AO MÓDULO ***
    Calcula a distância entre duas retas não paralelas no espaço.
    Um ponto na reta i é dado por: Pi = poi + vsi*t, sendo t um parâmetro livre.
    'distancia_entre_retas'
    :param po1: Vetor posição de um ponto de referência na reta 1
    :param vs1: Vetor orientação da reta 1
    :param po2: Vetor posição de um ponto de referência na reta 2
    :param vs2: Vetor orientação da reta 1
    :return: distância entre as retas (float, positivo ou nulo)
    """
    checa_vetor3(vs1)
    checa_vetor3(vs2)
    checa_vetor3(po1)
    checa_vetor3(po2)
    v1 = po1 - po2
    v2 = produto_vetorial(vs1, vs2)
    v2 = v2 / norma_vetor(v2)
    return norma_vetor(proj_vetores(v1, v2))


def __distancia_entre_retas_p(po1: np.ndarray, po2: np.ndarray, vs: np.ndarray) -> float:
    """
    *** FUNÇÃO INTERNA AO MÓDULO ***
    Calcula a distância entre duas retas paralelas no espaço.
    Um ponto na reta i será dado por Pi = poi + vs*t, sendo t um parâmetro independente.
    A verificação sobre o tamanho dos vetores será feita na função pública 'distancia_entre_retas'
    :param po1: Posição de um ponto de referência na reta 1
    :param po2: Posição de um ponto de referência na reta 2
    :param vs: Vetor direção de ambas as retas
    :return: distância entre as retas (float, não negativo)
    """
    checa_vetor3(po1)
    checa_vetor3(po2)
    checa_vetor3(vs)
    
    v1 = po1 - po2
    return norma_vetor(v1 - proj_vetores(v1, vs))


def distancia_entre_retas(po1: np.ndarray, vs1: np.ndarray, po2: np.ndarray, vs2: np.ndarray, angtol=1e-3) -> float:
    """
    Calcula a distância entre duas retas no espaço.
    Um ponto na reta i é dado por: Pi = poi + vsi*t, sendo t um parâmetro livre.
    :param po1: Vetor posição de um ponto de referência na reta 1
    :param vs1: Vetor orientação da reta 1
    :param po2: Vetor posição de um ponto de referência na reta 2
    :param vs2: Vetor orientação da reta 1
    :param angtol: Tolerância de ângulo entre as retas para decidir se são paralelas
    :return: Distância entre as retas (float, positivo ou nulo)
    """

    if angtol < 0:
        raise ValueError('A tolerância angular deve ser um valor não negativo')

    ang = np.abs(ang_vetores(vs1, vs2))
    if ang < angtol or np.abs(ang-np.pi) < angtol:
        return __distancia_entre_retas_p(po1, po2, vs1)
    else:
        return __distancia_entre_retas_np(po1, vs1, po2, vs2)



def __eixo_reta_12_np(po1: np.ndarray, vs1: np.ndarray, po2: np.ndarray, vs2: np.ndarray) -> np.ndarray:
    """
    *** FUNÇÃO INTERNA AO MÓDULO ***
    Calcula um vetor unitário perpendicular às retas 1 e 2 que aponta necessariamente da reta 1 à reta 2. As retas não
    podem ser paralelas.
    :param po1: Vetor posição de um ponto de referência na reta 1
    :param vs1: Vetor orientação da reta 1
    :param po2: Vetor posição de um ponto de referência na reta 2
    :param vs2: Vetor orientação da reta 1
    :return: vetor unitário que aponta da reta 1 à reta 2
    """
    e = produto_vetorial(vs1, vs2)
    e = e / norma_vetor(e)
    d = po2 - po1
    return e * np.sign(produto_escalar(d,e))



def __eixo_reta_12_p(po1: np.ndarray, po2: np.ndarray, vs: np.ndarray) -> float:
    """
    *** FUNÇÃO INTERNA AO MÓDULO ***
    Calcula um vetor unitário que vai da reta 1 à reta 2 necessariamente. As retas devem ser paralelas
    :param po1: Vetor posição de um ponto de referência na reta 1
    :param po2: Vetor posição de um ponto de referência na reta 2
    :param vs: Vetor direção de ambas as retas
    :return: vetor unitário que aponta da reta 1 à reta 2
    """
    d = po2 - po1
    dn = d - proj_vetores(d, vs)
    return dn / norma_vetor(dn)
    


def eixo_reta_12(po1: np.ndarray, vs1: np.ndarray, po2: np.ndarray, vs2: np.ndarray, angtol=1e-3) -> np.ndarray:
    """
    Calcula um vetor unitário que aponta da reta 1 à reta 2, independente de sua orientação.
    :param po1: Vetor posição de um ponto de referência na reta 1
    :param vs1: Vetor orientação da reta 1
    :param po2: Vetor posição de um ponto de referência na reta 2
    :param vs2: Vetor orientação da reta 1
    :param angtol: Tolerância de ângulo entre as retas para decidir se são paralelas
    :return: vetor unitário que aponta da reta 1 à reta 2
    """
    if angtol < 0:
        raise ValueError('A tolerância angular deve ser um valor não negativo')

    ang = np.abs(ang_vetores(vs1, vs2))
    if ang < angtol or np.abs(ang-np.pi) < angtol:
        return __eixo_reta_12_p(po1, po2, vs1)
    else:
        return __eixo_reta_12_np(po1, vs1, po2, vs2)


def ang_twist_dir_nc_rad(po1: np.ndarray, vs1: np.ndarray, po2: np.ndarray, vs2: np.ndarray, angtol=1e-3) -> float:
    """
    Função que calcula o ângulo de torção de um link em radianos no caso em que os eixos das juntas adjacentes não sejam
    concorrentes.
    :param po1: Vetor posição de um ponto de referência na reta 1 (eixo da junta 1)
    :param vs1: Vetor orientação da reta 1
    :param po2: Vetor posição de um ponto de referência na reta 2 (eixo da junta 2)
    :param vs2: Vetor orientação da reta 1
    :param angtol: Tolerância de ângulo entre as retas para decidir se são paralelas
    :return: Ângulo de torção do link com sinal direcional
    """
    pvs = produto_vetorial(vs1, vs2)
    e12 = eixo_reta_12(po1, vs1, po2, vs2, angtol=angtol)

    pe = produto_escalar(pvs, e12)

    return ang_vetores(vs1, vs2) * np.sign(pe)


def ang_twist_dir_ref_rad(vs1: np.ndarray, vs2: np.ndarray, vref: np.ndarray, projtol: float = 1e-3) -> float:
    """
    Calcula o ângulo de torção de um link para o caso de eixos concorrentes. Neste caso deve-se passar um eixo de
    referência vref para que se defina o sentido positivo da rotação de torção.
    :param vs1: Vetor orientação da reta 1 (eixo da junta 1)
    :param vs2: Vetor orientação da reta 1 (eixo da junta 2)
    :param vref: Eixo de referência para a definição da direção positiva da rotação
    :param projtol: Tolerância da projeção de vs1 e vs2 sobre vref para verificar se são perpendiculares
    :return: Ângulo de torção do link com sinal direcional
    """
    pvs = produto_vetorial(vs1, vs2)
    pe = produto_escalar(pvs, vref)

    if norma_vetor(vref - proj_vetores(vref, pvs) * np.sign(pe)) > projtol:
        raise ValueError('O vetor referencia na possui a orientacao apropriada')

    return ang_vetores(vs1, vs2) * np.sign(pe)

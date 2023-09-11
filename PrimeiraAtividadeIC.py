import pandas as pd
import seaborn as sb
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#definindo o numero de clusters
def calculate_wcss(data):
    wcss = []
    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)

    return wcss

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 =i + 2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)

    return distances.index(max(distances)) + 2


#importando a base
base = pd.read_csv("data_banknote_authentication.csv")

#estatistica descritiva basica
estatistica = base.describe()

#visualizacao do pairplot
sb.pairplot(base,hue="Veracidade”)

#separação do conjunto de input e target (previsores e classe)
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

#normalização dos dados
scaler = StandardScaler()
previsores_normalizado = scaler.fit_transform(previsores)

#calculando a soma dos quadrados para as 19 quantidade de clusters
sum_of_squares = calculate_wcss(previsores)

#calculando a quantidade ótima de clusters
n = optimal_number_of_clusters(sum_of_squares)

#clusterização
x = previsores_normalizado

kmeans = KMeans(n_clusters = n, random_state = 0)

kmeans . fit(x)

y = kmeans.labels_

base[ 'k-classes'] = y

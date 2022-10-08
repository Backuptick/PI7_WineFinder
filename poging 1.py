import pandas
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from random import randint, random, choice, gauss

def warn(*args, **kwargs):                                                                                         #haalt de warnings weg tijdens het runnen van het programma die door sklearn library erin worden gezet
    pass
import warnings
warnings.warn = warn

data = pandas.read_csv("wine.data", header = None)                                                                 #laadt data in het bestand en print die vervolgens in de terminal
print(data.iloc[:, 1:])                                                                                            #haalt eerste kolom weg (de 1-3, omdat dat de output is)
X = data.iloc[:, 1:].values.tolist()
y = data.iloc[:, 0].values.tolist()

def geneticAlgorithm(generatieSize, aantalIteraties, startHyperparameters):
    generatie = [changeFunction(startHyperparameters) for _ in range(generatieSize)]                                #Maakt eerste generatie aan voor alleen eerste keer.
    for i in range(aantalIteraties):
        scores = [(runNeuralNetwork(x), x) for x in generatie]
        scores.sort(key = (lambda x: x[0]), reverse = True)                                                         #sorteert het linkerdeel van de tuple en zet deze in een volgorde
        nieuweGeneratie = []
        for _ in range(generatieSize):
            daddy = scores[randint(0, randint(0, generatieSize-1))][1]                                              #pakt rechterdeel van een random tuple in scores (de hyperparameters)
            mommy = scores[randint(0, randint(0, generatieSize-1))][1]
            timmie = crossOver(daddy, mommy)
            timmie = changeFunction(timmie)
            nieuweGeneratie.append(timmie)
        generatie = nieuweGeneratie
        print("Iteratie", i, "/", aantalIteraties, "score:", scores[0][0], "/", "average:", sum([x[0] for x in scores]) / len(scores))
    scores = [(runNeuralNetwork(x), x) for x in generatie]
    scores.sort(key = (lambda x: x[0]), reverse = True)
    print("Iteratie", aantalIteraties, "/", aantalIteraties, "score:", scores[0][0], "/", "average:", sum([x[0] for x in scores]) / len(scores))
    return scores[0]

def crossOver(daddy: dict, mommy: dict):                                                                            #voor iedere key in daddy genereer een random getal (0 of 1) als het 1 is, geef je die aan timmie en als het 0 is geef je die van mommy aan timmie 
    timmie = {}
    for key in daddy.keys():
        if randint(0, 1) == 1:
            timmie[key] = daddy[key]
        else:
            timmie[key] = mommy[key]
    return timmie






#Maakt functie voor het runnen van het neurale netwerk volgens code site

def runNeuralNetwork(hyperParameters: dict):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
    clf = MLPClassifier(
        hidden_layer_sizes = hyperParameters["hidden_layer_sizes"], 
        activation = hyperParameters["activation"],
        alpha = hyperParameters["alpha"],
        learning_rate_init = hyperParameters["learning_rate_init"],
        max_iter = hyperParameters["max_iter"]
        ).fit(X_train, y_train)
    return clf.score(X_test, y_test)

#Maakt functie voor verandering van de hyper-parameters.

def changeFunction(hyperParameters: dict):
    output = hyperParameters.copy()
    parameterToChange = randint(1, 5)                                                                               #past aantal parameters aan
    if parameterToChange == 1:
        if len(output["hidden_layer_sizes"]) < 5 and randint(1, 2) == 1:                                            #past aantal layers aan
            output["hidden_layer_sizes"] = tuple(list(hyperParameters["hidden_layer_sizes"]) + [randint(3, 70)])           #voegt layers toe met een range van een grootte van nodes
        else:                                                                                                       #past aantal nodes in een layer aan
            layerIndex = randint(0, len(output["hidden_layer_sizes"])-1)
            nodes = output["hidden_layer_sizes"][layerIndex]
            if randint(1, 2) == 1 and nodes > 3:
                nodes -= min(randint(1, 5), nodes - 3)
            elif nodes < 70: 
                nodes += min(randint(1, 5), 70 - nodes)
            else: 
                nodes -= min(randint(1, 5), nodes - 3)
            layer = list(output["hidden_layer_sizes"])
            layer[layerIndex] = nodes
            output["hidden_layer_sizes"] = tuple(layer) 

    elif parameterToChange == 2:
        activationFunctions = ["identity", "logistic", "tanh", "relu"]
        activationFunctions.remove(output["activation"])
        output["activation"] = choice(activationFunctions) 

    elif parameterToChange == 3:
        output["alpha"] = max(gauss(0, 0.001) + output["alpha"], 0.00001)

    elif parameterToChange == 4:
        output["learning_rate_init"] = max(gauss(0, 0.01) + output["learning_rate_init"], 0.0001)

    elif parameterToChange == 5:
        if output["max_iter"] < 100:
            output["max_iter"] += randint(1, 5)                         # als < 100 dan een getal tussen 1 en 5 optellen
        elif output["max_iter"] > 300:
            output["max_iter"] += randint(-5, -1)                       # als > 300 dan een getal tussen -1 en -5 optellen
        else:
            output["max_iter"] += randint(-5, 5)

    return output

startHyperParameters = {
    "hidden_layer_sizes": (100, ),
    "activation": "relu",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "max_iter": 200
}
size = 100
iterations = 20

result = geneticAlgorithm(size, iterations, startHyperParameters)
print("score:", result[0], "hyperParameters:", result [1] )



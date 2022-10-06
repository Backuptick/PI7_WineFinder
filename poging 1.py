from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from random import randint, random, choice, gauss

X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
clf = MLPClassifier(random_state=2, max_iter=500).fit(X_train, y_train)
clf.predict_proba(X_test[:1])
clf.predict(X_test[:5, :])
clf.score(X_test, y_test)
print(clf.score(X_test, y_test))

#Maakt functie voor verandering van de hyper-parameters.

def changeFunction(hyperParameters: dict):
    output = hyperParameters.copy()
    parameterToChange = randint(1, 6)                                                               #past aantal parameters aan
    if parameterToChange == 1:
        if len(output["hidden_layer_sizes"]) < 5 and randint(1, 2) == 1:                            #past aantal layers aan
            output["hidden_layer_sizes"] = hyperParameters["hidden_layer_sizes"] + randint(3, 70)   #voegt layers toe met een range van een grootte van nodes
        else:                                                                                       #past aantal nodes in een layer aan
            layer = randint(0, len(output["hidden_layer_sizes"])-1)
            nodes = output["hidden_layer_sizes"][layer]
            if randint(1, 2) == 1 and nodes > 3:
                nodes -= min(randint(1, 5), nodes - 3)
            elif nodes < 70: 
                nodes += min(randint(1, 5), 70 - nodes)
            else: 
                nodes -= min(randint(1, 5), nodes - 3)
            output["hidden_layer_sizes"][layer] = nodes

    elif parameterToChange == 2:
        activationFunctions = ["identity", "logistic", "tanh", "relu"]
        activationFunctions.remove(output["activation"])
        output["activation"] = random.choice(activationFunctions) 

    elif parameterToChange == 3:
        output["alpha"] += gauss(0, 0.001)

    elif parameterToChange == 4:
        output["learning_rate_init"] += gauss(0, 0.01)

    elif parameterToChange == 5:
        

    elif parameterToChange == 6:



hyperParamaters = {
    "hidden_layer_sizes": (100, ),
    "activation": "relu",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "batch_size": "auto",
    "max_iter": 200
}
hyperParameters.max_iter

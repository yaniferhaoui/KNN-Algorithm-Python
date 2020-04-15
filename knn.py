#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:47:16 2020

@author: Manon Foulon & Yani Ferhaoui
"""
import operator
import numpy
    
# Methods
def distance4D(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2 + (a[3] - b[3])**2)**(1/2)

def knn(learned_values, test_values, k):
    
    euclidean_distance = []
    for i in range(len(learned_values)):
        d = distance4D(test_values, learned_values[i])
        euclidean_distance.append([i, d])
    euclidean_distance.sort(key = operator.itemgetter(1))

    result = []    
    for i in range(k):
        j = euclidean_distance[i][0]
        result.append(learned_values[j])
        
    flowers = {"Iris-setosa" : 0, "Iris-versicolor" : 0, "Iris-virginica" : 0}
    for flower in result:
        flowers[flower[4]] = flowers[flower[4]] + 1
    
    return next(iter(sorted(flowers.items(), key = operator.itemgetter(1), reverse = True)))[0]

#
# Begin Tests
#
with open("iris.data", "r", encoding = "UTF-8") as file:
    lines = [i for i in file.read().splitlines() if i] # To remove empty strings 

iris = []
learning = []
test = []
confusions = []

for line in lines:
    line = line.split(",")
    iris.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), str(line[4])])    

# Fill Test and Learning arrays
for i in range(25):
    learning.append(iris[i])
    
for i in range(25, 50):
    test.append(iris[i])
    
for i in range(50, 75):
    learning.append(iris[i])
    
for i in range(75, 100):
    test.append(iris[i])
    
for i in range(100, 125):
    learning.append(iris[i])
    
for i in range(125, 150):
    test.append(iris[i])  

for k in range(1, 76):
    
    confusion = [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ]
    
    for flower in test:
        
        res = knn(learning, flower, k)
        
        if res == "Iris-setosa" and flower[4] == "Iris-setosa":
            confusion[0][0] = confusion[0][0]+1
            
        elif res == "Iris-setosa" and flower[4] == "Iris-versicolor":
            confusion[0][1] = confusion[0][1]+1
            
        elif res == "Iris-setosa" and flower[4] == "Iris-virginica":
            confusion[0][2] = confusion[0][2]+1
            
        elif res == "Iris-versicolor" and flower[4] == "Iris-setosa":
            confusion[1][0] = confusion[1][0]+1
            
        elif res == "Iris-versicolor" and flower[4] == "Iris-versicolor":
            confusion[1][1] = confusion[1][1]+1
            
        elif res == "Iris-versicolor" and flower[4] == "Iris-virginica":
            confusion[1][2] = confusion[1][2]+1
            
        elif res == "Iris-virginica" and flower[4] == "Iris-setosa":
            confusion[2][0] = confusion[2][0]+1
            
        elif res == "Iris-virginica" and flower[4] == "Iris-versicolor":
            confusion[2][1] = confusion[2][1]+1
            
        elif res == "Iris-virginica" and flower[4] == "Iris-virginica":
            confusion[2][2] = confusion[2][2]+1
            
        diagonal_sum = confusion[0][0] + confusion[1][1] + confusion[2][2]
        
    confusions.append([confusion, diagonal_sum, k])
    
confusions.sort(key = operator.itemgetter(1))
for confusion in confusions:
    diagonal_sum = confusion[1]
    k = confusion[2]
    print("k = %d => %d successful tests out of 75 : \n%s \n" %(k, diagonal_sum, numpy.matrix(confusion[0])))

best = confusions[len(confusions) - 1]
print("The best k = %d with %d successful tests out of 75!" %(best[2], best[1]))

# - Q1
# Pondérer chaque voisin en fonction de son éloignement de l'individu
#
   
# - Q2
# Retourner le % de de rapprochement avec chaque type de fleur dans le but de retourner une quantité continue
#

# - Q6
# Nous pouvons remarquer que le taux d'erreurs est minimal pour k = 17 et 18
#
##################### IMPORTS #####################
import asyncio
from js import document
from io import BytesIO, TextIOWrapper
from pyscript import when, display

import numpy as np
import pandas as pd
import csv
import time
import math
import random
import operator
import matplotlib.pyplot as plt
from sklearn import svm

train_counts = []
test_counts = []
train_data = []
test_data = []

##################### DOM EDIT #####################
toast = document.querySelector("#toast")
toast_text = document.querySelector("#toast-text")
toast.classList.add("hidden")

async def hide_toast():
    global toast
    toast.classList.add("hidden")

async def edit_toast_text(text = ""):
    global toast_text
    toast_text.innerText = text

##################### SETTINGS #####################

# Label Dictionaries
generic = {
    0: "0",
    1: "1"
}

breast_cancer = {
    "M": 0,
    "B": 1,
    0: "MALIGNANT",
    1: "BENIGN"
}

parkinsons_disease = {
    0: "HEALTHY",
    1: "PARKINSON'S"
}

##################### GLOBAL #####################

async def read_parameters():
    global filePath, datasetName, plotF1Index, plotF1Name, plotF2Index, plotF2Name, correct_label_index, labelDict, test_train_ratio, centers, is_label_binary, dimensions, data_input_start, data_input_end, test_runs, grouped_data, file

    # !!!!!!!! Data File Path !!!!!!!! #
    filePath = './datasets/parkinsons.data'
    datasetName = document.querySelector("#input-dataset-name").value
    plotF1Index = int(document.querySelector("#input-plot-f1-index").value)
    plotF1Name = document.querySelector("#input-plot-f1-name").value
    plotF2Index = int(document.querySelector("#input-plot-f2-index").value)
    plotF2Name = document.querySelector("#input-plot-f2-name").value

    # !!!!!!!! Index of Correct Label !!!!!!!! #
    correct_label_index = int(document.querySelector("#input-correct-label-index").value)

    # !!!!!!!! Label Dictionary !!!!!!!! #
    labelDict = parkinsons_disease

    # !!!!!!!! Test/Train Ratio !!!!!!!! #
    test_train_ratio = float(document.querySelector("#input-test-train-ratio").value)

    # AMOUNT OF CLASSIFICATIONS OR LABELS
    centers = int(document.querySelector("#input-centers").value)
    is_label_binary = (centers == 2)

    # AMOUNT OF FEATURES OR DIMENSIONS
    dimensions = int(document.querySelector("#input-dimensions").value)
    data_input_start = int(document.querySelector("#input-data-start-index").value)
    data_input_end = int(document.querySelector("#input-data-end-index").value)

    # AMOUNT OF TEST RUN OF SIMULATIONS TO DO
    test_runs = int(document.querySelector("#input-test-runs").value)

    # FILE
    file = open(filePath, newline = '')

##################### CUSTOM DATA READ #####################

@when('change', '#preset-dataset-input')
async def is_preset_checked(*args):
    preset_mode_checkbox = document.querySelector("#preset-dataset-input")
    return (preset_mode_checkbox.checked)

async def processFile(file_input):
    csv_file = await file_input.arrayBuffer()
    file_bytes = csv_file.to_bytes()
    csv_file = BytesIO(file_bytes)
    return TextIOWrapper(csv_file, encoding='utf-8')

##################### RAW DATA READ #####################

async def load_data(preset_mode = True):
    global file
    global grouped_data

    if preset_mode:
        file = open(filePath, newline = '')
    else:
        file_input = document.getElementById('input-dataset-file').files.item(0)
        file = await processFile(file_input)

    raw_data = csv.reader(file, delimiter = ',')
    grouped_data = [ []*centers for i in range(centers)]

    # SEPARATE CLASSES, RESOLVE NON-INTEGER BINARY CLASSIFICATION
    for row in raw_data:
        converted_data = []
        for attr in row:
            try:
                converted_data.append(float(attr))
            except:
                converted_data.append(labelDict[attr])
        grouped_data[int(converted_data[correct_label_index])].append(converted_data)

    # DIVIDE DATA TO TEST AND TRAIN
    for data_group in grouped_data:
        train_counts.append(math.floor(len(data_group) * (1 - test_train_ratio)))
        test_counts.append(math.ceil(len(data_group) * test_train_ratio))

        random.shuffle(data_group)
    
        train_data.extend(data_group[test_counts[grouped_data.index(data_group)]:])
        test_data.extend(data_group[:test_counts[grouped_data.index(data_group)]])

##################### MATPLOTLIB #####################

async def plot_dataset(titleAppend = '', *args, **kwargs):
    global datasetName
    fig, ax = plt.subplots()

    featureM1 = [row[plotF1Index] for row in grouped_data[0]]
    featureM2 = [row[plotF2Index] for row in grouped_data[0]]
    featurem1 = [row[plotF1Index] for row in grouped_data[1]]
    featurem2 = [row[plotF2Index] for row in grouped_data[1]]

    plt.scatter(featureM1, featureM2, color = 'g')
    plt.scatter(featurem1, featurem2, color = 'b')
    
    plt.xlabel(plotF1Name)
    plt.ylabel(plotF2Name)

    plt.title((datasetName + titleAppend), fontsize = 12)

    fig
    return fig 

##################### HERON-CENTROID SMOTE #####################

async def hercenSMOTE(dataset, doPrint = True):
    if doPrint: print("USING HERON-CENTROID SMOTE: \n")
    minority = []
    majority = []
    train_counts = []
    test_counts = []
    train_data = []
    test_data = []
    imbalanceRatio = 2.0
            
    # DETERMINE MINORITY
    if len(dataset[0]) < len(dataset[1]):
        minority = dataset[0]
        majority = dataset[1]
    else:
        minority = dataset[1]
        majority = dataset[0]

    # HERON-CENTROID SMOTE ALGORITHM
    areas = []
    calculateMinority = minority.copy()
    while imbalanceRatio > 1.0:
        # DETERMINE IMBALANCE RATIO
        imbalanceRatio = len(majority)/len(minority)
        # DETERMINE TRIANGLE AREAS
        for instance in calculateMinority[::-1]:
            if len(calculateMinority) <= 0: break
            distances = []
            # DETERMINE TWO-NEAREST NEIGHBORS OF ALL MINORITY INSTANCES
            for neighbor in range(len(minority)):
                distance = np.linalg.norm(instance[data_input_start:data_input_end] - np.array(minority[neighbor][data_input_start:data_input_end]))
                if distance == 0: continue
                distances.append((minority[neighbor], distance))
            distances.sort(key=operator.itemgetter(1))
            neighbors = []
            for i in range(2):
                neighbors.append(distances[i][0])
            # DISTANCE OF THE TWO NEIGHBORS FROM EACH OTHER
            twoNeighborsDistance = np.linalg.norm(np.array(neighbors[0][data_input_start:data_input_end]) - np.array(neighbors[1][data_input_start:data_input_end]))
            # SEMIPERIMETER
            s = (twoNeighborsDistance + distances[0][1] + distances[1][1]) / 2
            # HERON'S FORMULA
            area = np.sqrt(s * (s - twoNeighborsDistance) * (s - distances[0][1]) * (s - distances[1][1]))
            areas.append((area, instance, distances[0][0], distances[1][0]))
            calculateMinority.pop()
        areas.sort(key=operator.itemgetter(0))
        # DETERMINE CENTROID COORDINATES
        centroid = [ []*dimensions for i in range(dimensions)]
        for feature in range(len(centroid)):
            centroid[feature] = (areas[-1][1][feature] + areas[-1][2][feature] + areas[-1][3][feature]) / 3
        # ADD CENTROID AS NEW MINORITY INSTANCE
        minority.append(centroid)
        # ADD VERTICES AND CENTROID TO CALCULATEMINORITY FOR RECALCULATION OF AREA
        #centroid.insert(0, 0) # EXTRA ID COLUMN
        centroid.insert(correct_label_index, areas[-1][1][correct_label_index]) # CLASSIFICATION
        calculateMinority.append(centroid)
        calculateMinority.append(areas[-1][1])
        calculateMinority.append(areas[-1][2])
        calculateMinority.append(areas[-1][3])
        # REMOVE LARGEST AREA AS IT IS RESOLVED
        areas.pop()
        # PRINT IMBALANCE RATIO
        if doPrint: print("\033[1A\033[KIR =", imbalanceRatio)

    # DIVIDE DATA TO TEST AND TRAIN
    for data_group in grouped_data:
        train_counts.append(math.floor(len(data_group) * (1 - test_train_ratio)))
        test_counts.append(math.ceil(len(data_group) * test_train_ratio))

        random.shuffle(data_group)
    
        train_data.extend(data_group[test_counts[grouped_data.index(data_group)]:])
        test_data.extend(data_group[:test_counts[grouped_data.index(data_group)]])

    total_train = len(train_data)
    total_test = len(test_data)

    if doPrint: 
        print("Train data:", total_train, "(0:", train_counts[0], "| 1:", train_counts[1],")")
        print("Test data:", total_test, "(0:", test_counts[0], "| 1:", test_counts[1],")")
        print("Test/Train Ratio:", len(test_data)/len(train_data))
        print("\n---\n")

    # PLOT DATASET
    fig = await plot_dataset(titleAppend = " (Heron-Centroid SMOTE)")
    display(fig, target = 'matplotlib-output-heron-centroid-smote')

##################### CENTROID SMOTE #####################

async def centroidSMOTE(dataset, k = 5, doPrint = True):
    if doPrint: print("USING CENTROID SMOTE: \n")
    minority = []
    majority = []
    train_counts = []
    test_counts = []
    train_data = []
    test_data = []
    imbalanceRatio = 2.0
            
    # DETERMINE MINORITY
    if len(dataset[0]) < len(dataset[1]):
        minority = dataset[0]
        majority = dataset[1]
    else:
        minority = dataset[1]
        majority = dataset[0]

    # CENTROID SMOTE ALGORITHM
    while imbalanceRatio > 1.0:
        # DETERMINE IMBALANCE RATIO
        imbalanceRatio = len(majority)/len(minority)
        # DETERMINE RANDOM MINORITY INSTANCE
        randomInstance = minority[random.randint(0, len(minority) - 1)]
        distances = []
        # DETERMINE TWO-NEAREST NEIGHBORS OF SELECTED MINORITY INSTANCE
        for neighbor in range(len(minority)):
            distance = np.linalg.norm(randomInstance[data_input_start:data_input_end] - np.array(minority[neighbor][data_input_start:data_input_end]))
            if distance == 0: continue
            distances.append((minority[neighbor], distance))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for i in range(k):
            neighbors.append(distances[i][0])
        chosenNeighbors = np.random.Generator.choice(np.array(neighbors), size = 2, replace = False)
        if doPrint: print(chosenNeighbors)
        # DETERMINE COORDINATE FOR NEW INSTANCE BASED ON CHOSEN RANDOM INSTANCE AND ITS TWO-NEAREST NEIGHBORS
        newInstance = [ []*dimensions for i in range(dimensions)]
        for feature in range(dimensions):
            newInstance[feature] = (randomInstance[feature] + neighbors[0][feature] + neighbors[1][feature]) / 3
        # ADD NEW INSTANCE
        #newInstance.insert(0, 0) # EXTRA ID COLUMN
        newInstance.insert(correct_label_index, randomInstance[correct_label_index]) # CLASSIFICATION
        minority.append(newInstance)
        # PRINT IMBALANCE RATIO
        if doPrint: print("\033[1A\033[KIR =", imbalanceRatio)

    # DIVIDE DATA TO TEST AND TRAIN
    for data_group in grouped_data:
        train_counts.append(math.floor(len(data_group) * (1 - test_train_ratio)))
        test_counts.append(math.ceil(len(data_group) * test_train_ratio))

        random.shuffle(data_group)
    
        train_data.extend(data_group[test_counts[grouped_data.index(data_group)]:])
        test_data.extend(data_group[:test_counts[grouped_data.index(data_group)]])

    total_train = len(train_data)
    total_test = len(test_data)

    if doPrint: 
        print("Train data:", total_train, "(0:", train_counts[0], "| 1:", train_counts[1],")")
        print("Test data:", total_test, "(0:", test_counts[0], "| 1:", test_counts[1],")")
        print("Test/Train Ratio:", len(test_data)/len(train_data))
        print("\n---\n")

    # PLOT DATASET
    fig = await plot_dataset(titleAppend = " (Centroid SMOTE)")
    display(fig, target = 'matplotlib-output-centroid-smote')

##################### BASE SMOTE #####################

async def SMOTE(dataset, k = 5, doPrint = True):
    if doPrint: print("USING SMOTE: \n")
    minority = []
    majority = []
    train_counts = []
    test_counts = []
    train_data = []
    test_data = []
    imbalanceRatio = 2.0
            
    # DETERMINE MINORITY
    if len(dataset[0]) < len(dataset[1]):
        minority = dataset[0]
        majority = dataset[1]
    else:
        minority = dataset[1]
        majority = dataset[0]

    # SMOTE ALGORITHM
    while imbalanceRatio > 1.0:
        # DETERMINE IMBALANCE RATIO
        imbalanceRatio = len(majority)/len(minority)
        # DETERMINE RANDOM MINORITY INSTANCE
        randomInstance = minority[random.randint(0, len(minority) - 1)]
        distances = []
        # DETERMINE K-NEAREST NEIGHBORS OF SELECTED MINORITY INSTANCE
        for neighbor in range(len(minority)):
            distance = np.linalg.norm(randomInstance[data_input_start:data_input_end] - np.array(minority[neighbor][data_input_start:data_input_end]))
            if distance == 0: continue
            distances.append((minority[neighbor], distance))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for i in range(k):
            neighbors.append(distances[i][0])
        # DETERMINE RANDOM NEAREST NEIGHBOR AND ITS DISTANCE
        chosenNeighbor = neighbors[random.randint(0, len(neighbors) - 1)]
        # DETERMINE COORDINATE FOR NEW INSTANCE BASED ON CHOSEN NEIGHBOR
        newInstance = [ []*dimensions for i in range(dimensions)]
        for feature in range(dimensions):
            newInstance[feature] = randomInstance[feature] + (chosenNeighbor[feature] - randomInstance[feature]) * random.random()
        # ADD NEW INSTANCE
        #newInstance.insert(0, 0) # EXTRA ID COLUMN
        newInstance.insert(correct_label_index, randomInstance[correct_label_index]) # CLASSIFICATION
        minority.append(newInstance)
        # PRINT IMBALANCE RATIO
        if doPrint: print("\033[1A\033[KIR =", imbalanceRatio)

    # DIVIDE DATA TO TEST AND TRAIN
    for data_group in grouped_data:
        train_counts.append(math.floor(len(data_group) * (1 - test_train_ratio)))
        test_counts.append(math.ceil(len(data_group) * test_train_ratio))

        random.shuffle(data_group)
    
        train_data.extend(data_group[test_counts[grouped_data.index(data_group)]:])
        test_data.extend(data_group[:test_counts[grouped_data.index(data_group)]])

    total_train = len(train_data)
    total_test = len(test_data)

    if doPrint: 
        print("Train data:", total_train, "(0:", train_counts[0], "| 1:", train_counts[1],")")
        print("Test data:", total_test, "(0:", test_counts[0], "| 1:", test_counts[1],")")
        print("Test/Train Ratio:", len(test_data)/len(train_data))
        print("\n---\n")

    # PLOT DATASET
    fig = await plot_dataset(titleAppend = " (Base SMOTE)")
    display(fig, target = 'matplotlib-output-base-smote')
    
##################### K-NEAREST NEIGHBORS #####################

def classify_knn(k, doPrint = True, evalOutputTarget = "output", infoOutputTarget = "output"):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    start = time.time()

    for test in (range(test_runs)):
        test_data_index = math.floor(np.random.random(size = None) * len(test_data))
        data_input = test_data[test_data_index][data_input_start:data_input_end]
        correct_label = test_data[test_data_index][correct_label_index]
        best_classification = []

        distances = []
    
        for i in range(len(train_data) - 1):
            distance = np.linalg.norm(data_input - np.array(train_data[i][data_input_start:data_input_end]))
            distances.append((train_data[i], distance))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for i in range(k):
            neighbors.append(distances[i][0])
        class_votes = {}
        for i in range(len(neighbors)):
            response = neighbors[i][correct_label_index]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)

        best_classification = sorted_votes[0][0]

        if best_classification == correct_label:
            is_correct = " CORRECT "
            if (not is_label_binary):
                true_pos += 1
                true_neg += 1
            else:
                if labelDict[correct_label] == "0" or labelDict[correct_label] == "MALIGNANT" or labelDict[correct_label] == "DIABETIC" or labelDict[correct_label] == "PARKINSON'S":
                    true_pos += 1
                else: 
                    true_neg += 1
        else:
            is_correct = " WRONG "
            if (not is_label_binary):
                false_pos += 1
                false_neg += 1
            else:
                if labelDict[correct_label] == "0" or labelDict[correct_label] == "MALIGNANT" or labelDict[correct_label] == "DIABETIC" or labelDict[correct_label] == "PARKINSON'S":
                    false_pos += 1
                else: 
                    false_neg += 1
        is_correct += "\t[" + labelDict[correct_label] + "]"
        if doPrint: print("\033[1A\033[KRUN #", test + 1, ": \t", labelDict[best_classification], "\t", is_correct)
   
    elapsed_time = time.time() - start

    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    sensitivity = true_pos / (true_pos + false_neg)
    specificity = true_neg / (false_pos + true_neg)
    precision = true_pos / (true_pos + false_pos)
    f1_score = (2 * (precision * sensitivity) / (precision + sensitivity))
    g_mean = (math.sqrt(sensitivity * specificity))

    display("Classification", "K-nearest Neighbors", target = infoOutputTarget)
    display("K-value", k, target = infoOutputTarget)
    display("Accuracy", "%.4f%%" % (accuracy * 100), target = evalOutputTarget)
    display("Sensitivity", "%.4f%%" % (sensitivity * 100), target = evalOutputTarget)
    display("Specificity", "%.4f%%" % (specificity * 100), target = evalOutputTarget)
    display("Precision", "%.4f%%" % (precision * 100), target = evalOutputTarget)
    display("F1-Score", "%.4f" % f1_score, target = evalOutputTarget)
    display("G-mean", "%.4f" % g_mean, target = evalOutputTarget)
    display("Total Tests", test_runs, target = infoOutputTarget)
    display("Average execution time", "%.4f seconds per test run\t\t" % (elapsed_time / test_runs), target = infoOutputTarget)
    display("Total execution time", "%.4f seconds\t\t" % elapsed_time, target = infoOutputTarget)

##################### SVM #####################

def classify_svm(doPrint = True, evalOutputTarget = "output", infoOutputTarget = "output"):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    start = time.time()

    for test in (range(test_runs)):
        test_data_index = math.floor(np.random.random(size = None) * len(test_data))
        data_input = test_data[test_data_index][data_input_start:data_input_end]
        correct_label = test_data[test_data_index][correct_label_index]
        
        classify = svm.SVC()
        classify.fit(np.array(train_data)[:, data_input_start:data_input_end], np.array(train_data)[:, correct_label_index])

        prediction = classify.predict(np.array(data_input).reshape(1, -1))

        if prediction == correct_label:
            is_correct = " CORRECT "
            if (not is_label_binary):
                true_pos += 1
                true_neg += 1
            else:
                if labelDict[correct_label] == "0" or labelDict[correct_label] == "MALIGNANT" or labelDict[correct_label] == "DIABETIC" or labelDict[correct_label] == "PARKINSON'S":
                    true_pos += 1
                else: 
                    true_neg += 1
        else:
            is_correct = " WRONG "
            if (not is_label_binary):
                false_pos += 1
                false_neg += 1
            else:
                if labelDict[correct_label] == "0" or labelDict[correct_label] == "MALIGNANT" or labelDict[correct_label] == "DIABETIC" or labelDict[correct_label] == "PARKINSON'S":
                    false_pos += 1
                else: 
                    false_neg += 1
        is_correct += "\t[" + labelDict[correct_label] + "]"
        if doPrint: print("\033[1A\033[KRUN #", test + 1, ": \t", labelDict[prediction], "\t", is_correct)
        
    elapsed_time = time.time() - start

    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    sensitivity = true_pos / (true_pos + false_neg)
    specificity = true_neg / (false_pos + true_neg)
    precision = true_pos / (true_pos + false_pos)
    f1_score = (2 * (precision * sensitivity) / (precision + sensitivity))
    g_mean = (math.sqrt(sensitivity * specificity))

    display("Classification", "Support Vector Machine", target = infoOutputTarget)
    display("Accuracy", "%.4f%%" % (accuracy * 100), target = evalOutputTarget)
    display("Sensitivity", "%.4f%%" % (sensitivity * 100), target = evalOutputTarget)
    display("Specificity", "%.4f%%" % (specificity * 100), target = evalOutputTarget)
    display("Precision", "%.4f%%" % (precision * 100), target = evalOutputTarget)
    display("F1-Score", "%.4f" % f1_score, target = evalOutputTarget)
    display("G-mean", "%.4f" % g_mean, target = evalOutputTarget)
    display("Total Tests", test_runs, target = infoOutputTarget)
    display("Average execution time", "%.4f seconds per test run\t\t" % (elapsed_time / test_runs), target = infoOutputTarget)
    display("Total execution time", "%.4f seconds\t\t" % elapsed_time, target = infoOutputTarget)

##################### RUN #####################

async def run_simulation(event):
    # Read Parameters
    await read_parameters()

    # Remove Skeleton DOMs
    document.getElementById("matplotlib-output-imbalanced").innerHTML = ""
    document.getElementById("matplotlib-output-base-smote").innerHTML = ""
    document.getElementById("matplotlib-output-heron-centroid-smote").innerHTML = ""
    
    # Imbalanced
    await load_data(await is_preset_checked())
    fig = await plot_dataset(titleAppend = " (Imbalanced)")
    await edit_toast_text("Plotting imbalanced data...")
    display(fig, target = 'matplotlib-output-imbalanced')
    await edit_toast_text("Classifying imbalanced data...")
    classify_knn(k = 3, doPrint = False, evalOutputTarget = "evaluation-output-imbalanced", infoOutputTarget = "information-output-imbalanced")

    # Base SMOTE
    await load_data(await is_preset_checked())
    await edit_toast_text("Plotting Base SMOTE data...")
    await SMOTE(grouped_data, doPrint = False)
    await edit_toast_text("Classifying Base SMOTE data...")
    classify_knn(k = 3, doPrint = False, evalOutputTarget = "evaluation-output-base-smote", infoOutputTarget = "information-output-base-smote")

    # Heron-centroid SMOTE
    await load_data(await is_preset_checked())
    await edit_toast_text("Plotting Heron-Centroid SMOTE data...")
    await hercenSMOTE(grouped_data, doPrint = False)
    await edit_toast_text("Classifying Heron-Centroid SMOTE data...")
    classify_knn(k = 3, doPrint = False, evalOutputTarget = "evaluation-output-heron-centroid-smote", infoOutputTarget = "information-output-heron-centroid-smote")

    # Simulation Finished
    document.getElementById("run-simulation").innerHTML = "Simulation Finished"
    await hide_toast()
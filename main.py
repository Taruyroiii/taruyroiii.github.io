##################### IMPORTS #####################
import asyncio
from js import document
from io import BytesIO, TextIOWrapper
from pyscript import when, display

import numpy as np
import csv
import copy
import time
import math
import random
import operator
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

train_counts = []
test_counts = []
train_data = []
test_data = []
majorityCount = 1
minorityCount = 1

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

##################### GLOBAL VARIABLES #####################

async def read_parameters():
    global filePath, datasetName, plotF1Index, plotF1Name, plotF2Index, plotF2Name, correct_label_index, test_train_ratio, data_points, classification_algo, centers, is_label_binary, dimensions, data_input_start, data_input_end, grouped_data, saved_data, file

    # !!!!!!!! Data File Path !!!!!!!! #
    filePath = document.querySelector("#input-dataset-file-path").value
    datasetName = document.querySelector("#input-dataset-name").value
    plotF1Index = int(document.querySelector("#input-plot-f1-index").value)
    plotF1Name = document.querySelector("#input-plot-f1-name").value
    plotF2Index = int(document.querySelector("#input-plot-f2-index").value)
    plotF2Name = document.querySelector("#input-plot-f2-name").value

    # !!!!!!!! Index of Correct Label !!!!!!!! #
    correct_label_index = int(document.querySelector("#input-correct-label-index").value)

    # !!!!!!!! Test/Train Data !!!!!!!! #
    test_train_ratio = float(document.querySelector("#input-test-train-ratio").value)
    data_points = float(document.querySelector("#input-data-points").value)
    classification_algo = document.querySelector("#input-classification-algo").value

    # AMOUNT OF CLASSIFICATIONS OR LABELS
    centers = int(document.querySelector("#input-centers").value)
    is_label_binary = (centers == 2)

    # AMOUNT OF FEATURES OR DIMENSIONS
    dimensions = int(document.querySelector("#input-dimensions").value)
    data_input_start = int(document.querySelector("#input-data-start-index").value)
    data_input_end = int(document.querySelector("#input-data-end-index").value)

    # FILE
    file = open(filePath, newline = '')
    grouped_data = [ []*centers for i in range(centers)]
    saved_data = [ []*centers for i in range(centers)]

##################### CUSTOM DATA READ #####################

@when('change', '#preset-dataset-input')
async def is_preset_checked(*args):
    preset_mode_checkbox = document.querySelector("#preset-dataset-input")
    return (preset_mode_checkbox.checked)

async def is_test_data_checked(*args):
    test_data_checkbox = document.querySelector("#test-dataset-input")
    return (test_data_checkbox.checked)

async def processFile(file_input):
    csv_file = await file_input.arrayBuffer()
    file_bytes = csv_file.to_bytes()
    csv_file = BytesIO(file_bytes)
    return TextIOWrapper(csv_file, encoding='utf-8')

##################### RAW DATA READ #####################

async def load_data(preset_mode = True):
    global file, grouped_data, saved_data, majorityCount, minorityCount, minority, majority, train_counts, test_counts, train_data, test_data, newSimulation
    
    minority = []
    majority = []
    train_counts = []
    test_counts = []
    train_data = []
    test_data = []

    if newSimulation:
        if preset_mode:
            file = open(filePath, newline = '')
        else:
            file_input = document.getElementById('input-dataset-file').files.item(0)
            file = await processFile(file_input)

        raw_data = csv.reader(file, delimiter = ',')
        saved_data = [ []*centers for i in range(centers)]

        # SEPARATE CLASSES, RESOLVE NON-INTEGER BINARY CLASSIFICATION
        for row in raw_data:
            converted_data = []
            for attr in row:
                try:
                    converted_data.append(float(attr))
                except:
                    converted_data.append(attr)
            saved_data[int(converted_data[correct_label_index])].append(converted_data)

        # RANDOMLY SELECT DATA POINTS BASED ON INPUTTED COUNT
        random.shuffle(saved_data[0])
        random.shuffle(saved_data[1])

        # CALCULATE HOW MUCH POINTS TO GET BASED ON GROUPED DATA RATIO
        dataPointCount_0 = len(saved_data[0])
        dataPointCount_1 = len(saved_data[1])
        totalCount = dataPointCount_0 + dataPointCount_1
        datasetSizeFactor = totalCount / data_points
        newCount_0 = math.floor(dataPointCount_0 / datasetSizeFactor)
        newCount_1 = math.ceil(dataPointCount_1 / datasetSizeFactor)

        saved_data[0] = saved_data[0][0:newCount_0]
        saved_data[1] = saved_data[1][0:newCount_1]

        # SET FLAG
        newSimulation = False

    grouped_data = copy.deepcopy(saved_data)

    # DETERMINE MAJORITY
    if len(grouped_data[0]) < len(grouped_data[1]):
        minority = grouped_data[0]
        majority = grouped_data[1]
    else:
        minority = grouped_data[1]
        majority = grouped_data[0]
    
    # UPDATE COUNT
    minorityCount = len(minority)
    majorityCount = len(majority)

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

    featureM1 = [row[plotF1Index] for row in majority]
    featureM2 = [row[plotF2Index] for row in majority]
    featurem1 = [row[plotF1Index] for row in minority]
    featurem2 = [row[plotF2Index] for row in minority]

    plt.scatter(featureM1, featureM2, color = 'b') # Majority is blue
    plt.scatter(featurem1, featurem2, color = 'g') # Minority is green
    
    plt.xlabel(plotF1Name)
    plt.ylabel(plotF2Name)

    plt.title((datasetName + titleAppend), fontsize = 12)

    fig
    return fig 

##################### EUCLIDEAN DISTANCE #####################

def euclidean_distance(array1, array2):
  # Check if the arrays have the same shape
  if array1.shape != array2.shape:
    raise ValueError("The arrays must have the same shape")
  else:
    # Calculate the squared difference between the arrays
    squared_diff = np.square(array1 - array2)
    # Sum up the squared differences
    sum_squared_diff = np.sum(squared_diff, axis=None)
    # Take the square root of the sum
    distance = np.sqrt(sum_squared_diff)

    return distance


##################### HERON-CENTROID SMOTE #####################

async def hercenSMOTE(dataset, doPrint = True):
    global majorityCount, minorityCount

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
        area = 0
        for instance in calculateMinority[::-1]:
            if len(calculateMinority) <= 0: break
            distances = []
            # DETERMINE TWO-NEAREST NEIGHBORS OF ALL MINORITY INSTANCES
            for neighbor in range(len(minority)):
                distance = euclidean_distance(np.array(instance[data_input_start:data_input_end]), np.array(minority[neighbor][data_input_start:data_input_end]))
                if distance == 0: continue
                distances.append((minority[neighbor], distance))
            distances.sort(key=operator.itemgetter(1))
            neighbors = []
            for i in range(2):
                neighbors.append(distances[i][0])
            # DISTANCE OF THE TWO NEIGHBORS FROM EACH OTHER
            twoNeighborsDistance = euclidean_distance(np.array(neighbors[0][data_input_start:data_input_end]), np.array(neighbors[1][data_input_start:data_input_end]))
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
        # ADD CLASSIFICATION
        centroid.insert(correct_label_index, areas[-1][1][correct_label_index]) # CLASSIFICATION
        # DETERMINE TWO-NEAREST MAJORITY INSTANCE OF CENTROID
        majorityDistances = []
        for neighbor in range(len(majority)):
            distance = euclidean_distance(np.array(centroid[data_input_start:data_input_end]), np.array(majority[neighbor][data_input_start:data_input_end]))
            majorityDistances.append((majority[neighbor], distance))
        majorityDistances.sort(key=operator.itemgetter(1))
        majorityNeighbors = []
        for i in range(2):
            majorityNeighbors.append(majorityDistances[i][0])
        # DISTANCE OF THE TWO MAJORITY NEIGHBORS FROM EACH OTHER
        twoMajorityNeighborsDistance = euclidean_distance(np.array(majorityNeighbors[0][data_input_start:data_input_end]), np.array(majorityNeighbors[1][data_input_start:data_input_end]))
        # MAJORITY SEMIPERIMETER
        s = (twoMajorityNeighborsDistance + majorityDistances[0][1] + majorityDistances[1][1]) / 2
        # MAJORITY HERON'S FORMULA
        majorityArea = np.sqrt(s * (s - twoMajorityNeighborsDistance) * (s - majorityDistances[0][1]) * (s - majorityDistances[1][1]))
        # ADD CENTROID AS NEW MINORITY INSTANCE IF MAJORITY AREA > MINORITY AREA
        if majorityArea > area:
            minority.append(centroid)
            calculateMinority.append(centroid)
            calculateMinority.append(areas[-1][1])
            calculateMinority.append(areas[-1][2])
            calculateMinority.append(areas[-1][3])
        # ADD VERTICES AND CENTROID TO CALCULATEMINORITY FOR RECALCULATION OF AREA
        #centroid.insert(0, 0) # EXTRA ID COLUMN
        # REMOVE LARGEST AREA AS IT IS RESOLVED
        areas.pop()
        # PRINT IMBALANCE RATIO
        if doPrint: print("IR =", imbalanceRatio)
        # UPDATE COUNT
        majorityCount = len(majority)
        minorityCount = len(minority)

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
    
##################### EXISTING SMOTE #####################

async def SMOTE(dataset, k = 5, doPrint = True):
    global majorityCount, minorityCount

    if doPrint: print("USING EXISTING SMOTE: \n")
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

    syntheticInstances = []

    # SMOTE ALGORITHM
    while imbalanceRatio > 1.0:
        # DETERMINE IMBALANCE RATIO
        imbalanceRatio = len(majority)/(len(minority) + len(syntheticInstances))
        # DETERMINE RANDOM MINORITY INSTANCE
        randomInstance = minority[random.randint(0, len(minority) - 1)]
        distances = []
        # DETERMINE K-NEAREST NEIGHBORS OF SELECTED MINORITY INSTANCE
        for neighbor in range(len(minority)):
            distance = euclidean_distance(np.array(randomInstance[data_input_start:data_input_end]), np.array(minority[neighbor][data_input_start:data_input_end]))
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
            newInstance[feature] = randomInstance[feature] + random.random() * (chosenNeighbor[feature] - randomInstance[feature])
        # ADD CLASSIFICATION
        newInstance.insert(correct_label_index, randomInstance[correct_label_index]) # CLASSIFICATION
        # ADD MINORITY INSTANCE
        syntheticInstances.append(newInstance)
        # PRINT IMBALANCE RATIO
        if doPrint: print("IR =", imbalanceRatio)
        # UPDATE COUNT
        majorityCount = len(majority)
        minorityCount = len(minority) + len(syntheticInstances)
    
    minority.extend(syntheticInstances)

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
    fig = await plot_dataset(titleAppend = " (Existing SMOTE)")
    display(fig, target = 'matplotlib-output-existing-smote')
    
##################### PERFORMANCE METRICS #####################
def calc_accuracy(true_pos, true_neg, false_pos, false_neg):
    if (true_pos + true_neg) == 0:
        return 0
    else:
        return ((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg))
    
def calc_sensitivity(true_pos, false_neg):
    if (true_pos + false_neg) == 0:
        return 0
    else:
        return (true_pos / (true_pos + false_neg))
    
def calc_specificity(true_neg, false_pos):
    if (false_pos + true_neg) == 0:
        return 0
    else:
        return (true_neg / (false_pos + true_neg))

def calc_precision(true_pos, false_pos):
    if (true_pos + false_pos) == 0:
        return 0
    else:
        return (true_pos / (true_pos + false_pos))

def calc_f1_score(precision, sensitivity):
    if (precision + sensitivity) == 0:
        return 0
    else:
        return (2 * (precision * sensitivity) / (precision + sensitivity))

def calc_g_mean(sensitivity, specificity):
    return (math.sqrt(sensitivity * specificity))

##################### K-NEAREST NEIGHBORS #####################

def classify_knn(k, doPrint = True, evalOutputTarget = "output", infoOutputTarget = "output", summarizedOutputTarget = "output"):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    if is_test_data_checked:
        input_data = test_data
    else:
        input_data = train_data

    start = time.time()

    knn = KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
    knn.fit(np.array(train_data)[:, data_input_start:data_input_end], np.array(train_data)[:, correct_label_index])

    test_runs = len(input_data)

    for test in (range(test_runs)):
        input_data_index = math.floor(np.random.random(size = None) * len(input_data))
        data_input = input_data[input_data_index][data_input_start:data_input_end]
        correct_label = input_data[input_data_index][correct_label_index]
        
        best_classification = knn.predict(np.array(data_input).reshape(1, -1))

        if best_classification == correct_label:
            is_correct = " CORRECT "
            if correct_label == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            is_correct = " WRONG "
            if correct_label == 1:
                false_neg += 1
            else:
                false_pos += 1

        is_correct += "\t[" + str(correct_label) + "]"
        if doPrint: print("\033[1A\033[KRUN #", test + 1, ": \t", str(best_classification), "\t", is_correct)
   
    elapsed_time = time.time() - start

    accuracy = calc_accuracy(true_pos, true_neg, false_pos, false_neg)
    sensitivity = calc_sensitivity(true_pos, false_neg)
    specificity = calc_specificity(true_neg, false_pos)
    precision = calc_precision(true_pos, false_pos)
    f1_score = calc_f1_score(precision, sensitivity)
    g_mean = calc_g_mean(sensitivity, specificity)

    document.getElementById("classification-info-output").innerHTML = ""
    display("Classification Algorithm", "K-nearest Neighbors" , target = "classification-info-output")
    display("K-value", k, target = "classification-info-output")
    display("Total Classifications", test_runs, target = "classification-info-output")

    display("Accuracy", "%.4f%%" % (accuracy * 100), target = evalOutputTarget)
    display("Sensitivity", "%.4f%%" % (sensitivity * 100), target = evalOutputTarget)
    display("Specificity", "%.4f%%" % (specificity * 100), target = evalOutputTarget)
    display("Precision", "%.4f%%" % (precision * 100), target = evalOutputTarget)
    display("F1-Score", "%.4f" % f1_score, target = evalOutputTarget)
    display("G-mean", "%.4f" % g_mean, target = evalOutputTarget)

    display("%.4f%%" % (accuracy * 100), target = summarizedOutputTarget)
    display("%.4f%%" % (sensitivity * 100), target = summarizedOutputTarget)
    display("%.4f%%" % (specificity * 100), target = summarizedOutputTarget)
    display("%.4f%%" % (precision * 100), target = summarizedOutputTarget)
    display("%.4f" % f1_score, target = summarizedOutputTarget)
    display("%.4f" % g_mean, target = summarizedOutputTarget)
    
    display("Total Instances", (majorityCount + minorityCount), target = infoOutputTarget)
    display("No. of Majority", majorityCount, target = infoOutputTarget)
    display("No. of Minority", minorityCount, target = infoOutputTarget)
    display("Imbalance Ratio", "%.4f" % (majorityCount / minorityCount), target = infoOutputTarget)
    display("Average execution time", "%.4f seconds per test run\t\t" % (elapsed_time / test_runs), target = infoOutputTarget)
    display("Total execution time", "%.4f seconds\t\t" % elapsed_time, target = infoOutputTarget)

##################### SVM #####################

def classify_svm(doPrint = True, evalOutputTarget = "output", infoOutputTarget = "output", summarizedOutputTarget = "output"):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    if is_test_data_checked:
        input_data = test_data
    else:
        input_data = train_data

    start = time.time()
        
    svm_classify = svm.SVC()
    svm_classify.fit(np.array(train_data)[:, data_input_start:data_input_end], np.array(train_data)[:, correct_label_index])

    test_runs = len(input_data)

    for test in (range(test_runs)):
        input_data_index = math.floor(np.random.random(size = None) * len(input_data))
        data_input = input_data[input_data_index][data_input_start:data_input_end]
        correct_label = input_data[input_data_index][correct_label_index]

        prediction = svm_classify.predict(np.array(data_input).reshape(1, -1))
        if prediction == correct_label:
            is_correct = " CORRECT "
            if correct_label == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            is_correct = " WRONG "
            if correct_label == 1:
                false_neg += 1
            else:
                false_pos += 1

        is_correct += "\t[" + str(correct_label) + "]"
        if doPrint: print("\033[1A\033[KRUN #", test + 1, ": \t", str(prediction), "\t", is_correct)
        
    elapsed_time = time.time() - start
    
    accuracy = calc_accuracy(true_pos, true_neg, false_pos, false_neg)
    sensitivity = calc_sensitivity(true_pos, false_neg)
    specificity = calc_specificity(true_neg, false_pos)
    precision = calc_precision(true_pos, false_pos)
    f1_score = calc_f1_score(precision, sensitivity)
    g_mean = calc_g_mean(sensitivity, specificity)

    document.getElementById("classification-info-output").innerHTML = ""
    display("Classification Algorithm", "Support Vector Machine", target = "classification-info-output")
    display("Total Classifications", test_runs, target = "classification-info-output")

    display("Accuracy", "%.4f%%" % (accuracy * 100), target = evalOutputTarget)
    display("Sensitivity", "%.4f%%" % (sensitivity * 100), target = evalOutputTarget)
    display("Specificity", "%.4f%%" % (specificity * 100), target = evalOutputTarget)
    display("Precision", "%.4f%%" % (precision * 100), target = evalOutputTarget)
    display("F1-Score", "%.4f" % f1_score, target = evalOutputTarget)
    display("G-mean", "%.4f" % g_mean, target = evalOutputTarget)

    display("%.4f%%" % (accuracy * 100), target = summarizedOutputTarget)
    display("%.4f%%" % (sensitivity * 100), target = summarizedOutputTarget)
    display("%.4f%%" % (specificity * 100), target = summarizedOutputTarget)
    display("%.4f%%" % (precision * 100), target = summarizedOutputTarget)
    display("%.4f" % f1_score, target = summarizedOutputTarget)
    display("%.4f" % g_mean, target = summarizedOutputTarget)
    
    display("Total Instances", (majorityCount + minorityCount), target = infoOutputTarget)
    display("No. of Majority", majorityCount, target = infoOutputTarget)
    display("No. of Minority", minorityCount, target = infoOutputTarget)
    display("Imbalance Ratio", "%.4f" % (majorityCount / minorityCount), target = infoOutputTarget)
    display("Average execution time", "%.4f seconds per test run\t\t" % (elapsed_time / test_runs), target = infoOutputTarget)
    display("Total execution time", "%.4f seconds\t\t" % elapsed_time, target = infoOutputTarget)

##################### RANDOM FOREST #####################

def classify_rf(doPrint = True, evalOutputTarget = "output", infoOutputTarget = "output", summarizedOutputTarget = "output"):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    if is_test_data_checked:
        input_data = test_data
    else:
        input_data = train_data

    start = time.time()
        
    rf_classify = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf_classify.fit(np.array(train_data)[:, data_input_start:data_input_end], np.array(train_data)[:, correct_label_index])

    test_runs = len(input_data)

    for test in (range(test_runs)):
        input_data_index = math.floor(np.random.random(size = None) * len(input_data))
        data_input = input_data[input_data_index][data_input_start:data_input_end]
        correct_label = input_data[input_data_index][correct_label_index]

        prediction = rf_classify.predict(np.array(data_input).reshape(1, -1))

        if prediction == correct_label:
            is_correct = " CORRECT "
            if correct_label == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            is_correct = " WRONG "
            if correct_label == 1:
                false_neg += 1
            else:
                false_pos += 1

        is_correct += "\t[" + str(correct_label) + "]"
        if doPrint: print("\033[1A\033[KRUN #", test + 1, ": \t", str(prediction), "\t", is_correct)
        
    elapsed_time = time.time() - start
    
    accuracy = calc_accuracy(true_pos, true_neg, false_pos, false_neg)
    sensitivity = calc_sensitivity(true_pos, false_neg)
    specificity = calc_specificity(true_neg, false_pos)
    precision = calc_precision(true_pos, false_pos)
    f1_score = calc_f1_score(precision, sensitivity)
    g_mean = calc_g_mean(sensitivity, specificity)

    document.getElementById("classification-info-output").innerHTML = ""
    display("Classification Algorithm", "Random Forest", target = "classification-info-output")
    display("Total Classifications", test_runs, target = "classification-info-output")

    display("Accuracy", "%.4f%%" % (accuracy * 100), target = evalOutputTarget)
    display("Sensitivity", "%.4f%%" % (sensitivity * 100), target = evalOutputTarget)
    display("Specificity", "%.4f%%" % (specificity * 100), target = evalOutputTarget)
    display("Precision", "%.4f%%" % (precision * 100), target = evalOutputTarget)
    display("F1-Score", "%.4f" % f1_score, target = evalOutputTarget)
    display("G-mean", "%.4f" % g_mean, target = evalOutputTarget)

    display("%.4f%%" % (accuracy * 100), target = summarizedOutputTarget)
    display("%.4f%%" % (sensitivity * 100), target = summarizedOutputTarget)
    display("%.4f%%" % (specificity * 100), target = summarizedOutputTarget)
    display("%.4f%%" % (precision * 100), target = summarizedOutputTarget)
    display("%.4f" % f1_score, target = summarizedOutputTarget)
    display("%.4f" % g_mean, target = summarizedOutputTarget)

    display("Total Instances", (majorityCount + minorityCount), target = infoOutputTarget)
    display("No. of Majority", majorityCount, target = infoOutputTarget)
    display("No. of Minority", minorityCount, target = infoOutputTarget)
    display("Imbalance Ratio", "%.4f" % (majorityCount / minorityCount), target = infoOutputTarget)
    display("Average execution time", "%.4f seconds per test run\t\t" % (elapsed_time / test_runs), target = infoOutputTarget)
    display("Total execution time", "%.4f seconds\t\t" % elapsed_time, target = infoOutputTarget)

##################### CLASSIFY #####################

def classify(doPrint, evalOutputTarget, infoOutputTarget, summarizedOutputTarget):
    if (classification_algo == "knn"):
        classify_knn(k = 3, doPrint = doPrint, evalOutputTarget = evalOutputTarget, infoOutputTarget = infoOutputTarget, summarizedOutputTarget = summarizedOutputTarget)
    elif (classification_algo == "svm"):
        classify_svm(doPrint = doPrint, evalOutputTarget = evalOutputTarget, infoOutputTarget = infoOutputTarget, summarizedOutputTarget = summarizedOutputTarget)
    elif (classification_algo == "rf"):
        classify_rf(doPrint = doPrint, evalOutputTarget = evalOutputTarget, infoOutputTarget = infoOutputTarget, summarizedOutputTarget = summarizedOutputTarget)
    else:
        raise Exception("Invalid classification algorithm mode.")

##################### RUN #####################

async def run_simulation(event):
    global grouped_data, saved_data, newSimulation
    newSimulation = True

    # Read Parameters
    await read_parameters()

    # Imbalanced
    await load_data(await is_preset_checked())
    fig = await plot_dataset(titleAppend = " (Imbalanced)")
    await edit_toast_text("Plotting imbalanced data...")
    display(fig, target = 'matplotlib-output-imbalanced')
    await edit_toast_text("Classifying imbalanced data...")
    display("Imbalanced", target = "summarized-output-imbalanced")
    classify(doPrint = False, evalOutputTarget = "evaluation-output-imbalanced", infoOutputTarget = "information-output-imbalanced", summarizedOutputTarget = "summarized-output-imbalanced")

    # Existing SMOTE
    await load_data(await is_preset_checked())
    await edit_toast_text("Plotting Existing SMOTE data...")
    await SMOTE(grouped_data, doPrint = False)
    await edit_toast_text("Classifying Existing SMOTE data...")
    display("Existing SMOTE", target = "summarized-output-existing-smote")
    classify(doPrint = False, evalOutputTarget = "evaluation-output-existing-smote", infoOutputTarget = "information-output-existing-smote", summarizedOutputTarget = "summarized-output-existing-smote")

    # Heron-centroid SMOTE
    await load_data(await is_preset_checked())
    await edit_toast_text("Plotting Heron-Centroid SMOTE data...")
    await hercenSMOTE(grouped_data, doPrint = False)
    await edit_toast_text("Classifying Heron-Centroid SMOTE data...")
    display("Heron-Centroid SMOTE", target = "summarized-output-heron-centroid-smote")
    classify(doPrint = False, evalOutputTarget = "evaluation-output-heron-centroid-smote", infoOutputTarget = "information-output-heron-centroid-smote", summarizedOutputTarget = "summarized-output-heron-centroid-smote")

    # Simulation Finished
    plt.close('all')
    document.getElementById("run-simulation").innerHTML = "Run Simulation"
    document.getElementById("run-simulation").disabled = False
    await hide_toast()
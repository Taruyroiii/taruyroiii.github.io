function updateDatasetInputData(presetMode = True) {
    customDiv = document.getElementById("mode-custom");
    presetDiv = document.getElementById("mode-preset");

    if (presetMode) {
        customDiv.style.display = "none";
        presetDiv.style.display = null;
    } else {
        presetDiv.style.display = "none";
        customDiv.style.display = null;
    }
};

function getJSONPath(datasetName) {
    switch (datasetName) {
        case "breast":
            return "./datasets/wdbc.json";
        case "heart":
            return "./datasets/heart_attack.json";
        case "diabetes":
            return "./datasets/pima_diabetes.json";
        case "liver":
            return "./datasets/indian_liver.json";
        case "parkinsons":
            return "./datasets/parkinsons.json";
        case "stroke":
            return "./datasets/stroke.json";
        default:
            return "";
    }
};

function updateDefaultParameters(datasetName) {
    fetch(getJSONPath(datasetName))
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not OK');
            }
            return response.json();
        })
        .then(data => {
            document.getElementById("input-dataset-name").value = data.name
            document.getElementById("input-dataset-file-path").value = data.file_path
            document.getElementById("input-dimensions").value = data.dimensions
            document.getElementById("input-data-start-index").value = data.data_input_start
            document.getElementById("input-data-end-index").value = data.data_input_end
            document.getElementById("input-correct-label-index").value = data.correct_label_index
            document.getElementById("input-plot-f1-index").value = data.plot_f1_index
            document.getElementById("input-plot-f1-name").value = data.plot_f1_Name
            document.getElementById("input-plot-f2-index").value = data.plot_f2_Index
            document.getElementById("input-plot-f2-name").value = data.plot_f2_Name
            document.getElementById("input-centers").value = data.labels
            document.getElementById("input-test-train-ratio").value = data.test_train_ratio
            document.getElementById("input-data-points").value = data.data_points
            document.getElementById("input-data-points").max = data.data_points
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
        });
};

function runSimulation(obj) {
    // Make Summarized Output Appear
    document.getElementById("summarized-output").style.display = null;
    document.getElementById("summarized-output-skeleton").style.display = "none";

    // Remove contents of some DOMs
    document.getElementById("classification-info-output").innerHTML = ""
    document.getElementById("matplotlib-output-imbalanced").innerHTML = ""
    document.getElementById("matplotlib-output-existing-smote").innerHTML = ""
    document.getElementById("matplotlib-output-heron-centroid-smote").innerHTML = ""
    document.getElementById("summarized-output-imbalanced").innerHTML = ""
    document.getElementById("summarized-output-existing-smote").innerHTML = ""
    document.getElementById("summarized-output-heron-centroid-smote").innerHTML = ""
    document.getElementById("evaluation-output-imbalanced").innerHTML = ""
    document.getElementById("evaluation-output-existing-smote").innerHTML = ""
    document.getElementById("evaluation-output-heron-centroid-smote").innerHTML = ""
    document.getElementById("information-output-imbalanced").innerHTML = ""
    document.getElementById("information-output-existing-smote").innerHTML = ""
    document.getElementById("information-output-heron-centroid-smote").innerHTML = ""

    obj.disabled = true;
    obj.innerHTML = 'Running...';
    document.getElementById('toast').classList.remove('hidden');
}
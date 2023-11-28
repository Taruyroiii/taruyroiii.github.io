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
        case "wisconsin":
            return "./datasets/wdbc.json";
        case "pima":
            return "./datasets/wdbc.json";
        case "indian":
            return "./datasets/wdbc.json";
        case "oxford":
            return "./datasets/parkinsons.json";
        case "cervical":
            return "./datasets/wdbc.json";
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
            document.getElementById("input-test-runs").value = data.classification_runs
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
        });
};
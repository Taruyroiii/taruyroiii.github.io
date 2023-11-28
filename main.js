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
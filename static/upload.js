let uploadedFile = null;

function handleFile(files) {
  try {
    if (files.length > 0) {
      const file = files[0];
      if (file.type !== "audio/wav") {
        alert("Only .wav files are accepted.");
        return;
      }
      uploadedFile = file;
      displayAudio(file);
    }
  } catch (error) {
    console.error("Error handling file:", error);
  }
}

function handleDrop(event) {
  try {
    event.preventDefault();
    const files = event.dataTransfer.files;
    handleFile(files);
    document.getElementById("file-input").value = "";
  } catch (error) {
    console.error("Error handling drop:", error);
  }
}

function handleDragOver(event) {
  event.preventDefault();
}

function displayAudio(file) {
  try {
    const audioElement = document.getElementById("audio");
    audioElement.innerHTML = "";

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = URL.createObjectURL(file);

    const removeBtn = document.createElement("button");
    removeBtn.innerHTML = "×";
    removeBtn.onclick = function (event) {
      event.stopPropagation(); // Prevent event from propagating to the parent
      removeAudio();
      audioElement.innerHTML = "Drag & Drop audio file here or click to upload";
    };

    audioElement.appendChild(audio);
    audioElement.appendChild(removeBtn);
  } catch (error) {
    console.error("Error displaying audio:", error);
  }
}

function removeAudio() {
  uploadedFile = null;
}

async function uploadFile() {
  try {
    if (!uploadedFile) {
      alert("Please select a .wav file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", uploadedFile);

    const response = await fetch("/result", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const result = await response.text();
      document.body.innerHTML = result;
    } else {
      alert("Failed to upload file.");
    }
  } catch (error) {
    console.error("Error uploading file:", error);
    alert("An error occurred while uploading the file.");
  }
}

const uploadSection = document.getElementById("container");
uploadSection.addEventListener("drop", handleDrop);
uploadSection.addEventListener("dragover", handleDragOver);

document
  .getElementById("file-input")
  .addEventListener("change", function (event) {
    handleFile(event.target.files);
  });

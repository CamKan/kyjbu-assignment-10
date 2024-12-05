document.addEventListener("DOMContentLoaded", function () {
  const imageUpload = document.getElementById("image-upload");
  const fileName = document.getElementById("file-name");
  const browseBtn = document.getElementById("browse-btn");
  const searchBtn = document.getElementById("search-btn");
  const queryType = document.getElementById("query-type");
  const imageQueryGroup = document.getElementById("image-query-group");
  const textQueryGroup = document.getElementById("text-query-group");
  const hybridWeightGroup = document.getElementById("hybrid-weight-group");
  const usePca = document.getElementById("use-pca");
  const pcaComponents = document.getElementById("pca-components");

  queryType.addEventListener("change", () => {
    switch (queryType.value) {
      case "image":
        imageQueryGroup.style.display = "block";
        textQueryGroup.style.display = "none";
        hybridWeightGroup.style.display = "none";
        break;
      case "text":
        imageQueryGroup.style.display = "none";
        textQueryGroup.style.display = "block";
        hybridWeightGroup.style.display = "none";
        break;
      case "hybrid":
        imageQueryGroup.style.display = "block";
        textQueryGroup.style.display = "block";
        hybridWeightGroup.style.display = "block";
        break;
    }
  });

  browseBtn.addEventListener("click", () => {
    imageUpload.click();
  });

  imageUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      fileName.textContent = file.name;
    }
  });

  usePca.addEventListener("change", (e) => {
    pcaComponents.disabled = !e.target.checked;
  });

  searchBtn.addEventListener("click", async () => {
    const formData = new FormData();

    if (queryType.value === "image" || queryType.value === "hybrid") {
      if (!imageUpload.files[0] && queryType.value === "image") {
        alert("Please select an image for image query");
        return;
      }
      formData.append("image", imageUpload.files[0]);
    }

    if (queryType.value === "text" || queryType.value === "hybrid") {
      const textQuery = document.getElementById("text-query").value;
      if (!textQuery && queryType.value === "text") {
        alert("Please enter text for text query");
        return;
      }
      formData.append("text_query", textQuery);
    }

    formData.append("query_type", queryType.value);
    if (queryType.value === "hybrid") {
      formData.append(
        "hybrid_weight",
        document.getElementById("hybrid-weight").value
      );
    }

    formData.append("use_pca", usePca.checked);
    formData.append("pca_components", pcaComponents.value);

    try {
      const response = await fetch("/search", {
        method: "POST",
        body: formData,
      });

      const results = await response.json();
      displayResults(results);
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred during search");
    }
  });

  function displayResults(results) {
    const resultsContainer = document.getElementById("results");
    resultsContainer.innerHTML = "";

    results.forEach((result) => {
      const resultDiv = document.createElement("div");
      resultDiv.className = "result-item";

      const img = document.createElement("img");
      img.src = `/images/${result.file_name}`;

      const similarity = document.createElement("div");
      similarity.className = "similarity";
      similarity.textContent = `Similarity: ${result.similarity.toFixed(3)}`;

      resultDiv.appendChild(img);
      resultDiv.appendChild(similarity);
      resultsContainer.appendChild(resultDiv);
    });
  }

  queryType.dispatchEvent(new Event("change"));
});

async function predictDisease() {
    const input = document.getElementById("imageInput");
    const file = input.files[0];

    if (!file) {
        alert("Please upload an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        document.getElementById("disease").innerText = data.prediction;
        document.getElementById("confidence").innerText = data.confidence;
        document.getElementById("description").innerText = data.description;

        const remediesList = document.getElementById("remedies");
        remediesList.innerHTML = "";
        data.remedies.forEach(item => {
            const li = document.createElement("li");
            li.innerText = item;
            remediesList.appendChild(li);
        });

        const careList = document.getElementById("careTips");
        careList.innerHTML = "";
        data.care_tips.forEach(item => {
            const li = document.createElement("li");
            li.innerText = item;
            careList.appendChild(li);
        });

        document.getElementById("result").classList.remove("hidden");

    } catch (error) {
        let errorMsg = "Error: Could not connect to backend.\n\n";
        
        if (error.message.includes("Failed to fetch")) {
            errorMsg += "❌ Backend server is not running.\n";
            errorMsg += "Make sure your Python server is running on http://localhost:8000";
        } else if (error.message.includes("Server error")) {
            errorMsg += "❌ Server returned an error.\n";
            errorMsg += "Check the backend logs for more details.";
        } else {
            errorMsg += error.message || "Unknown error occurred";
        }
        
        alert(errorMsg);
        console.error("Detailed error:", error);
    }
}

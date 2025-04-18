document.getElementById("stop").addEventListener("click", function() {
    fetch("/stop", { method: "POST" })
        .then(response => response.json())
        .then(data => {
            let container = document.getElementById("song-container");
            container.innerHTML = `
                <h2>Recommended Song: ${data.song}</h2>
                <img src="${data.thumbnail}" width="300">
                <p><a href="${data.url}" target="_blank">Listen on YouTube</a></p>
            `;
        });
});

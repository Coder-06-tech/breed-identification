// Get elements
const photoInput = document.getElementById('photo');
const preview = document.getElementById('animal-preview');
const aiBreed = document.getElementById('ai-breed');
const aiConfidence = document.getElementById('ai-confidence');
const aiDesc = document.getElementById('ai-breed-desc');
const aiImage = document.getElementById('ai-image');
const breedConfirm = document.getElementById('breed-confirm');
const breedResult = document.getElementById('breedResult');

// Show preview and send image to API
photoInput.addEventListener('change', () => {
    const file = photoInput.files[0];
    if (!file) return;

    // Show uploaded image preview
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';

    // Prepare form data
    const formData = new FormData();
    formData.append('image', file);

    // Call Flask API
    fetch('/api/predict', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            console.log("API response:", data); // debug

            // Make sure breedResult is visible
            breedResult.style.display = 'block';

            if (data.error) {
                aiBreed.textContent = "Error";
                aiConfidence.textContent = "—";
                aiDesc.textContent = data.error;
                aiImage.style.display = 'none';
                breedConfirm.innerHTML = '';
                return;
            }

            // Update top prediction
            aiBreed.textContent = data.top.breed;
            aiConfidence.textContent = data.top.confidence;
            aiImage.src = URL.createObjectURL(file);
            aiImage.style.display = 'block';

            // Show top 3 predictions
            aiDesc.innerHTML = "Top predictions:<br>" +
                data.predictions.map(p => `${p.breed}: ${p.confidence}%`).join('<br>');

            // Populate select dropdown for confirmation
            breedConfirm.innerHTML = '';
            data.predictions.forEach(p => {
                const option = document.createElement('option');
                option.value = p.breed;
                option.textContent = p.breed;
                if (p.breed === data.top.breed) option.selected = true;
                breedConfirm.appendChild(option);
            });
        })
        .catch(err => {
            console.error("Fetch error:", err);
            breedResult.style.display = 'block';
            aiBreed.textContent = "Error";
            aiConfidence.textContent = "—";
            aiDesc.textContent = "Prediction failed: " + err.message;
            aiImage.style.display = 'none';
            breedConfirm.innerHTML = '';
        });
});

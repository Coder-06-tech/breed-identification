// static/js/api.js

export function fetchClasses() {
    return fetch('/api/classes')
        .then(res => {
            if (!res.ok) throw new Error("Failed to load classes");
            return res.json();
        });
}

export function predictImage(fileBlob) {
    const formData = new FormData();
    formData.append('image', fileBlob);
    
    return fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(res => {
        if (!res.ok && res.status !== 422) {
            throw new Error(`Inference returned status code: ${res.status}`);
        }
        return res.json();
    });
}

export function predictImageUrl(url) {
    return fetch('/api/predict_url', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: url })
    })
    .then(res => {
        if (!res.ok && res.status !== 422) {
            throw new Error(`Inference returned status code: ${res.status}`);
        }
        return res.json();
    });
}

export function fetchBreedDetails(breedName) {
    return fetch(`/api/breed_info/${encodeURIComponent(breedName)}`)
        .then(res => {
            if (!res.ok) throw new Error("Failed to load Wikipedia info");
            return res.json();
        });
}

export function registerAnimal(formData) {
    return fetch('/api/register_animal', {
        method: 'POST',
        body: formData
    })
    .then(res => {
        if (!res.ok) {
            return res.json().then(d => { throw new Error(d.error || 'Server registration failed'); });
        }
        return res.json();
    });
}

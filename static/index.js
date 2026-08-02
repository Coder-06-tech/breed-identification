// static/index.js
import { initUI } from './js/ui.js';
import { initGestation } from './js/gestation.js';
import { initHealth } from './js/health.js';
import { getStoredRegistry, saveRegistry, renderRegistryTable, splitCamelCase } from './js/registry.js';
import { initCamera } from './js/camera.js';
import { fetchClasses, predictImage, predictImageUrl, fetchBreedDetails, registerAnimal } from './js/api.js';

// Global variables
let allBreeds = [];
let lastAnalyzedFile = null;
let lastAnalyzedUrl = null;
let hasLoadedLibrary = false;

// DOM Elements
const registrationForm = document.getElementById('registrationForm');
const photoInput = document.getElementById('photo');
const uploadArea = document.getElementById('uploadArea');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const previewContainer = document.getElementById('previewContainer');
const preview = document.getElementById('animal-preview');
const btnRemovePhoto = document.getElementById('btnRemovePhoto');

const aiLoading = document.getElementById('aiLoading');
const breedResult = document.getElementById('breedResult');
const aiBreed = document.getElementById('ai-breed');
const aiConfidence = document.getElementById('ai-confidence');
const predictionsList = document.getElementById('predictionsList');
const breedConfirm = document.getElementById('breed-confirm');

const wikiDetailBlock = document.getElementById('wikiDetailBlock');
const wikiImage = document.getElementById('wiki-image');
const wikiText = document.getElementById('wiki-text');
const wikiLink = document.getElementById('wiki-link');

const librarySearch = document.getElementById('librarySearch');
const btnLibraryClear = document.getElementById('btnLibraryClear');
const breedListUl = document.getElementById('breedListUl');
const libraryDetailsPlaceholder = document.getElementById('libraryDetailsPlaceholder');
const libraryDetailsContent = document.getElementById('libraryDetailsContent');
const libBreedTitle = document.getElementById('lib-breed-title');
const libBreedImg = document.getElementById('lib-breed-img');
const libBreedDesc = document.getElementById('lib-breed-desc');
const libBreedWiki = document.getElementById('lib-breed-wiki');

const successModal = document.getElementById('successModal');
const btnSuccessClose = document.getElementById('btnSuccessClose');
const receiptOwner = document.getElementById('receipt-owner');
const receiptPhone = document.getElementById('receipt-phone');
const receiptAnimal = document.getElementById('receipt-animal');
const receiptLoc = document.getElementById('receipt-loc');
const receiptBreed = document.getElementById('receipt-breed');
const receiptRegId = document.getElementById('receipt-reg-id');

// Bootstrap SPA Modules
initUI();
initGestation();
initHealth();

// Initial loads
loadBreedClasses();

// Custom event navigation listeners
document.addEventListener('spa-navigate', (e) => {
    const section = e.detail.section;
    if (section === 'library') {
        renderLibraryList();
    } else if (section === 'management') {
        renderRegistryTable();
    }
});

// Load standard classes list
function loadBreedClasses() {
    fetchClasses()
        .then(data => {
            if (data.classes) {
                allBreeds = data.classes;
                populateBreedDropdown(allBreeds);
            }
        })
        .catch(err => {
            console.error("Error loading classes from API:", err);
            allBreeds = ["Sahiwal", "Gir", "Jersey", "Holstein Friesian", "Murrah Buffalo"];
            populateBreedDropdown(allBreeds);
        });
}

function populateBreedDropdown(breeds, selectedBreed = "") {
    breedConfirm.innerHTML = '<option value="" disabled selected>Select confirmed breed...</option>';
    
    let listToRender = [...breeds];
    if (selectedBreed && !listToRender.includes(selectedBreed)) {
        listToRender.push(selectedBreed);
    }
    
    listToRender.forEach(breed => {
        const option = document.createElement('option');
        option.value = breed;
        option.textContent = splitCamelCase(breed);
        if (breed === selectedBreed) option.selected = true;
        breedConfirm.appendChild(option);
    });
}

// Prediction Method Tabs toggles
const predictTabBtns = document.querySelectorAll('[data-predict-tab]');
const predictTabContents = document.querySelectorAll('.predict-tab-content');

predictTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        predictTabBtns.forEach(b => b.classList.remove('active'));
        predictTabContents.forEach(c => {
            c.classList.remove('active');
            c.style.display = 'none';
        });
        
        btn.classList.add('active');
        const targetTab = btn.getAttribute('data-predict-tab');
        const targetContent = document.getElementById(`predict-tab-${targetTab}`);
        targetContent.classList.add('active');
        targetContent.style.display = 'block';
        
        breedResult.style.display = 'none';
        wikiDetailBlock.style.display = 'none';
    });
});

// Initialize Camera Module
initCamera((blob) => {
    // Show preview using canvas blob
    preview.src = URL.createObjectURL(blob);
    
    // Switch to upload preview tab
    document.querySelector('[data-predict-tab="upload"]').click();
    previewContainer.style.display = 'flex';
    uploadPlaceholder.style.display = 'none';
    
    lastAnalyzedFile = blob;
    lastAnalyzedUrl = null;
    runAIPrediction(blob);
});

// Drag & Drop highlighting
['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    }, false);
});

uploadArea.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt.files.length > 0) {
        photoInput.files = dt.files;
        handlePhotoUpload(dt.files[0]);
    }
});

uploadArea.addEventListener('click', (e) => {
    if (e.target !== btnRemovePhoto && !btnRemovePhoto.contains(e.target) && e.target !== photoInput) {
        photoInput.click();
    }
});

photoInput.addEventListener('change', () => {
    if (photoInput.files.length > 0) {
        handlePhotoUpload(photoInput.files[0]);
    }
});

function handlePhotoUpload(file) {
    preview.src = URL.createObjectURL(file);
    uploadPlaceholder.style.display = 'none';
    previewContainer.style.display = 'flex';
    
    breedResult.style.display = 'none';
    wikiDetailBlock.style.display = 'none';
    
    lastAnalyzedFile = file;
    lastAnalyzedUrl = null;
    runAIPrediction(file);
}

btnRemovePhoto.addEventListener('click', (e) => {
    e.stopPropagation();
    resetPhotoUpload();
});

function resetPhotoUpload() {
    photoInput.value = '';
    preview.src = '';
    previewContainer.style.display = 'none';
    uploadPlaceholder.style.display = 'flex';
    aiLoading.style.display = 'none';
    uploadArea.classList.remove('scanning');
    breedResult.style.display = 'none';
    wikiDetailBlock.style.display = 'none';
    lastAnalyzedFile = null;
    lastAnalyzedUrl = null;
    populateBreedDropdown(allBreeds);
}

// Perform PyTorch Local Prediction
function runAIPrediction(fileBlob) {
    aiLoading.style.display = 'flex';
    uploadArea.classList.add('scanning');
    
    predictImage(fileBlob)
    .then(data => {
        aiLoading.style.display = 'none';
        uploadArea.classList.remove('scanning');
        
        if (data.error) {
            renderPredictionError(data.error);
            return;
        }
        renderPredictionSuccess(data);
    })
    .catch(err => {
        aiLoading.style.display = 'none';
        uploadArea.classList.remove('scanning');
        renderPredictionError(err.message);
    });
}

// Paste Image URL Logic
const imageUrlInput = document.getElementById('image-url');
const btnAnalyzeUrl = document.getElementById('btnAnalyzeUrl');

if (btnAnalyzeUrl) {
    btnAnalyzeUrl.addEventListener('click', () => {
        const url = imageUrlInput.value.trim();
        if (!url) {
            alert("Please enter a valid image web URL link.");
            return;
        }
        
        aiLoading.style.display = 'flex';
        breedResult.style.display = 'none';
        wikiDetailBlock.style.display = 'none';
        
        predictImageUrl(url)
        .then(data => {
            aiLoading.style.display = 'none';
            
            preview.src = url;
            
            // Switch back to upload preview tab
            document.querySelector('[data-predict-tab="upload"]').click();
            previewContainer.style.display = 'flex';
            uploadPlaceholder.style.display = 'none';
            
            lastAnalyzedFile = null;
            lastAnalyzedUrl = url;
            
            if (data.error) {
                renderPredictionError(data.error);
                return;
            }
            renderPredictionSuccess(data);
        })
        .catch(err => {
            aiLoading.style.display = 'none';
            renderPredictionError(err.message);
        });
    });
}

function renderPredictionError(message) {
    breedResult.style.display = 'block';
    aiBreed.textContent = "Uncertain Breed";
    aiConfidence.textContent = "—";
    wikiDetailBlock.style.display = 'none';
    
    const detectedBadge = document.getElementById('detectedBadge');
    if (detectedBadge) {
        detectedBadge.classList.remove('gemini-ai');
        detectedBadge.textContent = "Top AI Prediction";
    }
    
    predictionsList.innerHTML = `
        <div class="error-msg" style="color: var(--danger); font-size: 0.9rem; padding: 0.75rem; background: var(--danger-light); border-radius: 6px; border: 1px solid #f8baba;">
            <strong>Warning:</strong> ${message}. Please select a breed manually in the dropdown below to proceed with registration, or upload a clearer photo.
        </div>
    `;
    populateBreedDropdown(allBreeds);
}

function renderPredictionSuccess(data) {
    breedResult.style.display = 'block';
    aiBreed.textContent = splitCamelCase(data.top.breed);
    aiConfidence.textContent = data.top.confidence;
    
    const detectedBadge = document.getElementById('detectedBadge');
    if (detectedBadge) {
        if (data.ai_fallback) {
            detectedBadge.classList.add('gemini-ai');
            detectedBadge.textContent = "✨ Identified by Gemini AI";
        } else {
            detectedBadge.classList.remove('gemini-ai');
            detectedBadge.textContent = "Top AI Prediction";
        }
    }
    
    predictionsList.innerHTML = '';
    data.predictions.forEach(p => {
        const item = document.createElement('div');
        item.className = 'breakdown-item';
        item.innerHTML = `
            <div class="breakdown-label">
                <span>${splitCamelCase(p.breed)}</span>
                <strong>${p.confidence}%</strong>
            </div>
            <div class="progress-track">
                <div class="progress-bar" style="width: 0%"></div>
            </div>
        `;
        predictionsList.appendChild(item);
        setTimeout(() => {
            item.querySelector('.progress-bar').style.width = `${p.confidence}%`;
        }, 100);
    });
    
    populateBreedDropdown(allBreeds, data.top.breed);
    loadWikiDetailForHome(data.top.breed);
}

function loadWikiDetailForHome(breedName) {
    wikiDetailBlock.style.display = 'block';
    wikiImage.style.display = 'none';
    wikiText.textContent = "Querying national breed library details...";
    
    fetchBreedDetails(breedName)
    .then(data => {
        if (data.image) {
            wikiImage.src = data.image;
            wikiImage.style.display = 'block';
        }
        wikiText.textContent = data.summary;
        wikiLink.href = data.url;
    })
    .catch(() => {
        wikiText.textContent = `Detailed Wikipedia information for ${splitCamelCase(breedName)} is currently loading.`;
        wikiLink.href = `https://en.wikipedia.org/wiki/Special:Search?search=${encodeURIComponent(splitCamelCase(breedName))}`;
    });
}

// Manual Override breed change update
breedConfirm.addEventListener('change', () => {
    if (breedConfirm.value) {
        wikiDetailBlock.style.display = 'block';
        loadWikiDetailForHome(breedConfirm.value);
    }
});

// Encyclopedia Library Render
function renderLibraryList() {
    if (hasLoadedLibrary && allBreeds.length > 0) return;
    
    if (allBreeds.length === 0) {
        fetchClasses()
            .then(data => {
                allBreeds = data.classes || [];
                buildLibraryUI();
            })
            .catch(() => {
                allBreeds = ["Sahiwal", "Gir", "Jersey", "Holstein Friesian", "Murrah Buffalo"];
                buildLibraryUI();
            });
    } else {
        buildLibraryUI();
    }
}

function buildLibraryUI() {
    breedListUl.innerHTML = '';
    allBreeds.forEach(breed => {
        const li = document.createElement('li');
        li.className = 'breed-list-item';
        li.setAttribute('data-breed', breed);
        li.innerHTML = `
            <span>${splitCamelCase(breed)}</span>
            <span class="material-icons-round item-arrow">chevron_right</span>
        `;
        
        li.addEventListener('click', () => {
            document.querySelectorAll('.breed-list-item').forEach(el => el.classList.remove('selected'));
            li.classList.add('selected');
            showLibraryBreedDetails(breed);
        });
        
        breedListUl.appendChild(li);
    });
    hasLoadedLibrary = true;
}

function showLibraryBreedDetails(breedName) {
    libraryDetailsPlaceholder.style.display = 'none';
    libraryDetailsContent.style.display = 'block';
    
    libBreedTitle.textContent = splitCamelCase(breedName);
    libBreedImg.src = '';
    libBreedImg.style.display = 'none';
    libBreedDesc.textContent = "Loading cattle encyclopedic facts...";
    
    fetchBreedDetails(breedName)
    .then(data => {
        if (data.image) {
            libBreedImg.src = data.image;
            libBreedImg.style.display = 'block';
        }
        libBreedDesc.textContent = data.summary;
        libBreedWiki.href = data.url;
        
        const isZebu = ["Gir", "Sahiwal", "Kankrej", "Tharparkar", "Haryana", "RedSindhi", "Deoni", "Kenwariya", "Malvi", "Nimari"].includes(breedName) || breedName.toLowerCase().includes("gir") || breedName.toLowerCase().includes("sahiwal");
        document.getElementById('lib-breed-type').textContent = isZebu ? "Bos taurus indicus (Zebu - Humped Cattle)" : "Bos taurus taurus (European / Taurine)";
    })
    .catch(() => {
        libBreedDesc.textContent = "Detailed description is currently loading from server.";
    });
}

// Library Search functionality
if (librarySearch) {
    librarySearch.addEventListener('input', () => {
        const query = librarySearch.value.toLowerCase().trim();
        const items = breedListUl.querySelectorAll('.breed-list-item');
        
        items.forEach(item => {
            const text = item.textContent.toLowerCase();
            item.style.display = text.includes(query) ? 'flex' : 'none';
        });
    });
}

if (btnLibraryClear) {
    btnLibraryClear.addEventListener('click', () => {
        librarySearch.value = '';
        const items = breedListUl.querySelectorAll('.breed-list-item');
        items.forEach(item => item.style.display = 'flex');
    });
}

// Animal Registration form submission
registrationForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const ownerName = document.getElementById('owner-name').value.trim();
    const ownerPhone = document.getElementById('onwer-no').value.trim();
    const animalAge = document.getElementById('animal-age').value.trim();
    const animalGender = document.getElementById('animal-health').value;
    const locationInput = document.getElementById('location').value.trim();
    const confirmedBreed = breedConfirm.value;
    
    if (!confirmedBreed) {
        alert("Verification required: Please confirm the cattle breed before submitting.");
        return;
    }
    
    const btnRegister = document.getElementById('btnRegister');
    btnRegister.disabled = true;
    btnRegister.innerHTML = '<span class="material-icons-round">pending</span> Saving Record...';
    
    const formData = new FormData();
    formData.append('owner', ownerName);
    formData.append('phone', ownerPhone);
    formData.append('age', animalAge);
    formData.append('gender', animalGender);
    formData.append('location', locationInput);
    formData.append('breed', confirmedBreed);
    
    if (lastAnalyzedFile) {
        formData.append('image', lastAnalyzedFile, 'cow.jpg');
    } else if (lastAnalyzedUrl) {
        formData.append('image_url', lastAnalyzedUrl);
    }
    
    registerAnimal(formData)
    .then(data => {
        btnRegister.disabled = false;
        btnRegister.innerHTML = '<span class="material-icons-round">app_registration</span> Register Animal';
        
        const record = data.record;
        
        // Save to LocalStorage registry
        const registry = getStoredRegistry();
        registry.unshift(record);
        saveRegistry(registry);
        
        // Fill Success Modal
        receiptRegId.textContent = record.regId;
        receiptOwner.textContent = record.owner;
        receiptPhone.textContent = record.phone;
        receiptAnimal.textContent = `${record.age} yrs / ${record.gender}`;
        receiptLoc.textContent = record.location;
        receiptBreed.textContent = splitCamelCase(record.breed);
        
        successModal.classList.add('open');
    })
    .catch(err => {
        btnRegister.disabled = false;
        btnRegister.innerHTML = '<span class="material-icons-round">app_registration</span> Register Animal';
        alert("Registration Failed: " + err.message);
    });
});

btnSuccessClose.addEventListener('click', () => {
    successModal.classList.remove('open');
    registrationForm.reset();
    resetPhotoUpload();
});

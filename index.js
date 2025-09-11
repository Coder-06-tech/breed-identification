
  const API_BASE = "http://localhost:8000"; // Flask server

  const breedInfo = {
    Sahiwal: {
      img: "https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Sahiwal_cow.jpg/320px-Sahiwal_cow.jpg",
      desc: "Sahiwal: Premium indigenous dairy breed, known for high milk yield and resilience."
    },
    Gir: {
      img: "https://upload.wikimedia.org/wikipedia/commons/2/26/Gir_Cow.JPG",
      desc: "Gir: Origin Gujarat. Famous for milk, strong frame, domed forehead."
    },
    "Murrah Buffalo": {
      img: "https://upload.wikimedia.org/wikipedia/commons/9/9e/Buffalo_indian.jpg",
      desc: "Murrah Buffalo: Haryana. Jet-black, curled horns, excels at milk."
    },
    Jersey: {
      img: "https://upload.wikimedia.org/wikipedia/commons/0/01/Jersey_cow_in_field.JPG",
      desc: "Jersey: Exotic breed, light brown, popular for high butterfat milk."
    },
    "Holstein Friesian": {
      img: "https://upload.wikimedia.org/wikipedia/commons/2/2f/Holstein_Friesian_cow.jpg",
      desc: "Holstein Friesian: Large size, black-white, highest global milk yield."
    }
  };

  const photoInput = document.getElementById('photo');
  const previewImg = document.getElementById('animal-preview');
  const resultBox = document.getElementById('breedResult');
  const aiBreed = document.getElementById('ai-breed');
  const aiConf = document.getElementById('ai-confidence');
  const aiImg = document.getElementById('ai-image');
  const aiDesc = document.getElementById('ai-breed-desc');
  const breedSelect = document.getElementById('breed-confirm');

  function setLoading(state) {
    if (state) {
      resultBox.style.display = "block";
      aiBreed.innerText = "Detecting…";
      aiConf.innerText = "—";
      aiImg.src = "";
      aiDesc.innerText = "Running AI model on the uploaded image…";
    }
  }

  async function callPredictAPI(file) {
    const fd = new FormData();
    fd.append("image", file);
    const res = await fetch(`${API_BASE}`/api/predict, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || HTTP `${res.status}`);
    }
    return res.json();
  }

  function updateUIFromPrediction(payload) {
    const { top, predictions } = payload;
    const topBreed = top.breed;
    const conf = top.confidence;

    aiBreed.innerText = topBreed;
    aiConf.innerText = conf;

    if (breedInfo[topBreed]) {
      aiImg.src = breedInfo[topBreed].img;
      aiImg.alt = `${topBreed}`.image;
      aiDesc.innerText = breedInfo[topBreed].desc;
    } else {
      aiImg.src = "";
      aiDesc.innerText = topBreed;
    }

    const set = new Set(predictions.map(p => p.breed));
    const all = [...predictions.map(p => p.breed), ...Object.keys(breedInfo)]
      .filter((b, i, arr) => arr.indexOf(b) === i);

    breedSelect.innerHTML = "";
    for (const b of all) {
      const opt = document.createElement("option");
      opt.value = b;
      opt.textContent = b;
      if (b === topBreed) opt.selected = true;
      breedSelect.appendChild(opt);
    }
  }

  photoInput.addEventListener('change', async function (e) {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = ev => {
      previewImg.style.display = "block";
      previewImg.src = ev.target.result;
      previewImg.alt = "Uploaded animal photo preview";
    };
    reader.readAsDataURL(file);

    setLoading(true);
    try {
      const data = await callPredictAPI(file);
      updateUIFromPrediction(data);
    } catch (err) {
      aiBreed.innerText = "Detection failed";
      aiConf.innerText = "—";
      aiImg.src = "";
      aiDesc.innerText = String(err.message || err);
      console.error(err);
    }
  });

  breedSelect.addEventListener('change', function(){
    const breed = this.value;
    aiBreed.innerText = breed;
    aiConf.innerText = "—";
    if (breedInfo[breed]) {
      aiDesc.innerText = breedInfo[breed].desc;
      aiImg.src = breedInfo[breed].img;
      aiImg.alt = breed + " image";
    } else {
      aiDesc.innerText = breed;
      aiImg.src = "";
      aiImg.alt = "";
    }
  });
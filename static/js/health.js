// static/js/health.js

export function initHealth() {
    const btnDiagnose = document.getElementById('btnDiagnose');
    const diagnosticPlaceholder = document.getElementById('diagnosticPlaceholder');
    const diagnosticContent = document.getElementById('diagnosticContent');
    const diagWarningBox = document.getElementById('diagWarningBox');
    const diagWarningText = document.getElementById('diagWarningText');
    const diagStatusCard = document.getElementById('diagStatusCard');
    const diagStatusIcon = document.getElementById('diagStatusIcon');
    const diagStatusTitle = document.getElementById('diagStatusTitle');
    const diagStatusDesc = document.getElementById('diagStatusDesc');
    const diagAdviceText = document.getElementById('diagAdviceText');

    if (!btnDiagnose) return;

    btnDiagnose.addEventListener('click', () => {
        const checkedBoxes = document.querySelectorAll('.symptom-check:checked');
        const checkedValues = Array.from(checkedBoxes).map(el => el.value);
        
        diagnosticPlaceholder.style.display = 'none';
        diagnosticContent.style.display = 'block';
        
        diagWarningBox.style.display = 'none';
        diagStatusCard.className = 'alert-status-card'; // reset classes
        
        if (checkedValues.length === 0) {
            diagStatusCard.classList.add('healthy');
            diagStatusIcon.innerHTML = '<span class="material-icons-round">check_circle</span>';
            diagStatusTitle.textContent = "Healthy Rumen & Status";
            diagStatusDesc.textContent = "No observable physical symptoms checked.";
            diagAdviceText.textContent = "Your cattle appears to be in sound health. Continue routine feed intake monitoring, vaccine charts, and shelter cleanup.";
            return;
        }
        
        // Condition mapping checks
        const hasFever = checkedValues.includes('fever');
        const hasMilkDrop = checkedValues.includes('milk_drop');
        const hasLameness = checkedValues.includes('lameness');
        const hasDrooling = checkedValues.includes('drooling');
        const hasSwelling = checkedValues.includes('swelling');
        const hasBloat = checkedValues.includes('bloated');
        
        // 1. Critical check: Foot and Mouth Disease (FMD)
        if (hasFever && hasDrooling && hasLameness) {
            diagStatusCard.classList.add('critical');
            diagStatusIcon.innerHTML = '<span class="material-icons-round">error</span>';
            diagStatusTitle.textContent = "High Risk: Foot & Mouth Disease (FMD)";
            diagStatusDesc.textContent = "Critical combinations of viral symptoms detected.";
            diagAdviceText.textContent = "Quarantine the affected animal immediately. Wash mouth lesions with 1% potassium permanganate solution. Keep hoofs dry and clean. Do not allow common grazing.";
            
            diagWarningBox.style.display = 'flex';
            diagWarningText.textContent = "Contact Block Veterinary Officer (BVO) immediately for antiviral treatment and emergency ring vaccinations.";
            return;
        }
        
        // 2. Warning check: Mastitis
        if (hasSwelling && hasMilkDrop) {
            diagStatusCard.classList.add('warning');
            diagStatusIcon.innerHTML = '<span class="material-icons-round">warning</span>';
            diagStatusTitle.textContent = "Moderate Risk: Mastitis (Udder Infection)";
            diagStatusDesc.textContent = "Symptomatic metrics matching bacterial udder inflammation.";
            diagAdviceText.textContent = "Thoroughly clean the udder with antiseptic wash prior to milking. Practice dry cow therapy. Perform strip-cup tests on milk columns.";
            
            diagWarningBox.style.display = 'flex';
            diagWarningText.textContent = "Consider consulting a veterinarian for an intramammary antibiotic infusion chart if swelling persists.";
            return;
        }
        
        // 3. Warning check: Rumen Bloat
        if (hasBloat) {
            diagStatusCard.classList.add('warning');
            diagStatusIcon.innerHTML = '<span class="material-icons-round">warning</span>';
            diagStatusTitle.textContent = "Moderate Risk: Rumen Tympany (Bloat)";
            diagStatusDesc.textContent = "Rumen gas accumulation causing abdomen inflation.";
            diagAdviceText.textContent = "Administer anti-foaming carminative mixtures (like turpentine oil with linseed oil). Keep cattle walking. Avoid feeding high-legume wet grass forage.";
            
            diagWarningBox.style.display = 'flex';
            diagWarningText.textContent = "If severe gas creates acute respiratory distress, emergency trocarization of the left flank may be necessary.";
            return;
        }
        
        // General minor symptom warnings
        diagStatusCard.classList.add('warning');
        diagStatusIcon.innerHTML = '<span class="material-icons-round">info</span>';
        diagStatusTitle.textContent = "General Symptomatic Alert";
        diagStatusDesc.textContent = `${checkedValues.length} isolated indicators selected.`;
        diagAdviceText.textContent = "Keep the cattle under observation for 24-48 hours. Ensure isolation from calves and other healthy animals. Ensure clean, soft drinking water is readily accessible.";
    });
}

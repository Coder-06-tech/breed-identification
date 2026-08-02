// static/js/gestation.js

export function initGestation() {
    const btnCalculateBreeding = document.getElementById('btnCalculateBreeding');
    const breedingDateInput = document.getElementById('breeding-date');
    const breedingPlaceholder = document.getElementById('breedingPlaceholder');
    const breedingContent = document.getElementById('breedingContent');
    const expectedCalvingDateStr = document.getElementById('expectedCalvingDate');
    const daysRemainingBadge = document.getElementById('daysRemainingBadge');
    const gestationPercentStr = document.getElementById('gestationPercent');
    const gestationProgressBar = document.getElementById('gestationProgressBar');
    const timelineMatingDate = document.getElementById('timeline-mating-date');
    const timelinePdDate = document.getElementById('timeline-pd-date');
    const timelineDryDate = document.getElementById('timeline-dry-date');
    const timelineCalvingDate = document.getElementById('timeline-calving-date');

    if (!btnCalculateBreeding) return;

    btnCalculateBreeding.addEventListener('click', () => {
        const rawDate = breedingDateInput.value;
        if (!rawDate) {
            alert("Please enter a valid mating/insemination date.");
            return;
        }
        
        breedingPlaceholder.style.display = 'none';
        breedingContent.style.display = 'block';
        
        const matingDate = new Date(rawDate);
        const currentDate = new Date();
        
        // Gestation standard is 283 days for cows
        const GESTATION_DAYS = 283;
        
        // Milestone intervals
        const pdDays = 60;
        const dryDays = 220;
        
        // Compute milestone dates
        const pdDate = new Date(matingDate.getTime() + pdDays * 24 * 60 * 60 * 1000);
        const dryDate = new Date(matingDate.getTime() + dryDays * 24 * 60 * 60 * 1000);
        const calvingDate = new Date(matingDate.getTime() + GESTATION_DAYS * 24 * 60 * 60 * 1000);
        
        // Display dates formatted
        const options = { day: '2-digit', month: 'short', year: 'numeric' };
        expectedCalvingDateStr.textContent = calvingDate.toLocaleDateString('en-IN', options);
        
        timelineMatingDate.textContent = `Mated on ${matingDate.toLocaleDateString('en-IN', options)}`;
        timelinePdDate.textContent = `Expected Pregnancy Diagnosis: ${pdDate.toLocaleDateString('en-IN', options)}`;
        timelineDryDate.textContent = `Expected Dry-off period start: ${dryDate.toLocaleDateString('en-IN', options)}`;
        timelineCalvingDate.textContent = `Expected Calving date: ${calvingDate.toLocaleDateString('en-IN', options)}`;
        
        // Calculate progress
        const elapsedMs = currentDate - matingDate;
        const elapsedDays = Math.max(0, Math.floor(elapsedMs / (1000 * 60 * 60 * 24)));
        
        let percent = Math.min(100, Math.round((elapsedDays / GESTATION_DAYS) * 100));
        if (percent < 0) percent = 0;
        
        gestationPercentStr.textContent = `${percent}%`;
        gestationProgressBar.style.width = `${percent}%`;
        
        // Days remaining till calving
        const remainingMs = calvingDate - currentDate;
        const remainingDays = Math.max(0, Math.ceil(remainingMs / (1000 * 60 * 60 * 24)));
        
        if (percent >= 100) {
            daysRemainingBadge.textContent = "Delivery Overdue / Calved";
            daysRemainingBadge.style.backgroundColor = "var(--success)";
            daysRemainingBadge.style.color = "white";
        } else {
            daysRemainingBadge.textContent = `${remainingDays} Days Left`;
            daysRemainingBadge.style.backgroundColor = "var(--accent-light)";
            daysRemainingBadge.style.color = "var(--accent-dark)";
        }
    });
}

// static/js/ui.js

export function initUI() {
    // 1. SPA ROUTING & NAVIGATION
    const navBtns = document.querySelectorAll('.nav-btn');
    const spaSections = document.querySelectorAll('.spa-section');
    
    navBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            navBtns.forEach(b => b.classList.remove('active'));
            spaSections.forEach(s => s.classList.remove('active'));
            
            btn.classList.add('active');
            const targetSection = btn.getAttribute('data-section');
            document.getElementById(`section-${targetSection}`).classList.add('active');
            
            // Dispatch custom navigation event for other modules
            const event = new CustomEvent('spa-navigate', { detail: { section: targetSection } });
            document.dispatchEvent(event);
            
            // Collapse mobile menu if open
            const headerNav = document.getElementById('navLinks');
            if (headerNav) headerNav.classList.remove('open');
        });
    });

    // Mobile menu toggle
    const menuToggle = document.getElementById('menuToggle');
    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            document.getElementById('navLinks').classList.toggle('open');
        });
    }

    // 2. GUIDELINES MODAL
    const btnShowGuidelines = document.getElementById('btnShowGuidelines');
    const guidelinesModal = document.getElementById('guidelinesModal');
    const btnCloseGuidelines = document.getElementById('btnCloseGuidelines');
    const btnAcceptGuidelines = document.getElementById('btnAcceptGuidelines');

    if (btnShowGuidelines && guidelinesModal) {
        btnShowGuidelines.addEventListener('click', (e) => {
            e.preventDefault();
            guidelinesModal.classList.add('open');
        });
        
        const closeMod = () => guidelinesModal.classList.remove('open');
        if (btnCloseGuidelines) btnCloseGuidelines.addEventListener('click', closeMod);
        if (btnAcceptGuidelines) btnAcceptGuidelines.addEventListener('click', closeMod);
    }

    // 3. FAQ ACCORDIONS (SUPPORT)
    const faqQuestions = document.querySelectorAll('.faq-question');
    faqQuestions.forEach(q => {
        q.addEventListener('click', () => {
            const item = q.parentElement;
            const wasActive = item.classList.contains('active');
            
            document.querySelectorAll('.faq-item').forEach(i => i.classList.remove('active'));
            if (!wasActive) {
                item.classList.add('active');
            }
        });
    });
}

// static/js/registry.js

export function splitCamelCase(str) {
    if (!str) return "";
    return str.replace(/([a-z])([A-Z])/g, '$1 $2');
}

export function getStoredRegistry() {
    const data = localStorage.getItem('registered_cattle');
    return data ? JSON.parse(data) : [];
}

export function saveRegistry(arr) {
    localStorage.setItem('registered_cattle', JSON.stringify(arr));
}

export function renderRegistryTable() {
    const registryTableBody = document.getElementById('registryTableBody');
    const registryCountBadge = document.getElementById('registryCountBadge');
    
    if (!registryTableBody) return;

    const registry = getStoredRegistry();
    if (registryCountBadge) {
        registryCountBadge.textContent = `${registry.length} Registered`;
    }
    
    registryTableBody.innerHTML = '';
    if (registry.length === 0) {
        registryTableBody.innerHTML = `
            <tr>
                <td colspan="7" class="empty-table-msg">No cattle registered yet. Go to the Home tab to add a record.</td>
            </tr>
        `;
        return;
    }
    
    registry.forEach((item, index) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${item.regId}</strong></td>
            <td>
                <div><strong>${item.owner}</strong></div>
                <div style="font-size:0.8rem; color:var(--text-light);">${item.phone}</div>
            </td>
            <td>${item.age} yrs / ${item.gender}</td>
            <td>${item.location}</td>
            <td><span class="upload-badge">${splitCamelCase(item.breed)}</span></td>
            <td>${item.date}</td>
            <td>
                <button type="button" class="btn-delete" title="Delete Profile" data-index="${index}">
                     <span class="material-icons-round">delete</span>
                </button>
            </td>
        `;
        
        const deleteBtn = tr.querySelector('.btn-delete');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => {
                if (confirm(`Are you sure you want to delete registration record ${item.regId}?`)) {
                    deleteRegistryRecord(index);
                }
            });
        }
        
        registryTableBody.appendChild(tr);
    });
}

export function deleteRegistryRecord(index) {
    const registry = getStoredRegistry();
    registry.splice(index, 1);
    saveRegistry(registry);
    renderRegistryTable();
}

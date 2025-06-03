let currentDocument = null;
let validationResults = {};

// Load document list on page load
document.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
});

async function loadDocuments() {
    const response = await fetch('/api/documents');
    const documents = await response.json();
    
    const listEl = document.getElementById('documentList');
    listEl.innerHTML = documents.map(doc => `
        <div class="document-item" onclick="selectDocument(${doc.document_id})">
            <div class="document-title">${doc.doc_title || 'Untitled'}</div>
            <div class="document-info">
                ${doc.date_formatted || 'No date'} • ${doc.amici_count} amici
            </div>
        </div>
    `).join('');
}

async function selectDocument(docId) {
    // Update UI
    document.querySelectorAll('.document-item').forEach(el => el.classList.remove('active'));
    event.target.closest('.document-item').classList.add('active');
    
    // Load document details
    const response = await fetch(`/api/document/${docId}`);
    const data = await response.json();
    currentDocument = data;
    
    // Load validation results from database
    validationResults = {};
    data.amici.forEach(amicus => {
        const validated = currentDocument.validation_stats.validated;
        if (validated > 0) {
            // This is simplified - in production you'd fetch actual validation status per amicus
            validationResults[amicus.amicus_id] = null;
        }
    });
    
    // Display metadata
    displayMetadata(data);
    
    // Load PDF
    loadPDF(docId);
    
    // Display entities
    displayAmici(data.amici);
    displayLawyers(data.lawyers);
    loadMissedAmici(docId);
    
    // Update stats
    updateStats(data.validation_stats);
}

function displayMetadata(data) {
    const metadataEl = document.getElementById('metadata');
    const doc = data.document;
    const docket = data.docket;
    
    metadataEl.innerHTML = `
        <h3>Document Metadata</h3>
        <p><strong>Date:</strong> ${doc.date_formatted || 'Unknown'}</p>
        <p><strong>URL:</strong> <a href="${doc.url}" target="_blank">View Original</a></p>
        ${docket ? `
            <p><strong>Docket:</strong> ${docket.year}-${docket.number}</p>
            <p><strong>Position:</strong> ${docket.position}</p>
        ` : ''}
        <p><strong>Counsel of Record:</strong> ${doc.counsel_of_record || 'Not specified'}</p>
        <p><strong>OCR Used:</strong> ${doc.neededOCR ? 'Yes' : 'No'}</p>
    `;
}

function loadPDF(docId) {
    const viewerEl = document.getElementById('pdfViewer');
    viewerEl.innerHTML = `<iframe src="/api/document/${docId}/pdf"></iframe>`;
}

function displayAmici(amici) {
    const listEl = document.getElementById('amiciList');
    listEl.innerHTML = amici.map(amicus => {
        const status = validationResults[amicus.amicus_id];
        const validated = status !== undefined;
        
        return `
            <div class="entity-item ${validated ? 'validated' : ''}">
                <div class="entity-name">
                    ${amicus.name}
                    ${validated ? `
                        <span class="validation-status ${status ? 'status-correct' : 'status-incorrect'}">
                            ${status ? '✓ Correct' : '✗ Incorrect'}
                        </span>
                    ` : ''}
                </div>
                ${amicus.category ? `<div class="entity-category">${amicus.category}</div>` : ''}
                <div class="validation-buttons">
                    <button class="btn btn-correct" onclick="validateAmicus(${amicus.amicus_id}, true)">
                        Correct
                    </button>
                    <button class="btn btn-incorrect" onclick="validateAmicus(${amicus.amicus_id}, false)">
                        Incorrect
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function displayLawyers(lawyers) {
    const listEl = document.getElementById('lawyersList');
    listEl.innerHTML = lawyers.map(lawyer => `
        <div class="lawyer-item">
            <div class="lawyer-name">${lawyer.name}</div>
            <div class="lawyer-info">
                ${lawyer.role || 'Attorney'}
                ${lawyer.employer ? ` • ${lawyer.employer}` : ''}
                ${lawyer.is_counsel_of_record ? ' • Counsel of Record' : ''}
            </div>
        </div>
    `).join('');
}

async function validateAmicus(amicusId, isCorrect) {
    const response = await fetch('/api/validate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            document_id: currentDocument.document.document_id,
            amicus_id: amicusId,
            is_correct: isCorrect
        })
    });
    
    if (response.ok) {
        validationResults[amicusId] = isCorrect;
        displayAmici(currentDocument.amici);
        
        // Update stats
        const stats = currentDocument.validation_stats;
        if (validationResults[amicusId] === undefined) {
            stats.validated++;
        }
        if (isCorrect) {
            stats.correct++;
        } else {
            stats.incorrect++;
        }
        updateStats(stats);
    }
}

async function addMissedAmicus() {
    const name = document.getElementById('missedName').value.trim();
    const category = document.getElementById('missedCategory').value.trim();
    
    if (!name) return;
    
    const response = await fetch('/api/add-missed', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            document_id: currentDocument.document.document_id,
            name: name,
            category: category
        })
    });
    
    if (response.ok) {
        document.getElementById('missedName').value = '';
        document.getElementById('missedCategory').value = '';
        loadMissedAmici(currentDocument.document.document_id);
        
        // Update stats
        currentDocument.validation_stats.missed++;
        updateStats(currentDocument.validation_stats);
    }
}

async function loadMissedAmici(docId) {
    const response = await fetch(`/api/missed/${docId}`);
    const missed = await response.json();
    
    const listEl = document.getElementById('missedList');
    listEl.innerHTML = missed.map(m => `
        <div class="entity-item">
            <div class="entity-name">${m.name}</div>
            ${m.category ? `<div class="entity-category">${m.category}</div>` : ''}
        </div>
    `).join('');
}

function updateStats(stats) {
    const statsEl = document.getElementById('stats');
    const total = stats.total;
    const accuracy = stats.validated > 0 
        ? ((stats.correct / stats.validated) * 100).toFixed(1)
        : 'N/A';
    
    statsEl.innerHTML = `
        <div class="stat-row">
            <span class="stat-label">Total Extracted:</span>
            <span class="stat-value">${total}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Validated:</span>
            <span class="stat-value">${stats.validated} / ${total}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Correct:</span>
            <span class="stat-value">${stats.correct}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Incorrect:</span>
            <span class="stat-value">${stats.incorrect}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Missed:</span>
            <span class="stat-value">${stats.missed}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Accuracy:</span>
            <span class="stat-value">${accuracy}%</span>
        </div>
    `;
}

function switchTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    event.target.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.getElementById(`${tab}Tab`).classList.add('active');
}
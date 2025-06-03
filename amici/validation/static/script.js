// Add this at the very top to see console errors
window.addEventListener('error', (e) => {
    console.error('Global error:', e);
});

let currentDocument = null;
let pdfDoc = null;
let currentPage = 1;
let documents = [];

// Initialize PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

// Add this right after the pdfjsLib worker setup
console.log('Script loaded, initializing...');

// Load documents on page load
window.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
});

async function loadDocuments() {
    console.log('Loading documents...');
    try {
        const response = await fetch('/api/documents');
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        documents = await response.json();
        console.log('Loaded documents:', documents.length);
        console.log('First few documents:', documents.slice(0, 3));
        
        const select = document.getElementById('documentSelect');
        select.innerHTML = '<option value="">Select a document...</option>';
        
        documents.forEach(doc => {
            const option = document.createElement('option');
            option.value = doc.document_id;
            option.textContent = `${doc.label || `Doc ${doc.document_id}`} (${doc.case_id || 'No match'})`;
            if (doc.is_confirmed) {
                option.textContent += ' ✓';
            }
            select.appendChild(option);
        });
        
        // Load first document if any
        if (documents.length > 0) {
            console.log('Loading first document:', documents[0].document_id);
            loadDocument(documents[0].document_id);
        } else {
            console.warn('No documents found!');
            // Show a message to the user
            document.getElementById('documentInfo').innerHTML = 
                '<span class="error">No documents found. Check if complete_amici_list = 1 for any documents.</span>';
        }
    } catch (error) {
        console.error('Error loading documents:', error);
        document.getElementById('documentInfo').innerHTML = 
            `<span class="error">Error loading documents: ${error.message}</span>`;
    }
}

async function loadDocument(docId) {
    if (!docId) return;
    
    try {
        // Update select
        document.getElementById('documentSelect').value = docId;
        
        // Load document data
        const response = await fetch(`/api/document/${docId}`);
        const data = await response.json();
        currentDocument = data;
        
        // Update document info
        const info = data.document;
        document.getElementById('documentInfo').innerHTML = `
            <strong>Document:</strong> ${info.label || `Document ${info.document_id}`} | 
            <strong>Date:</strong> ${info.date_formatted} | 
            <strong>Docket:</strong> 20${info.year}-${String(info.number).padStart(3, '0')} | 
            <strong>Match:</strong> ${info.case_id || 'None'}
        `;
        
        // Update match info
        document.getElementById('matchInfo').textContent = 
            info.case_id ? `Matched to spreadsheet row: ${info.case_id}` : 'No match';
        
        // Load entities
        loadEntities(data);
        
        // Load PDF
        loadPDF(docId);
        
    } catch (error) {
        console.error('Error loading document:', error);
    }
}

function loadEntities(data) {
    const markedEntities = data.marked_entities || [];
    const markedMap = {};
    markedEntities.forEach(m => {
        markedMap[`${m.entity_type}:${m.entity_name}`] = true;
    });
    
    // Load their entities
    const theirContainer = document.getElementById('theirEntities');
    theirContainer.innerHTML = '';
    
    data.their_amici.forEach(entity => {
        const div = document.createElement('div');
        div.className = 'entity-item';
        
        const isMarked = markedMap[`missed_by_me:${entity}`];
        if (isMarked) {
            div.classList.add('marked', 'missed-by-me');
        }
        
        div.innerHTML = `
            <input type="checkbox" class="entity-checkbox" data-entity="${entity}" data-type="their">
            <span>${entity}</span>
        `;
        
        // Add click handler to toggle marking
        div.addEventListener('dblclick', () => toggleMark(entity, 'missed_by_me'));
        
        theirContainer.appendChild(div);
    });
    
    // Load my entities
    const myContainer = document.getElementById('myEntities');
    myContainer.innerHTML = '';
    
    data.my_amici.forEach(entity => {
        const div = document.createElement('div');
        div.className = 'entity-item';
        
        const isMarked = markedMap[`missed_by_them:${entity}`];
        if (isMarked) {
            div.classList.add('marked', 'missed-by-them');
        }
        
        div.innerHTML = `
            <input type="checkbox" class="entity-checkbox" data-entity="${entity}" data-type="my">
            <span>${entity}</span>
        `;
        
        // Add click handler to toggle marking
        div.addEventListener('dblclick', () => toggleMark(entity, 'missed_by_them'));
        
        myContainer.appendChild(div);
    });
}

async function toggleMark(entityName, entityType) {
    if (!currentDocument) return;
    
    const markedEntities = currentDocument.marked_entities || [];
    const isMarked = markedEntities.some(m => 
        m.entity_name === entityName && m.entity_type === entityType
    );
    
    const endpoint = isMarked ? '/api/unmark_entity' : '/api/mark_entity';
    
    try {
        await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                document_id: currentDocument.document.document_id,
                entity_name: entityName,
                entity_type: entityType
            })
        });
        
        // Reload document to refresh UI
        loadDocument(currentDocument.document.document_id);
    } catch (error) {
        console.error('Error toggling mark:', error);
    }
}

async function markSelected(type) {
    if (!currentDocument) return;
    
    const checkboxes = document.querySelectorAll(
        `.entity-checkbox[data-type="${type}"]:checked`
    );
    
    const entityType = type === 'their' ? 'missed_by_me' : 'missed_by_them';
    
    for (const checkbox of checkboxes) {
        const entityName = checkbox.dataset.entity;
        
        try {
            await fetch('/api/mark_entity', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    document_id: currentDocument.document.document_id,
                    entity_name: entityName,
                    entity_type: entityType
                })
            });
        } catch (error) {
            console.error('Error marking entity:', error);
        }
    }
    
    // Reload document to refresh UI
    loadDocument(currentDocument.document.document_id);
}

async function loadPDF(docId) {
    try {
        const canvas = document.getElementById('pdfCanvas');
        const context = canvas.getContext('2d');
        
        // Clear canvas
        context.clearRect(0, 0, canvas.width, canvas.height);
        canvas.width = 0;
        canvas.height = 0;
        
        // Load PDF
        const pdfData = await fetch(`/api/document/${docId}/pdf`);
        const arrayBuffer = await pdfData.arrayBuffer();
        
        pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        currentPage = 1;
        
        renderPage(currentPage);
        
    } catch (error) {
        console.error('Error loading PDF:', error);
        document.getElementById('pdfViewer').innerHTML = 
            '<div class="error">Error loading PDF</div>';
    }
}

async function renderPage(pageNum) {
    if (!pdfDoc) return;
    
    try {
        const page = await pdfDoc.getPage(pageNum);
        const canvas = document.getElementById('pdfCanvas');
        const context = canvas.getContext('2d');
        
        // Calculate scale to fit width
        const containerWidth = canvas.parentElement.offsetWidth - 40;
        const viewport = page.getViewport({ scale: 1 });
        const scale = containerWidth / viewport.width;
        const scaledViewport = page.getViewport({ scale });
        
        canvas.height = scaledViewport.height;
        canvas.width = scaledViewport.width;
        
        const renderContext = {
            canvasContext: context,
            viewport: scaledViewport
        };
        
        await page.render(renderContext).promise;
        
        // Update page info
        document.getElementById('pageInfo').textContent = 
            `Page ${pageNum} of ${pdfDoc.numPages}`;
            
    } catch (error) {
        console.error('Error rendering page:', error);
    }
}

function changePage(delta) {
    if (!pdfDoc) return;
    
    const newPage = currentPage + delta;
    if (newPage >= 1 && newPage <= pdfDoc.numPages) {
        currentPage = newPage;
        renderPage(currentPage);
   }
}

async function cycleMatch() {
   if (!currentDocument) return;
   
   try {
       const response = await fetch(`/api/cycle_match/${currentDocument.document.document_id}`);
       const data = await response.json();
       
       if (response.ok) {
           // Show preview of next match
           const confirmMessage = `Switch to match:\nCase ID: ${data.case_id}\nSupport: ${data.support_n}\nEntities: ${data.their_amici.slice(0, 3).join(', ')}${data.their_amici.length > 3 ? '...' : ''}`;
           
           if (confirm(confirmMessage)) {
               // Update match
               await fetch('/api/update_match', {
                   method: 'POST',
                   headers: { 'Content-Type': 'application/json' },
                   body: JSON.stringify({
                       document_id: currentDocument.document.document_id,
                       spreadsheet_row_index: data.spreadsheet_row_index,
                       case_id: data.case_id
                   })
               });
               
               // Reload document
               loadDocument(currentDocument.document.document_id);
           }
       } else {
           alert(data.error || 'No alternative matches found');
       }
   } catch (error) {
       console.error('Error cycling match:', error);
       alert('Error finding alternative match');
   }
}

function navigateDocument(delta) {
   if (!currentDocument || documents.length === 0) return;
   
   const currentIndex = documents.findIndex(d => 
       d.document_id === currentDocument.document.document_id
   );
   
   const newIndex = currentIndex + delta;
   if (newIndex >= 0 && newIndex < documents.length) {
       loadDocument(documents[newIndex].document_id);
   }
}

async function exportResults() {
   try {
       const response = await fetch('/api/export_results');
       const results = await response.json();
       
       // Create downloadable JSON file
       const blob = new Blob([JSON.stringify(results, null, 2)], 
           { type: 'application/json' });
       const url = URL.createObjectURL(blob);
       const a = document.createElement('a');
       a.href = url;
       a.download = `amici_validation_results_${new Date().toISOString().split('T')[0]}.json`;
       document.body.appendChild(a);
       a.click();
       document.body.removeChild(a);
       URL.revokeObjectURL(url);
       
       alert(`Exported ${results.length} comparison results`);
   } catch (error) {
       console.error('Error exporting results:', error);
       alert('Error exporting results');
   }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
   if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
   
   switch(e.key) {
       case 'ArrowLeft':
           if (e.ctrlKey) {
               navigateDocument(-1);
           } else {
               changePage(-1);
           }
           break;
       case 'ArrowRight':
           if (e.ctrlKey) {
               navigateDocument(1);
           } else {
               changePage(1);
           }
           break;
       case 'c':
           if (e.ctrlKey) {
               cycleMatch();
           }
           break;
   }
});
// src/GérerDocument.ts
console.log("Gérer Document script loaded.");
// --- Helper Functions ---
function showError(inputId, message) {
    const inputElement = document.getElementById(inputId);
    if (!inputElement) {
        console.warn(`showError: Element with ID '${inputId}' not found.`);
        return;
    }
    const formGroup = inputElement.closest('.form-group');
    const errorSpan = formGroup === null || formGroup === void 0 ? void 0 : formGroup.querySelector('.error-message');
    inputElement.classList.add('invalid');
    inputElement.setAttribute('aria-invalid', 'true');
    const errorId = `error-${inputId}`;
    inputElement.setAttribute('aria-describedby', errorId);
    if (errorSpan) {
        errorSpan.textContent = message;
        errorSpan.classList.add('visible');
        errorSpan.id = errorId;
    }
    else {
        console.warn(`showError: No .error-message span found near #${inputId}`);
    }
}
function clearError(inputId) {
    const inputElement = document.getElementById(inputId);
    if (!inputElement) {
        console.warn(`clearError: Element with ID '${inputId}' not found.`);
        return;
    }
    const formGroup = inputElement.closest('.form-group');
    const errorSpan = formGroup === null || formGroup === void 0 ? void 0 : formGroup.querySelector('.error-message');
    inputElement.classList.remove('invalid');
    inputElement.removeAttribute('aria-invalid');
    inputElement.removeAttribute('aria-describedby');
    if (errorSpan) {
        errorSpan.textContent = '';
        errorSpan.classList.remove('visible');
    }
}
// --- End Helper Functions ---
// --- DOM Elements ---
const addDocForm = document.getElementById('addDocForm');
const addModuleSelect = document.getElementById('addModule');
const addFileInput = document.getElementById('addFile');
const documentsTableBody = document.getElementById('documentsTableBody');
const logoutBtn = document.getElementById('logoutBtnManage');
// --- Available Modules ---
const availableModules = ["Algorithme", "Java", "C++", "Base de données", "JEE", "C", "Réseau", "PLSQL"];
// --- Helper Function to Create Table Row (Revised for Robustness) ---
function createTableRow(docId, docName, docModule) {
    try {
        const tr = document.createElement('tr');
        tr.setAttribute('data-doc-id', docId);
        tr.setAttribute('data-doc-name', docName);
        tr.setAttribute('data-doc-module', docModule);
        // File Name Cell
        const nameTd = document.createElement('td');
        nameTd.className = 'px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900';
        const filenameDisplaySpan = document.createElement('span');
        filenameDisplaySpan.className = 'filename-display';
        filenameDisplaySpan.textContent = docName;
        const filenameEditInput = document.createElement('input');
        filenameEditInput.type = 'file';
        filenameEditInput.accept = '.pdf';
        filenameEditInput.className = 'filename-edit-input hidden'; // Use correct class
        nameTd.appendChild(filenameDisplaySpan);
        nameTd.appendChild(filenameEditInput);
        // Module Cell
        const moduleTd = document.createElement('td');
        moduleTd.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
        const moduleDisplaySpan = document.createElement('span');
        moduleDisplaySpan.className = 'module-display';
        moduleDisplaySpan.textContent = docModule;
        const moduleEditSelect = document.createElement('select');
        moduleEditSelect.className = 'module-edit hidden w-full px-2 py-1 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-manage-interactive focus:border-manage-interactive text-sm';
        availableModules.forEach(mod => {
            const option = document.createElement('option');
            option.value = mod;
            option.textContent = mod;
            if (mod === docModule)
                option.selected = true;
            moduleEditSelect.appendChild(option);
        });
        moduleTd.appendChild(moduleDisplaySpan);
        moduleTd.appendChild(moduleEditSelect);
        // Actions Cell
        const actionsTd = document.createElement('td');
        actionsTd.className = 'px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2';
        // Ensure SVGs are valid and icons have pointer-events: none (in CSS or inline)
        actionsTd.innerHTML = `
            <button class="edit-btn table-button" title="Modifier Fichier/Module">
                <svg class="icon text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path></svg>
            </button>
            <button class="save-btn table-button hidden" title="Sauvegarder Changements Locaux">
                <svg class="icon text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
            </button>
            <button class="cancel-btn table-button hidden" title="Annuler">
                <svg class="icon text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
            </button>
            <button class="delete-btn table-button" title="Supprimer">
                <svg class="icon text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
            </button>
        `;
        tr.appendChild(nameTd);
        tr.appendChild(moduleTd);
        tr.appendChild(actionsTd);
        return tr;
    }
    catch (error) {
        console.error("Error in createTableRow:", error);
        return null;
    }
}
// --- Function to Load Documents ---
function loadDocuments() {
    console.log("Loading documents...");
    if (!documentsTableBody) {
        console.error("Table body 'documentsTableBody' not found!");
        return;
    }
    const simulatedDocs = [
        { id: "doc1", name: "cours_C++_chap3.pdf", module: "C++" },
        { id: "doc2", name: "cours_java_chap2.pdf", module: "Java" },
        { id: "doc3", name: "cours_DBA_chap5.pdf", module: "Base de données" },
        { id: "doc4", name: "cours_plsql_chap5.pdf", module: "PLSQL" },
        { id: "doc5", name: "cours_reseau_chap1.pdf", module: "Réseau" },
    ];
    documentsTableBody.innerHTML = '';
    if (simulatedDocs.length === 0) {
        const tr = documentsTableBody.insertRow();
        const td = tr.insertCell();
        td.textContent = "Aucun document trouvé.";
        td.colSpan = 3;
        td.className = 'px-6 py-4 text-center text-sm text-gray-500';
    }
    else {
        simulatedDocs.forEach(doc => {
            const row = createTableRow(doc.id, doc.name, doc.module);
            if (row) {
                documentsTableBody.appendChild(row);
            }
        });
    }
    console.log(`Loaded ${simulatedDocs.length} simulated documents.`);
}
// --- Event Listeners ---
// Logout Button
if (logoutBtn) {
    logoutBtn.addEventListener('click', () => { console.log("Logout clicked"); alert("Déconnexion (simulation)..."); window.location.href = './index.html'; });
}
else {
    console.warn("Logout button 'logoutBtnManage' not found.");
}
// Add Document Form Submission (FIXED TypeScript null errors)
if (addDocForm && addModuleSelect && addFileInput && documentsTableBody) {
    addDocForm.addEventListener('submit', (e) => {
        var _a;
        e.preventDefault();
        console.log("Add form submitted.");
        const module = addModuleSelect.value;
        const files = addFileInput.files;
        const file = files && files.length > 0 ? files[0] : null;
        // Clear previous errors
        clearError('addModule');
        clearError('addFile');
        // Validation
        let formValid = true;
        if (!module) {
            showError("addModule", "Veuillez choisir un module.");
            formValid = false;
        }
        if (!file) { // Check if file is null
            showError("addFile", "Veuillez choisir un fichier PDF.");
            formValid = false;
        }
        else if (file.type !== "application/pdf") {
            showError("addFile", "Le fichier doit être au format PDF.");
            formValid = false;
        }
        if (!formValid) {
            console.warn("Add form validation failed.");
            const firstInvalid = addDocForm.querySelector('.invalid');
            firstInvalid === null || firstInvalid === void 0 ? void 0 : firstInvalid.focus();
            return; // Stop if invalid
        }
        // ***** If validation passed, 'file' is definitely not null here *****
        // Use non-null assertion operator (!) to inform TypeScript
        console.log(`Form valid. Adding document: ${file.name}, Module: ${module}`); // FIXED: Added '!'
        // --- Simulation: Create and Add Row ---
        try {
            const newId = `sim-${Date.now()}`;
            // Use non-null assertion here as well
            const newRow = createTableRow(newId, file.name, module); // FIXED: Added '!'
            if (newRow) {
                const noDocsRow = (_a = documentsTableBody.querySelector('td[colspan="3"]')) === null || _a === void 0 ? void 0 : _a.closest('tr');
                if (noDocsRow) {
                    noDocsRow.remove();
                }
                documentsTableBody.appendChild(newRow);
                // Use non-null assertion here as well
                console.log(`Row for ${file.name} added to table.`); // FIXED: Added '!'
                addDocForm.reset();
                addFileInput.value = ''; // Clear file input display
            }
            else {
                console.error("Failed to create table row for new document.");
                alert("Erreur lors de l'ajout de la ligne au tableau.");
            }
        }
        catch (error) {
            console.error("Error during add row simulation:", error);
            alert("Erreur lors de l'ajout du document (simulation).");
        }
        // --- End Simulation ---
        // TODO: Actual backend upload logic would go here.
    });
}
else {
    console.error("One or more elements for the 'Add Document' form are missing.");
    if (!addDocForm)
        console.error("addDocForm is missing");
    if (!addModuleSelect)
        console.error("addModuleSelect is missing");
    if (!addFileInput)
        console.error("addFileInput is missing");
    if (!documentsTableBody)
        console.error("documentsTableBody is missing");
}
// --- Table Actions Event Delegation (Reviewed and Robust) ---
if (documentsTableBody) {
    documentsTableBody.addEventListener('click', (e) => {
        const target = e.target;
        const button = target.closest('button.table-button');
        if (!button)
            return;
        const row = button.closest('tr');
        if (!row) {
            console.warn("Could not find parent row for clicked button.");
            return;
        }
        // Get elements robustly using querySelector within the specific row
        const filenameDisplay = row.querySelector('.filename-display');
        const filenameEditInput = row.querySelector('.filename-edit-input');
        const moduleDisplay = row.querySelector('.module-display');
        const moduleEditSelect = row.querySelector('.module-edit');
        const editBtn = row.querySelector('.edit-btn');
        const saveBtn = row.querySelector('.save-btn');
        const cancelBtn = row.querySelector('.cancel-btn');
        const deleteBtn = row.querySelector('.delete-btn');
        // Crucial check: Ensure all elements are found before proceeding
        if (!filenameDisplay || !filenameEditInput || !moduleDisplay || !moduleEditSelect || !editBtn || !saveBtn || !cancelBtn || !deleteBtn) {
            console.error("Missing one or more required elements inside the table row:", row);
            return; // Prevent errors if structure is wrong
        }
        // Get data attributes *after* ensuring the row exists
        let docId = row.dataset.docId || 'unknown';
        let docName = row.dataset.docName || 'unknown'; // Use let to allow update
        let currentModule = row.dataset.docModule || 'unknown'; // Use let to allow update
        // --- Edit Button Click ---
        if (button.classList.contains('edit-btn')) {
            console.log(`EDIT action for ${docId}`);
            filenameDisplay.classList.add('hidden');
            filenameEditInput.classList.remove('hidden');
            filenameEditInput.value = '';
            moduleDisplay.classList.add('hidden');
            moduleEditSelect.classList.remove('hidden');
            moduleEditSelect.value = currentModule;
            editBtn.classList.add('hidden');
            deleteBtn.classList.add('hidden');
            saveBtn.classList.remove('hidden');
            cancelBtn.classList.remove('hidden');
            filenameEditInput.focus(); // Focus the file input now
        }
        // --- Cancel Button Click ---
        else if (button.classList.contains('cancel-btn')) {
            console.log(`CANCEL action for ${docId}`);
            filenameEditInput.classList.add('hidden');
            filenameEditInput.value = '';
            filenameDisplay.classList.remove('hidden'); // Show original name
            moduleEditSelect.classList.add('hidden');
            moduleDisplay.classList.remove('hidden'); // Show original module
            saveBtn.classList.add('hidden');
            cancelBtn.classList.add('hidden');
            editBtn.classList.remove('hidden');
            deleteBtn.classList.remove('hidden');
        }
        // --- Save Button Click ---
        else if (button.classList.contains('save-btn')) {
            console.log(`SAVE action for ${docId}`);
            const newModule = moduleEditSelect.value;
            const selectedFiles = filenameEditInput.files;
            const newFile = selectedFiles && selectedFiles.length > 0 ? selectedFiles[0] : null;
            let fileChanged = false, moduleChanged = false;
            // Update filename if a new file was selected
            if (newFile) {
                const newFileName = newFile.name;
                if (newFileName !== docName) {
                    filenameDisplay.textContent = newFileName;
                    row.dataset.docName = newFileName; // Update data-*
                    docName = newFileName; // Update local var for logs
                    fileChanged = true;
                    console.log(` > Filename updated visually to: ${newFileName}`);
                }
                else {
                    console.log(` > File selected, name is the same.`);
                }
            }
            else {
                console.log(` > No new file selected.`);
            }
            // Update module if changed
            if (newModule !== currentModule) {
                moduleDisplay.textContent = newModule;
                row.dataset.docModule = newModule; // Update data-*
                currentModule = newModule; // Update local var
                moduleChanged = true;
                console.log(` > Module updated visually to: ${newModule}`);
            }
            else {
                console.log(` > Module unchanged.`);
            }
            if (fileChanged || moduleChanged)
                console.log(` > Visual changes applied (temporary).`);
            else
                console.log(` > No visual changes detected.`);
            // Revert UI to display mode
            filenameEditInput.classList.add('hidden');
            filenameEditInput.value = '';
            filenameDisplay.classList.remove('hidden');
            moduleEditSelect.classList.add('hidden');
            moduleDisplay.classList.remove('hidden');
            saveBtn.classList.add('hidden');
            cancelBtn.classList.add('hidden');
            editBtn.classList.remove('hidden');
            deleteBtn.classList.remove('hidden');
        }
        // --- Delete Button Click ---
        else if (button.classList.contains('delete-btn')) {
            console.log(`DELETE action for ${docId}`);
            if (confirm(`Êtes-vous sûr de vouloir supprimer (visuellement) "${docName}"?`)) {
                row.remove();
                // Check if table is empty after removal, show message if so
                if (documentsTableBody.rows.length === 0) {
                    const tr = documentsTableBody.insertRow();
                    const td = tr.insertCell();
                    td.textContent = "Aucun document trouvé.";
                    td.colSpan = 3;
                    td.className = 'px-6 py-4 text-center text-sm text-gray-500';
                }
                console.log(` > Row for ${docId} removed visually.`);
            }
            else {
                console.log(` > Deletion cancelled.`);
            }
        }
    });
}
else {
    console.error("Table body 'documentsTableBody' not found for event delegation.");
}
// --- Initial Load ---
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadDocuments);
}
else {
    loadDocuments();
} // If already loaded, run directly
export {};

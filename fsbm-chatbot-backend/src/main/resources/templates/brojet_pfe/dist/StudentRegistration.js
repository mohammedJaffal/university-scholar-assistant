// src/StudentRegistration.ts
// --- Helper Functions (Defined ONCE at the top) ---
function showError(inputId, message) {
    const input = document.getElementById(inputId);
    if (!input)
        return;
    const formGroup = input.closest('.form-group');
    const errorSpan = formGroup === null || formGroup === void 0 ? void 0 : formGroup.querySelector('.error-message');
    input.classList.add('invalid');
    input.setAttribute('aria-invalid', 'true');
    if (errorSpan) {
        errorSpan.textContent = message;
        errorSpan.classList.add('visible');
        input.setAttribute('aria-describedby', errorSpan.id || '');
    }
}
function clearError(inputId) {
    const input = document.getElementById(inputId);
    if (!input)
        return;
    const formGroup = input.closest('.form-group');
    const errorSpan = formGroup === null || formGroup === void 0 ? void 0 : formGroup.querySelector('.error-message');
    input.classList.remove('invalid');
    input.removeAttribute('aria-invalid');
    input.removeAttribute('aria-describedby');
    if (errorSpan) {
        errorSpan.textContent = '';
        errorSpan.classList.remove('visible');
    }
}
// --- End Helper Functions ---
const form = document.getElementById("studentForm");
if (form) {
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        console.log("Student form submitted...");
        const nomInput = document.getElementById("nom");
        const prenomInput = document.getElementById("prenom");
        const emailInput = document.getElementById("email");
        const emailUniversitaireInput = document.getElementById("emailUniversitaire");
        const filiereInput = document.getElementById("filiere");
        const passwordInput = document.getElementById("password");
        const confirmPasswordInput = document.getElementById("confirmPassword");
        if (!nomInput || !prenomInput || !emailInput || !emailUniversitaireInput || !filiereInput || !passwordInput || !confirmPasswordInput) {
            console.error("Form elements missing!");
            alert("Erreur de formulaire.");
            return;
        }
        let isValid = true;
        const inputs = [nomInput, prenomInput, emailInput, emailUniversitaireInput, filiereInput, passwordInput, confirmPasswordInput];
        inputs.forEach(input => input ? clearError(input.id) : null); // Use helpers defined above
        // Validation logic...
        if (!nomInput.value.trim()) {
            showError("nom", "Le nom est requis.");
            isValid = false;
        }
        if (!prenomInput.value.trim()) {
            showError("prenom", "Le prénom est requis.");
            isValid = false;
        }
        if (!emailInput.value.trim()) {
            showError("email", 'L\'adresse email est requise.');
            isValid = false;
        }
        else if (!/\S+@\S+\.\S+/.test(emailInput.value)) {
            showError("email", 'Veuillez entrer une adresse email valide.');
            isValid = false;
        }
        if (!emailUniversitaireInput.value.trim()) {
            showError("emailUniversitaire", 'L\'email universitaire est requis.');
            isValid = false;
        }
        else if (!/\S+@\S+\.\S+/.test(emailUniversitaireInput.value)) {
            showError("emailUniversitaire", 'Veuillez entrer un email universitaire valide.');
            isValid = false;
        }
        if (!filiereInput.value) {
            showError("filiere", "Veuillez choisir une filière.");
            isValid = false;
        }
        const passwordValue = passwordInput.value;
        if (!passwordValue) {
            showError("password", "Le mot de passe est requis.");
            isValid = false;
        }
        else if (passwordValue.length < 8) {
            showError("password", "Le mot de passe doit contenir au moins 8 caractères.");
            isValid = false;
        }
        const confirmPasswordValue = confirmPasswordInput.value;
        if (!confirmPasswordValue) {
            showError("confirmPassword", "Veuillez confirmer le mot de passe.");
            isValid = false;
        }
        else if (passwordValue && passwordValue.length >= 8 && passwordValue !== confirmPasswordValue) {
            showError("confirmPassword", "Les mots de passe ne correspondent pas.");
            showError("password", "Les mots de passe ne correspondent pas.");
            isValid = false;
        }
        if (!isValid) {
            console.log("Student validation failed.");
            const firstInvalid = form.querySelector('.invalid');
            firstInvalid === null || firstInvalid === void 0 ? void 0 : firstInvalid.focus();
            return;
        }
        console.log("Student validation successful. Redirecting to Verify Email...");
        alert("Compte créé (simulation). Veuillez vérifier votre e-mail.");
        try {
            window.location.href = './VerifyEmail.html';
        }
        catch (error) {
            console.error("Redirection error:", error);
        }
    });
    const filiereSelect = document.getElementById("filiere");
    if (filiereSelect) { /* ... select change listener ... */
        filiereSelect.addEventListener('change', () => {
            if (filiereSelect.value) {
                filiereSelect.classList.remove('text-gray-500');
                filiereSelect.classList.add('text-signup-input-text');
                clearError('filiere');
            }
            else {
                filiereSelect.classList.add('text-gray-500');
                filiereSelect.classList.remove('text-signup-input-text');
            }
        });
        if (filiereSelect.value) {
            filiereSelect.classList.remove('text-gray-500');
            filiereSelect.classList.add('text-signup-input-text');
        }
    }
}
else {
    console.error("Formulaire Étudiant (studentForm) introuvable !");
}
export {};

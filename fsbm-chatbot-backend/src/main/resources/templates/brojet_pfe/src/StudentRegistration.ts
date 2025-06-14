// src/StudentRegistration.ts

// --- Helper Functions (Defined ONCE at the top) ---
function showError(inputId: string, message: string): void {
    const input = document.getElementById(inputId) as HTMLInputElement | HTMLSelectElement | null; if (!input) return;
    const formGroup = input.closest('.form-group'); const errorSpan = formGroup?.querySelector('.error-message') as HTMLSpanElement | null;
    input.classList.add('invalid'); input.setAttribute('aria-invalid', 'true');
    if (errorSpan) { errorSpan.textContent = message; errorSpan.classList.add('visible'); input.setAttribute('aria-describedby', errorSpan.id || ''); }
}
function clearError(inputId: string): void {
    const input = document.getElementById(inputId) as HTMLInputElement | HTMLSelectElement | null; if (!input) return;
    const formGroup = input.closest('.form-group'); const errorSpan = formGroup?.querySelector('.error-message') as HTMLSpanElement | null;
    input.classList.remove('invalid'); input.removeAttribute('aria-invalid'); input.removeAttribute('aria-describedby');
    if (errorSpan) { errorSpan.textContent = ''; errorSpan.classList.remove('visible'); }
}
// --- End Helper Functions ---


const form = document.getElementById("studentForm") as HTMLFormElement | null;
if (form) {
    form.addEventListener("submit", (e: SubmitEvent) => {
        e.preventDefault(); console.log("Student form submitted...");
        const nomInput = document.getElementById("nom") as HTMLInputElement | null;
        const prenomInput = document.getElementById("prenom") as HTMLInputElement | null;
        const emailInput = document.getElementById("email") as HTMLInputElement | null;
        const emailUniversitaireInput = document.getElementById("emailUniversitaire") as HTMLInputElement | null;
        const filiereInput = document.getElementById("filiere") as HTMLSelectElement | null;
        const passwordInput = document.getElementById("password") as HTMLInputElement | null;
        const confirmPasswordInput = document.getElementById("confirmPassword") as HTMLInputElement | null;
        if (!nomInput || !prenomInput || !emailInput || !emailUniversitaireInput || !filiereInput || !passwordInput || !confirmPasswordInput) { console.error("Form elements missing!"); alert("Erreur de formulaire."); return; }
        let isValid = true; const inputs = [nomInput, prenomInput, emailInput, emailUniversitaireInput, filiereInput, passwordInput, confirmPasswordInput];
        inputs.forEach(input => input ? clearError(input.id) : null); // Use helpers defined above

        // Validation logic...
        if (!nomInput.value.trim()) { showError("nom", "Le nom est requis."); isValid = false; }
        if (!prenomInput.value.trim()) { showError("prenom", "Le prénom est requis."); isValid = false; }
        if (!emailInput.value.trim()) { showError("email", 'L\'adresse email est requise.'); isValid = false; }
        else if (!/\S+@\S+\.\S+/.test(emailInput.value)) { showError("email", 'Veuillez entrer une adresse email valide.'); isValid = false; }
        if (!emailUniversitaireInput.value.trim()) { showError("emailUniversitaire", 'L\'email universitaire est requis.'); isValid = false; }
        else if (!/\S+@\S+\.\S+/.test(emailUniversitaireInput.value)) { showError("emailUniversitaire", 'Veuillez entrer un email universitaire valide.'); isValid = false; }
        if (!filiereInput.value) { showError("filiere", "Veuillez choisir une filière."); isValid = false; }
        const passwordValue = passwordInput.value;
        if (!passwordValue) { showError("password", "Le mot de passe est requis."); isValid = false; }
        else if (passwordValue.length < 8) { showError("password", "Le mot de passe doit contenir au moins 8 caractères."); isValid = false; }
        const confirmPasswordValue = confirmPasswordInput.value;
        if (!confirmPasswordValue) { showError("confirmPassword", "Veuillez confirmer le mot de passe."); isValid = false; }
        else if (passwordValue && passwordValue.length >= 8 && passwordValue !== confirmPasswordValue) { showError("confirmPassword", "Les mots de passe ne correspondent pas."); showError("password", "Les mots de passe ne correspondent pas."); isValid = false; }

        if (!isValid) { console.log("Student validation failed."); const firstInvalid = form.querySelector('.invalid') as HTMLElement | null; firstInvalid?.focus(); return; }
        console.log("Student validation successful. Redirecting to Verify Email...");
        alert("Compte créé (simulation). Veuillez vérifier votre e-mail.");
        try { window.location.href = './VerifyEmail.html'; } catch (error) { console.error("Redirection error:", error); }
    });
    const filiereSelect = document.getElementById("filiere") as HTMLSelectElement | null;
    if (filiereSelect) { /* ... select change listener ... */
        filiereSelect.addEventListener('change', () => {
            if (filiereSelect.value) { filiereSelect.classList.remove('text-gray-500'); filiereSelect.classList.add('text-signup-input-text'); clearError('filiere'); }
            else { filiereSelect.classList.add('text-gray-500'); filiereSelect.classList.remove('text-signup-input-text'); }
        });
        if (filiereSelect.value) { filiereSelect.classList.remove('text-gray-500'); filiereSelect.classList.add('text-signup-input-text'); }
    }
} else { console.error("Formulaire Étudiant (studentForm) introuvable !"); }

// ★★★ REMOVED duplicate helper function definitions from here ★★★

export {}; // Keep this to ensure it's treated as a module
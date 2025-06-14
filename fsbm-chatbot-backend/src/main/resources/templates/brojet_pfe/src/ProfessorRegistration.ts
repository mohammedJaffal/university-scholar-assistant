// src/ProfessorRegistration.ts

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


const form = document.getElementById("registerForm") as HTMLFormElement | null;

if (form) {
    form.addEventListener("submit", (e: SubmitEvent) => {
        e.preventDefault(); console.log("Professor form submitted...");
        const nomInput = document.getElementById("nom") as HTMLInputElement | null;
        const prenomInput = document.getElementById("prenom") as HTMLInputElement | null;
        const emailInput = document.getElementById("email") as HTMLInputElement | null;
        const modulesInput = document.getElementById("modules") as HTMLInputElement | null;
        const passwordInput = document.getElementById("password") as HTMLInputElement | null;
        const confirmPasswordInput = document.getElementById("confirmPassword") as HTMLInputElement | null;

        if (!nomInput || !prenomInput || !emailInput || !modulesInput || !passwordInput || !confirmPasswordInput) { console.error("Form elements missing!"); alert("Erreur de formulaire."); return; }
        let isValid = true; const inputs = [nomInput, prenomInput, emailInput, modulesInput, passwordInput, confirmPasswordInput];
        inputs.forEach(input => input ? clearError(input.id) : null); // Use helpers defined above

        // Validation logic...
        if (!nomInput.value.trim()) { showError("nom", "Le nom est requis."); isValid = false; }
        if (!prenomInput.value.trim()) { showError("prenom", "Le prénom est requis."); isValid = false; }
        if (!emailInput.value.trim()) { showError("email", 'L\'email est requis.'); isValid = false; }
        else if (!/\S+@\S+\.\S+/.test(emailInput.value)) { showError("email", 'Veuillez entrer une adresse email valide.'); isValid = false; }
        if (!modulesInput.value.trim()) { showError("modules", "Veuillez indiquer le(s) module(s)."); isValid = false; }
        const passwordValue = passwordInput.value;
        if (!passwordValue) { showError("password", "Le mot de passe est requis."); isValid = false; }
        else if (passwordValue.length < 8) { showError("password", "Le mot de passe doit contenir au moins 8 caractères."); isValid = false; }
        const confirmPasswordValue = confirmPasswordInput.value;
        if (!confirmPasswordValue) { showError("confirmPassword", "Veuillez confirmer le mot de passe."); isValid = false; }
        else if (passwordValue && passwordValue.length >= 8 && passwordValue !== confirmPasswordValue) { showError("confirmPassword", "Les mots de passe ne correspondent pas."); showError("password", "Les mots de passe ne correspondent pas."); isValid = false; }

        if (!isValid) { console.log("Professor validation failed."); const firstInvalid = form.querySelector('.invalid') as HTMLElement | null; firstInvalid?.focus(); return; }
        console.log("Professor validation successful. Redirecting to Verify Email...");
        alert("Compte créé (simulation). Veuillez vérifier votre e-mail.");
        try { window.location.href = './VerifyEmail.html'; } catch (error) { console.error("Redirection error:", error); }
    });
} else { console.error("Formulaire Professeur (registerForm) introuvable !"); }
export {}; // Keep this to ensure it's treated as a module
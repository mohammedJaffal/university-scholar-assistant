// src/VerifyEmail.ts
// --- Helper Functions ---
function showError(inputId, message) { }
function clearError(inputId) { }
// --- End Helper Functions ---
console.log("Verify Email script loaded.");
const verifyForm = document.getElementById('verifyForm');
const verificationCodeInput = document.getElementById('verificationCode');
const resendCodeBtn = document.getElementById('resendCodeBtn');
const userEmailPlaceholder = document.getElementById('userEmailPlaceholder');
// Optional: Get email from previous page (if passed via query params or local storage)
// Example using localStorage (set this in ProfessorRegistration.ts and StudentRegistration.ts)
// const userEmail = localStorage.getItem('registrationEmail');
// if (userEmailPlaceholder && userEmail) {
//     userEmailPlaceholder.textContent = userEmail;
// } else if (userEmailPlaceholder) {
//     userEmailPlaceholder.textContent = 'votre adresse e-mail'; // Fallback
// }
if (verifyForm && verificationCodeInput) {
    verifyForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const code = verificationCodeInput.value.trim();
        clearError('verificationCode');
        let isValidCode = true;
        if (!code) {
            showError("verificationCode", "Veuillez entrer le code de vérification.");
            isValidCode = false;
        }
        else if (code.length !== 6 || !/^\d+$/.test(code)) {
            showError("verificationCode", "Le code doit être composé de 6 chiffres.");
            isValidCode = false;
        }
        if (!isValidCode) {
            verificationCodeInput.focus();
            return;
        }
        console.log("Code entré:", code);
        // --- !!! --- MODIFICATION HERE --- !!! ---
        // SIMULATION: Assume verification is successful and the user is a PROFESSOR.
        // In a real app, the backend would verify the code AND know the user's role.
        alert("Code validé (Simulation). Compte activé. Redirection vers la page de gestion des documents.");
        try {
            // Redirect PROFESSOR to the document management page
            window.location.href = './GérerDocument.html'; // <-- CHANGED REDIRECT DESTINATION
            // Optional: Clear any stored registration info after successful verification
            // localStorage.removeItem('registrationEmail');
            // localStorage.removeItem('pendingVerificationRole'); // If using role tracking simulation
        }
        catch (error) {
            console.error("Redirection error after verification:", error);
            alert("Erreur lors de la redirection."); // User feedback
        }
        // --- !!! --- END OF MODIFICATION --- !!! ---
    });
}
else {
    console.error("Formulaire de vérification ou champ de code introuvable.");
}
if (resendCodeBtn) {
    resendCodeBtn.addEventListener('click', () => {
        console.log("Demande de renvoi du code...");
        // TODO: Implement actual resend code logic (API call)
        alert("Un nouveau code a été envoyé (Simulation).");
        resendCodeBtn.disabled = true;
        resendCodeBtn.textContent = "Code renvoyé...";
        resendCodeBtn.classList.add('opacity-50', 'cursor-not-allowed');
        setTimeout(() => {
            resendCodeBtn.disabled = false;
            resendCodeBtn.textContent = "Renvoyer le code";
            resendCodeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        }, 30000); // 30 seconds cooldown
    });
}
export {};

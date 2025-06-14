// src/VerifyEmail.ts
// --- Helper Functions ---
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
 const submitBntForm=document.getElementById("validateBtn");
 const inputVerfication=document.getElementById("verificationCode");
let totalTime = 2 * 60;
    const timerSpan = document.getElementById("timer");
    function updateTimer() {
        const minutes = Math.floor(totalTime / 60).toString().padStart(2, '0');
        const seconds = (totalTime % 60).toString().padStart(2, '0');
        timerSpan.textContent = `${minutes}:${seconds}`;
        if (totalTime <= 0) {
            clearInterval(timerInterval);
            timerSpan.textContent = "00:00";
            timerSpan.style.color = "red";
            inputVerfication.disabled = true;
            submitBntForm.disabled=true;
        }
        totalTime--;
}
const timerInterval = setInterval(updateTimer, 1000);
updateTimer();
const verifyForm = document.getElementById('verifyForm');
const verificationCodeInput = document.getElementById('verificationCode');
const resendCodeBtn = document.getElementById('resendCodeBtn');
const userEmailPlaceholder = document.getElementById('userEmailPlaceholder');
// Optional: Get email from previous page (if passed via query params or local storage)
// Example using localStorage (set this in ProfessorRegistration.ts and StudentRegistration.ts)
 const userSimpleData = JSON.parse(sessionStorage.getItem('userSimpleData') || '');
 if (userEmailPlaceholder && userSimpleData) {
     userEmailPlaceholder.textContent = getEmailTextString(userSimpleData.email);
 } else if (userEmailPlaceholder) {
     userEmailPlaceholder.textContent = 'votre adresse e-mail';  Fallback
 }
 if ((Date.parse(userSimpleData.verificationCodeExpiry) + 3 * 60 * 1000 < Date.now())) {
   window.location.href = './index.html';
}else{
    console.log(true);
    
}
 function getEmailTextString(email) {
    const [localPart, domain] = email.split("@");
    const start = localPart.slice(0, 4);
    const end = localPart.slice(-2);
    const stars = "*".repeat(localPart.length - (start.length + end.length));
    return `${start}${stars}${end}@${domain}`;
}

if (verifyForm && verificationCodeInput) {
    verifyForm.addEventListener('submit', async (e) => {
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
            if (totalTime > 0){
                const Data= {
                nom: userSimpleData.nom,
                prenom: userSimpleData.prenom,
                email: userSimpleData.email,
                role: userSimpleData.role,
                verificationCode: code
            };
            try {
            const response = await fetch("http://localhost:8443/api/auth/verify-email", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(Data)
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || "Erreur d'inscription");
            }
            const data = await response.json();
            if(data.statue == true){
                verificationCodeInput.style.border="green";
                window.location.href = './index.html';
            }
            
        }
        catch (err) {
            console.error("Erreur d'inscription :", err.message);
            alert("Erreur : " + err.message);
        }
        }
        // --- !!! --- MODIFICATION HERE --- !!! ---
        // SIMULATION: Assume verification is successful and the user is a PROFESSOR.
        // In a real app, the backend would verify the code AND know the user's role.
        alert("Code validé (Simulation). Compte activé. Redirection vers la page de gestion des documents.");
      
        // --- !!! --- END OF MODIFICATION --- !!! ---
    });
}
else {
    console.error("Formulaire de vérification ou champ de code introuvable.");
}
if (resendCodeBtn) {
    resendCodeBtn.addEventListener('click',async () => {
        resendCodeBtn.disabled = true;
        resendCodeBtn.textContent = "Code renvoyé...";
        resendCodeBtn.classList.add('opacity-50', 'cursor-not-allowed');
         const Data= {
                nom: userSimpleData.nom,
                prenom: userSimpleData.prenom,
                email: userSimpleData.email,
                role: userSimpleData.role,
        };
        const response = await fetch("http://localhost:8443/api/auth/resend-verification-code", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(Data)
            });
         if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || "Erreur d'inscription");
            }
        const data = await response.json();
        if(data.statue){
            resendCodeBtn.disabled = false;
            resendCodeBtn.textContent = "Renvoyer le code";
            resendCodeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            inputVerfication.disabled=false;
            submitBntForm.disabled=false;
            totalTime = 2 * 60;
        }
    });
}
export {};

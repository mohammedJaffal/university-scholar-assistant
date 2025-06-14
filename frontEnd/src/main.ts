// src/main.ts - Manages index.html (Login Page)

document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm') as HTMLFormElement | null;
    const emailInput = document.getElementById('email') as HTMLInputElement | null;
    const passwordInput = document.getElementById('password') as HTMLInputElement | null;
    const registerStudentBtn = document.getElementById('registerStudentBtn') as HTMLButtonElement | null;
    const registerProfBtn = document.getElementById('registerProfBtn') as HTMLButtonElement | null;

    if (loginForm && emailInput && passwordInput) {
        loginForm.addEventListener('submit', (event: SubmitEvent) => {
            event.preventDefault();
            const email = emailInput.value;
            const password = passwordInput.value;

            // Basic frontend validation (optional but good practice)
            if (!email || !password) {
                alert('Veuillez remplir les champs Email et Mot de passe.');
                return; // Stop submission
            }

            console.log('Tentative de connexion (simulation): Email:', email);

            // --- !!! --- MODIFICATION HERE --- !!! ---
            // SIMULATION: Assume login is successful and the user is a PROFESSOR.
            // In a real app, you would send email/password to the backend.
            // The backend would verify credentials and return the user's role.
            // Based on the role, you would redirect.

            // Simulate successful login for a Professor
            alert('Connexion réussie (simulation). Redirection vers la page de gestion des documents.');
            try {
                // Redirect PROFESSOR to the document management page
                window.location.href = './GérerDocument.html'; // <-- ADDED REDIRECT
            } catch (error) {
                 console.error("Redirection error after login:", error);
                 alert("Erreur lors de la redirection."); // User feedback
            }
            // --- !!! --- END OF MODIFICATION --- !!! ---

            // TODO: Replace simulation with actual backend authentication call
            /*
            fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            })
            .then(response => {
                if (!response.ok) throw new Error('Login failed');
                return response.json();
            })
            .then(userData => {
                // Assuming backend returns user data including role, e.g., { role: 'professor', token: '...' }
                if (userData.role === 'professor') {
                     // Store token if needed (e.g., localStorage.setItem('authToken', userData.token);)
                     window.location.href = './GérerDocument.html';
                } else if (userData.role === 'student') {
                     // Store token if needed
                     window.location.href = './StudentDashboard.html'; // Or wherever students go
                } else {
                    throw new Error('Unknown user role');
                }
            })
            .catch(error => {
                console.error('Login error:', error);
                alert('Email ou mot de passe incorrect.'); // Provide user feedback
            });
            */
        });
    } else { console.warn("Login form or input elements not found!"); }

    // Navigation buttons remain the same
    if (registerStudentBtn) {
        registerStudentBtn.addEventListener('click', () => { window.location.href = './StudentSignup.html'; });
    }
    if (registerProfBtn) {
        registerProfBtn.addEventListener('click', () => { window.location.href = './TeacherSignup.html'; });
    }
  });

// export {}; // Not strictly necessary if not importing/exporting anything else
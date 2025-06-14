"use strict";
// src/main.ts - Manages index.html (Login Page)
document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const registerStudentBtn = document.getElementById('registerStudentBtn');
    const registerProfBtn = document.getElementById('registerProfBtn');
    if (loginForm && emailInput && passwordInput) {
        loginForm.addEventListener('submit', (event) => {
            event.preventDefault();
            const email = emailInput.value;
            const password = passwordInput.value;
            // Basic frontend validation (optional but good practice)
            if (!email || !password) {
                alert('Veuillez remplir les champs Email et Mot de passe.');
                return; // Stop submission
            }
            const user={ email:email, 
                password:password 
            }
            fetch('http://localhost:8443/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(user)
            })
            .then(response => {
                if (!response.ok) throw new Error('Login failed');
                return response.json();
            })
            .then(userData => {
                localStorage.setItem('accessToken', userData.accessToken);
                localStorage.setItem('tokenType', userData.tokenType);
                const role = userData.role.toUpperCase();
                if (role === 'ROLE_PROFESSOR') {
                    window.location.href = './Page_de_Chat.html';
                } else if (role === 'ROLE_STUDENT') {
                    window.location.href = './Page_de_Chat.html';
                } else {
                    throw new Error('Unknown user role');
                }
            })
            .catch(error => {
                console.error('Login error:', error);
                alert('Email ou mot de passe incorrect.');
            });

           
        });
    }
    else {
        console.warn("Login form or input elements not found!");
    }
    // Navigation buttons remain the same
    if (registerStudentBtn) {
        registerStudentBtn.addEventListener('click', () => { window.location.href = './StudentSignup.html'; });
    }
    if (registerProfBtn) {
        registerProfBtn.addEventListener('click', () => { window.location.href = './TeacherSignup.html'; });
    }
});
// export {}; // Not strictly necessary if not importing/exporting anything else

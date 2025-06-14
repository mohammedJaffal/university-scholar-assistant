package com.fsbmchatbot.fsbmchatbotbackend.controller;

import com.fsbmchatbot.fsbmchatbotbackend.dto.*;
import com.fsbmchatbot.fsbmchatbotbackend.model.Module;
import com.fsbmchatbot.fsbmchatbotbackend.security.UserDetailsImpl;
import com.fsbmchatbot.fsbmchatbotbackend.service.AuthService;
import jakarta.validation.Valid;

import java.util.ArrayList;
import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
@CrossOrigin(origins = "*", maxAge = 3600)  // En dev, autorise toutes origines. En prod, restreindre ici.
public class AuthController {

    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/register/student")
    public ResponseEntity<ArrayList<MessageResponse>> registerStudent(@Valid @RequestBody StudentRegisterRequest request) {
        ArrayList<MessageResponse> response = authService.registerStudent(request);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/register/professor")
    public ResponseEntity<ArrayList<MessageResponse>> registerProfessor(@Valid @RequestBody ProfessorRegisterRequest request) {
          
        ArrayList<MessageResponse> response = authService.registerProfessor(request);
        return ResponseEntity.ok(response);
    }
    @PostMapping("/GetAllModules")
    public ResponseEntity<List<Module>> GetAllModules() {
        List<Module> modules= authService.GetAllModule();
        return ResponseEntity.ok(modules);
    }
    @PostMapping("/login")
    public ResponseEntity<AuthResponse> authenticateUser(@Valid @RequestBody LoginRequest request) {
        System.out.println("test valide ////");
        System.out.println("test valide ////");
        System.out.println("test valide ////");
        System.out.println("test valide ////");
        System.out.println(request);
        AuthResponse authResponse = authService.loginUser(request);
        return ResponseEntity.ok(authResponse);
    }

    @PostMapping("/verify-email")
    public ResponseEntity<MessageResponse> verifyEmail(@Valid @RequestBody VerifyEmailRequest verifyEmailRequest) {
        System.out.println(verifyEmailRequest);
        MessageResponse response = authService.verifyEmail(verifyEmailRequest);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/resend-verification-code")
    public ResponseEntity<MessageResponse> resendVerificationCode(@Valid @RequestBody VerifyEmailRequest verifyEmailRequestl) {
        System.out.println(verifyEmailRequestl);
        if (verifyEmailRequestl.getEmail() == null || verifyEmailRequestl.getEmail().trim().isEmpty()) {
            return ResponseEntity.badRequest().body(new MessageResponse("Email parameter is required."));
        }
        MessageResponse response = authService.resendVerificationCode(verifyEmailRequestl);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/logout")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<MessageResponse> logoutUser(@AuthenticationPrincipal UserDetailsImpl currentUser) {
        // Si tu as une méthode de logout côté service pour invalider le token, appelle-la ici
        // authService.logout(currentUser);
        return ResponseEntity.ok(new MessageResponse("Logged out successfully."));
    }
}

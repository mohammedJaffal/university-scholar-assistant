package com.fsbmchatbot.fsbmchatbotbackend.dto;

import com.fsbmchatbot.fsbmchatbotbackend.model.Role;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AuthResponse {
    private String accessToken;
    private String tokenType = "Bearer";
    private Long userId;
    private String email;
    private String nom;
    private String prenom;
    private Role role;
    private String filiere; // Filiere name, can be null
    private boolean isActive;
    private boolean isVerified; // To inform frontend if verification step is next

    public AuthResponse(String accessToken, Long userId, String email, String nom, String prenom, Role role, String filiere, boolean isActive, boolean isVerified) {
        this.accessToken = accessToken;
        this.userId = userId;
        this.email = email;
        this.nom = nom;
        this.prenom = prenom;
        this.role = role;
        this.filiere = filiere;
        this.isActive = isActive;
        this.isVerified = isVerified;
    }
}
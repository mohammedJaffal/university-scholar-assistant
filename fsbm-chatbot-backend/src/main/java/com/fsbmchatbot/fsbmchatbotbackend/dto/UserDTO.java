package com.fsbmchatbot.fsbmchatbotbackend.dto;

import com.fsbmchatbot.fsbmchatbotbackend.model.Role;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserDTO {
    private Long id;
    private String nom;
    private String prenom;
    private String email;
    private String emailUniversitaire;
    private Role role;
    private String filiereNom; // Name of the filiere
    private boolean isActive;
    private boolean isVerified;
    private LocalDateTime createdAt;
}
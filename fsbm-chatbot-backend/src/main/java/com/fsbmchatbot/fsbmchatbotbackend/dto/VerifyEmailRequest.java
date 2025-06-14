package com.fsbmchatbot.fsbmchatbotbackend.dto;
import com.fsbmchatbot.fsbmchatbotbackend.model.Role;
import lombok.Data;

@Data
public class VerifyEmailRequest {
    private String nom;
    private String prenom;
    private String email;
    private Role role;
    private String verificationCode;
}


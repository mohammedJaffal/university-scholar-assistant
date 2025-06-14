package com.fsbmchatbot.fsbmchatbotbackend.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class ProfessorRegisterRequest {

    @NotBlank(message = "Nom cannot be blank")
    @Size(min = 2, max = 50, message = "Nom must be between 2 and 50 characters")
    private String nom;

    @NotBlank(message = "Prénom cannot be blank")
    @Size(min = 2, max = 50, message = "Prénom must be between 2 and 50 characters")
    private String prenom;

    @NotBlank(message = "Email cannot be blank")
    @Email(message = "Email should be valid")
    @Size(max = 100)
    private String email;

    @NotBlank(message = "Password cannot be blank")
    @Size(min = 8, max = 40, message = "Password must be between 8 and 40 characters")
    private String password;

    @NotBlank(message = "Confirm password cannot be blank")
    private String confirmPassword;

    @NotNull(message = "Module ID cannot be null")
    private Long module;  // ou String si tu envoies le nom
}

package com.fsbmchatbot.fsbmchatbotbackend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class FiliereDTO {
    private Long id;

    @NotBlank(message = "Filiere name cannot be blank")
    @Size(max = 50)
    private String nom;

    @Size(max = 255)
    private String description;
}
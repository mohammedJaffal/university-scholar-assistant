package com.fsbmchatbot.fsbmchatbotbackend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ModuleDTO {
    private Long id; // Present in response, not required in request for creation

    @NotBlank(message = "Module name cannot be blank")
    @Size(max = 100)
    private String nom;

    @Size(max = 255)
    private String description;
    @Size(max = 255)
    private String code_module;
}
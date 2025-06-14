package com.fsbmchatbot.fsbmchatbotbackend.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class DocumentResponse {
    private Long id;
    private String originalFileName;
    private String moduleNom;
    private Long moduleId;
    private String uploadedByFullName; // e.g., "Nom Prénom Professeur"
    private LocalDateTime uploadedAt;
    private String downloadUrl; // We can construct this in the service
}